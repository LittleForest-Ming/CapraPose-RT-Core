"""Training loop for CapraPose-RT."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

try:  # pragma: no cover - simple dependency fallback
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable=None, **kwargs):
        return iterable

from caprapose_rt.engine.checkpoint import save_checkpoint
from caprapose_rt.engine.data_loading import iterate_batches
from caprapose_rt.engine.evaluator import evaluate_model
from caprapose_rt.utils.device import create_amp_autocast, create_grad_scaler


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class Trainer:
    """Simple single-process trainer suitable for initial experiments."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: torch.device,
        logger,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.runtime_cfg = config.get("runtime", {})
        self.work_dir = Path(config["work_dir"])
        self.artifact_cfg = config.get("artifacts", {})
        self.checkpoint_dir = self.work_dir / self.artifact_cfg.get("checkpoints_dir", "checkpoints")
        self.evaluation_dir = self.work_dir / self.artifact_cfg.get("evaluation_dir", "evaluation")
        self.metrics_dir = self.work_dir / self.artifact_cfg.get("metrics_dir", "metrics")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = self._build_optimizer(config["optimizer"])
        self.scheduler = self._build_scheduler(config["scheduler"])
        self.scaler = create_grad_scaler(device=device, amp_cfg=config.get("amp", {}))
        self.best_map = float("-inf")
        self.best_oks = float("-inf")
        self.best_epoch = 0
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.history_path = self.metrics_dir / "metrics_history.jsonl"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.work_dir / "experiment_summary.json"
        self.latest_eval_summary_path = self.evaluation_dir / "latest_val_metrics.json"
        self.best_eval_summary_path = self.evaluation_dir / "best_val_metrics.json"
        self.freeze_epochs = int(self.config["train"].get("freeze_detector_epochs", 0))
        self.freeze_modules = tuple(self.config["train"].get("freeze_detector_modules", ()))
        self._detector_frozen = False
        self.start_epoch = 1
        self._write_experiment_summary(status="initialized", current_epoch=0)

    def _build_optimizer(self, optimizer_cfg: dict):
        optimizer_type = optimizer_cfg.get("type", "AdamW")
        if optimizer_type != "AdamW":
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        return AdamW(
            self.model.parameters(),
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
        )

    def _build_scheduler(self, scheduler_cfg: dict):
        scheduler_type = scheduler_cfg.get("type", "StepLR")
        if scheduler_type != "StepLR":
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return StepLR(
            self.optimizer,
            step_size=int(scheduler_cfg["step_size"]),
            gamma=float(scheduler_cfg["gamma"]),
        )

    def fit(self) -> None:
        train_cfg = self.config["train"]
        total_epochs = int(train_cfg["epochs"])

        for epoch in range(self.start_epoch, total_epochs + 1):
            self._apply_epoch_freeze_state(epoch)
            start_time = time.time()
            train_metrics = self._train_one_epoch()
            epoch_time = time.time() - start_time
            train_images_per_sec = train_metrics["images"] / max(epoch_time, 1e-6)
            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | heatmap=%.4f | coord=%.4f | coord_cls=%.4f | structural=%.4f | data=%.4fs | fetch=%.4fs | h2d=%.4fs | compute=%.4fs | step=%.4fs | img/s=%.2f | %.2fs",
                epoch,
                total_epochs,
                train_metrics["loss"],
                train_metrics["heatmap_loss"],
                train_metrics["coord_loss"],
                train_metrics["coord_cls_loss"],
                train_metrics["structural_loss"],
                train_metrics["data_time"],
                train_metrics["fetch_time"],
                train_metrics["transfer_time"],
                train_metrics["compute_time"],
                train_metrics["step_time"],
                train_images_per_sec,
                epoch_time,
            )

            evaluation_summary = None
            should_validate = (
                self.val_loader is not None
                and (
                    epoch % int(train_cfg["val_interval"]) == 0
                    or epoch == total_epochs
                )
            )
            if should_validate:
                epoch_eval_path = self.evaluation_dir / f"epoch_{epoch:03d}_val_metrics.json"
                evaluation_summary = evaluate_model(
                    model=self.model,
                    data_loader=self.val_loader,
                    device=self.device,
                    input_size=tuple(self.config["dataset"]["input_size"]),
                    pck_threshold=float(self.config["eval"]["pck_threshold"]),
                    summary_path=str(epoch_eval_path),
                    amp_cfg=self.config.get("amp", {}),
                    use_cuda_prefetch=bool(self.config["eval"].get("cuda_prefetch", True)),
                    max_batches=self.config["eval"].get("max_batches"),
                    sync_timing=bool(self.runtime_cfg.get("sync_timing", False)),
                    channels_last=bool(self.runtime_cfg.get("channels_last", False)),
                    logger=self.logger,
                )
                self.logger.info("Validation metrics: %s", evaluation_summary)
                self.latest_eval_summary_path.write_text(
                    json.dumps(evaluation_summary, indent=2),
                    encoding="utf-8",
                )

            current_map = evaluation_summary["mAP"] if evaluation_summary is not None else None
            current_oks = (
                float(evaluation_summary.get("OKS", 0.0))
                if evaluation_summary is not None
                else None
            )
            is_best = (
                current_map is not None
                and (
                    current_map > self.best_map
                    or (
                        current_map == self.best_map
                        and current_oks is not None
                        and current_oks > self.best_oks
                    )
                )
            )
            if is_best:
                self.best_map = float(current_map)
                self.best_oks = float(current_oks) if current_oks is not None else float("-inf")
                self.best_epoch = epoch

            checkpoint_state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": self.config,
                "best_map": self.best_map,
                "best_oks": self.best_oks,
                "best_epoch": self.best_epoch,
            }
            checkpoint_interval = int(train_cfg.get("checkpoint_interval", 1))
            save_latest = (
                epoch % checkpoint_interval == 0
                or epoch == total_epochs
                or is_best
            )
            if save_latest:
                latest_checkpoint_path = save_checkpoint(
                    checkpoint_state,
                    self.checkpoint_dir,
                    "latest.pth",
                )
            else:
                latest_checkpoint_path = self.checkpoint_dir / "latest.pth"

            if is_best:
                best_checkpoint_path = save_checkpoint(checkpoint_state, self.checkpoint_dir, "best.pth")
                if evaluation_summary is not None:
                    self.best_eval_summary_path.write_text(
                        json.dumps(evaluation_summary, indent=2),
                        encoding="utf-8",
                    )
            else:
                best_checkpoint_path = self.checkpoint_dir / "best.pth"

            self._append_history(
                epoch=epoch,
                epoch_time=epoch_time,
                train_metrics=train_metrics,
                evaluation_summary=evaluation_summary,
                train_images_per_sec=train_images_per_sec,
            )
            self._write_experiment_summary(
                status="running" if epoch < total_epochs else "completed",
                current_epoch=epoch,
                latest_checkpoint_path=latest_checkpoint_path,
                best_checkpoint_path=best_checkpoint_path,
                latest_evaluation=evaluation_summary,
            )
            self.scheduler.step()

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        scaler_state = checkpoint.get("scaler")
        if scaler_state:
            self.scaler.load_state_dict(scaler_state)

        resumed_epoch = int(checkpoint.get("epoch", 0))
        self.start_epoch = resumed_epoch + 1
        self.best_map = float(checkpoint.get("best_map", self.best_map))
        self.best_oks = float(checkpoint.get("best_oks", self.best_oks))
        self.best_epoch = int(checkpoint.get("best_epoch", self.best_epoch))
        self.logger.info(
            "Resumed training state from %s | epoch=%d | next_epoch=%d | best_map=%.4f | best_oks=%s | best_epoch=%d",
            checkpoint_path,
            resumed_epoch,
            self.start_epoch,
            self.best_map if self.best_map != float("-inf") else float("nan"),
            f"{self.best_oks:.4f}" if self.best_oks != float("-inf") else "n/a",
            self.best_epoch,
        )
        self._write_experiment_summary(status="resumed", current_epoch=resumed_epoch)

    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        meters = {
            "loss": AverageMeter(),
            "heatmap_loss": AverageMeter(),
            "coord_loss": AverageMeter(),
            "coord_cls_loss": AverageMeter(),
            "structural_loss": AverageMeter(),
            "data_time": AverageMeter(),
            "fetch_time": AverageMeter(),
            "transfer_time": AverageMeter(),
            "compute_time": AverageMeter(),
            "step_time": AverageMeter(),
        }
        images_seen = 0

        log_interval = int(self.config["train"]["log_interval"])
        batch_iterator = iterate_batches(
            data_loader=self.train_loader,
            device=self.device,
            use_cuda_prefetch=bool(self.config["train"].get("cuda_prefetch", True)),
            channels_last=bool(self.runtime_cfg.get("channels_last", False)),
        )
        max_batches = self.config["train"].get("max_batches")
        for step, (batch, batch_timing) in enumerate(
            tqdm(batch_iterator, desc="train", leave=False, total=len(self.train_loader)),
            start=1,
        ):
            step_start = time.perf_counter()
            with create_amp_autocast(device=self.device, amp_cfg=self.config.get("amp", {})):
                outputs = self.model(batch["image"])
                loss_dict = self.model.compute_losses(outputs, batch, self.config["loss"])

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.device.type == "cuda" and bool(self.runtime_cfg.get("sync_timing", False)):
                torch.cuda.synchronize(self.device)
            compute_time = time.perf_counter() - step_start
            step_time = float(batch_timing["data_time"]) + compute_time

            batch_size = batch["image"].shape[0]
            images_seen += batch_size
            meters["loss"].update(float(loss_dict["loss"].item()), batch_size)
            meters["heatmap_loss"].update(float(loss_dict["heatmap_loss"].item()), batch_size)
            meters["coord_loss"].update(float(loss_dict["coord_loss"].item()), batch_size)
            meters["coord_cls_loss"].update(float(loss_dict.get("coord_cls_loss", torch.tensor(0.0)).item()), batch_size)
            meters["structural_loss"].update(float(loss_dict["structural_loss"].item()), batch_size)
            meters["data_time"].update(float(batch_timing["data_time"]), batch_size)
            meters["fetch_time"].update(float(batch_timing["fetch_time"]), batch_size)
            meters["transfer_time"].update(float(batch_timing["transfer_time"]), batch_size)
            meters["compute_time"].update(float(compute_time), batch_size)
            meters["step_time"].update(float(step_time), batch_size)

            if step % log_interval == 0:
                self.logger.info(
                    "step=%d | loss=%.4f | heatmap=%.4f | coord=%.4f | coord_cls=%.4f | structural=%.4f | data=%.4fs | fetch=%.4fs | h2d=%.4fs | compute=%.4fs | step=%.4fs",
                    step,
                    meters["loss"].average,
                    meters["heatmap_loss"].average,
                    meters["coord_loss"].average,
                    meters["coord_cls_loss"].average,
                    meters["structural_loss"].average,
                    meters["data_time"].average,
                    meters["fetch_time"].average,
                    meters["transfer_time"].average,
                    meters["compute_time"].average,
                    meters["step_time"].average,
                )
            if max_batches is not None and step >= int(max_batches):
                break

        result = {name: meter.average for name, meter in meters.items()}
        result["images"] = int(images_seen)
        return result

    def _append_history(
        self,
        epoch: int,
        epoch_time: float,
        train_metrics: dict[str, float],
        evaluation_summary: dict[str, float] | None,
        train_images_per_sec: float,
    ) -> None:
        record = {
            "epoch": int(epoch),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "epoch_time": float(epoch_time),
            "train": {key: float(value) for key, value in train_metrics.items()},
            "train_images_per_sec": float(train_images_per_sec),
            "val": (
                {key: float(value) for key, value in evaluation_summary.items()}
                if evaluation_summary is not None
                else None
            ),
            "best_map": float(self.best_map),
            "best_oks": float(self.best_oks) if self.best_oks != float("-inf") else None,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _write_experiment_summary(
        self,
        status: str,
        current_epoch: int,
        latest_checkpoint_path: Path | None = None,
        best_checkpoint_path: Path | None = None,
        latest_evaluation: dict[str, float] | None = None,
    ) -> None:
        summary = {
            "experiment_name": self.config["experiment_name"],
            "protocol_name": self.config.get("protocol", {}).get("name"),
            "status": status,
            "work_dir": str(self.work_dir),
            "seed": int(self.config["seed"]),
            "epochs_total": int(self.config["train"]["epochs"]),
            "current_epoch": int(current_epoch),
            "best_epoch": int(self.best_epoch),
            "best_map": float(self.best_map) if self.best_map != float("-inf") else None,
            "best_oks": float(self.best_oks) if self.best_oks != float("-inf") else None,
            "num_parameters": int(self.parameter_count),
            "paths": {
                "config_snapshot": str(self.work_dir / "resolved_config.json"),
                "train_log": str(self.work_dir / "train.log"),
                "history": str(self.history_path),
                "checkpoints_dir": str(self.checkpoint_dir),
                "evaluation_dir": str(self.evaluation_dir),
                "latest_checkpoint": (
                    str(latest_checkpoint_path) if latest_checkpoint_path is not None else str(self.checkpoint_dir / "latest.pth")
                ),
                "best_checkpoint": (
                    str(best_checkpoint_path) if best_checkpoint_path is not None else str(self.checkpoint_dir / "best.pth")
                ),
                "latest_eval_summary": str(self.latest_eval_summary_path),
                "best_eval_summary": str(self.best_eval_summary_path),
                "qualitative_dir": str(
                    self.work_dir / self.artifact_cfg.get("qualitative_dir", "qualitative")
                ),
            },
            "train": {
                "batch_size": int(self.config["train"]["batch_size"]),
                "num_workers": int(self.config["train"]["num_workers"]),
                "max_batches": self.config["train"].get("max_batches"),
            },
            "eval": {
                "batch_size": int(self.config["eval"]["batch_size"]),
                "num_workers": int(self.config["eval"]["num_workers"]),
                "max_batches": self.config["eval"].get("max_batches"),
            },
            "latest_evaluation": latest_evaluation,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _apply_epoch_freeze_state(self, epoch: int) -> None:
        if self.freeze_epochs <= 0 or not self.freeze_modules:
            return
        should_freeze = epoch <= self.freeze_epochs
        if should_freeze == self._detector_frozen:
            return
        for module_name in self.freeze_modules:
            module = getattr(self.model, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = not should_freeze
        state = "frozen" if should_freeze else "unfrozen"
        self.logger.info(
            "Stage-2 detector modules %s for epoch %d/%d: %s",
            state,
            epoch,
            int(self.config["train"]["epochs"]),
            ", ".join(self.freeze_modules),
        )
        self._detector_frozen = should_freeze
