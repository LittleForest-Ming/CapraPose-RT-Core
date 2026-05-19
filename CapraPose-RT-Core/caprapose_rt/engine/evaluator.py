"""Evaluation helpers and lightweight research metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

try:  # pragma: no cover - simple dependency fallback
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable=None, **kwargs):
        return iterable

from caprapose_rt.constants import OKS_SIGMAS
from caprapose_rt.engine.data_loading import iterate_batches
from caprapose_rt.utils.device import create_amp_autocast
from caprapose_rt.utils.geometry import crop_coords_to_image


def _compute_single_metrics(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    visibility: torch.Tensor,
    bbox: torch.Tensor,
    area: torch.Tensor,
    confidence: torch.Tensor,
    pck_threshold: float,
) -> dict[str, float]:
    visible = visibility > 0
    if visible.sum() == 0:
        return {"pck": 0.0, "nme": 0.0, "oks": 0.0, "confidence": 0.0}
    if len(OKS_SIGMAS) != pred_coords.shape[0]:
        raise ValueError(
            f"Expected {pred_coords.shape[0]} OKS sigmas, got {len(OKS_SIGMAS)}."
        )

    distances = torch.linalg.norm(pred_coords - gt_coords, dim=-1)
    normalization = max(float(bbox[2]), float(bbox[3]), 1.0)
    pck = ((distances[visible] / normalization) < pck_threshold).float().mean().item()
    nme = (distances[visible] / normalization).mean().item()

    sigmas = torch.tensor(OKS_SIGMAS, dtype=pred_coords.dtype, device=pred_coords.device)
    scale = max(float(area), 1.0) ** 0.5
    oks_term = torch.exp(
        -(distances**2) / (2.0 * (scale * sigmas).clamp_min(1e-6) ** 2)
    )
    oks = oks_term[visible].mean().item()
    mean_confidence = confidence[visible].mean().item()
    return {"pck": pck, "nme": nme, "oks": oks, "confidence": mean_confidence}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    input_size: tuple[int, int],
    pck_threshold: float = 0.2,
    output_path: str | None = None,
    summary_path: str | None = None,
    amp_cfg: dict | None = None,
    use_cuda_prefetch: bool = True,
    max_batches: int | None = None,
    sync_timing: bool = False,
    channels_last: bool = False,
    logger=None,
) -> dict[str, float]:
    """Evaluate the model and optionally dump predictions to JSON."""

    model.eval()
    metrics = {"pck": [], "nme": [], "oks": [], "confidence": []}
    predictions: list[dict] = []
    data_times: list[float] = []
    fetch_times: list[float] = []
    transfer_times: list[float] = []
    compute_times: list[float] = []
    step_times: list[float] = []
    total_images = 0
    eval_start = time.perf_counter()

    batch_iterator = iterate_batches(
        data_loader=data_loader,
        device=device,
        use_cuda_prefetch=use_cuda_prefetch,
        channels_last=channels_last,
    )
    for step, (batch, batch_timing) in enumerate(
        tqdm(batch_iterator, desc="eval", leave=False, total=len(data_loader)),
        start=1,
    ):
        step_start = time.perf_counter()
        with create_amp_autocast(device=device, amp_cfg=amp_cfg or {}):
            outputs = model(batch["image"])
        pred_image = crop_coords_to_image(
            coords=outputs["refined_coords"],
            crop_box=batch["crop_box"],
            input_size=input_size,
        )
        if device.type == "cuda" and sync_timing:
            torch.cuda.synchronize(device)
        compute_time = time.perf_counter() - step_start
        data_times.append(float(batch_timing["data_time"]))
        fetch_times.append(float(batch_timing["fetch_time"]))
        transfer_times.append(float(batch_timing["transfer_time"]))
        compute_times.append(compute_time)
        step_times.append(float(batch_timing["data_time"]) + compute_time)
        total_images += pred_image.shape[0]

        for batch_index in range(pred_image.shape[0]):
            sample_metrics = _compute_single_metrics(
                pred_coords=pred_image[batch_index],
                gt_coords=batch["keypoints_image"][batch_index],
                visibility=batch["visibility"][batch_index],
                bbox=batch["bbox"][batch_index],
                area=batch["area"][batch_index],
                confidence=outputs["confidence"][batch_index].squeeze(-1),
                pck_threshold=pck_threshold,
            )
            for key in metrics:
                metrics[key].append(sample_metrics[key])

            predictions.append(
                {
                    "image_id": int(batch["image_id"][batch_index]),
                    "annotation_id": int(batch["annotation_id"][batch_index]),
                    "category_id": 1,
                    "file_name": batch["file_name"][batch_index],
                    "keypoints": pred_image[batch_index].detach().cpu().tolist(),
                    "confidence": outputs["confidence"][batch_index]
                    .squeeze(-1)
                    .detach()
                    .cpu()
                    .tolist(),
                    "score": float(sample_metrics["confidence"]),
                    "oks": sample_metrics["oks"],
                }
            )
        if max_batches is not None and step >= int(max_batches):
            break

    oks_scores = metrics["oks"]
    thresholds = [round(0.50 + 0.05 * idx, 2) for idx in range(10)]
    summary = {
        "PCK@0.2": _safe_mean(metrics["pck"]),
        "NME": _safe_mean(metrics["nme"]),
        "OKS": _safe_mean(metrics["oks"]),
        "mean_confidence": _safe_mean(metrics["confidence"]),
        "AP50": _safe_mean([score >= 0.50 for score in oks_scores]),
        "AP75": _safe_mean([score >= 0.75 for score in oks_scores]),
        "mAP": _safe_mean(
            [_safe_mean([score >= threshold for score in oks_scores]) for threshold in thresholds]
        ),
        "avg_data_time": _safe_mean(data_times),
        "avg_fetch_time": _safe_mean(fetch_times),
        "avg_transfer_time": _safe_mean(transfer_times),
        "avg_compute_time": _safe_mean(compute_times),
        "avg_step_time": _safe_mean(step_times),
        "total_images": int(total_images),
        "total_time": time.perf_counter() - eval_start,
    }
    summary["images_per_sec"] = (
        summary["total_images"] / max(summary["total_time"], 1e-6)
        if summary["total_images"] > 0
        else 0.0
    )

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"metrics": summary, "predictions": predictions}, handle, indent=2)

    if summary_path:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    if logger is not None:
        logger.info(
            "Evaluation timing | avg_data=%.4fs | avg_fetch=%.4fs | avg_h2d=%.4fs | avg_compute=%.4fs | avg_step=%.4fs | images/s=%.2f",
            summary["avg_data_time"],
            summary["avg_fetch_time"],
            summary["avg_transfer_time"],
            summary["avg_compute_time"],
            summary["avg_step_time"],
            summary["images_per_sec"],
        )

    return summary


def _safe_mean(values: list) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item())
