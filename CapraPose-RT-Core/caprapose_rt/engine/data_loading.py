"""GPU-oriented dataloader helpers."""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from caprapose_rt.engine.common import move_batch_to_device


def build_dataloader(
    dataset,
    loader_cfg: dict,
    shuffle: bool,
    seed: int | None = None,
    worker_init_fn=None,
    drop_last: bool = False,
    logger=None,
    split: str = "train",
):
    """Build a dataloader with GPU-friendly defaults."""

    start_time = time.perf_counter()
    num_workers = int(loader_cfg["num_workers"])
    persistent_workers = bool(loader_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(loader_cfg.get("prefetch_factor", 2))

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    sampler = None
    use_hard_case_sampler = (
        split == "train"
        and hasattr(dataset, "get_sampling_weights")
        and (
            float(loader_cfg.get("hard_case_oversample_factor", 1.0)) > 1.0
            or float(loader_cfg.get("hard_case_priority_factor", 1.0)) > 1.0
            or float(loader_cfg.get("ambiguous_sample_weight", 1.0)) != 1.0
        )
    )
    if use_hard_case_sampler:
        sampling_weights = dataset.get_sampling_weights(
            hard_case_oversample_factor=float(loader_cfg.get("hard_case_oversample_factor", 1.0)),
            hard_case_priority_factor=float(loader_cfg.get("hard_case_priority_factor", 1.0)),
            ambiguous_sample_weight=float(loader_cfg.get("ambiguous_sample_weight", 1.0)),
        )
        if (
            sampling_weights.size > 0
            and float(sampling_weights.max() - sampling_weights.min()) > 1e-6
        ):
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sampling_weights, dtype=torch.double),
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )

    data_loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(loader_cfg["batch_size"]),
        "shuffle": bool(shuffle and sampler is None),
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", True)),
        "drop_last": drop_last,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }
    if sampler is not None:
        data_loader_kwargs["sampler"] = sampler
    if num_workers > 0:
        data_loader_kwargs["persistent_workers"] = persistent_workers
        data_loader_kwargs["prefetch_factor"] = prefetch_factor

    data_loader = DataLoader(**data_loader_kwargs)
    creation_time = time.perf_counter() - start_time
    if logger is not None:
        if getattr(dataset, "cache_images", False) == "ram" and num_workers > 0:
            logger.warning(
                "RAM image caching is enabled with num_workers=%d. "
                "Worker processes may duplicate cached host memory.",
                num_workers,
            )
        logger.info(
            "Built %s dataloader in %.2fs | batch_size=%d | workers=%d | pin_memory=%s | persistent_workers=%s | prefetch_factor=%s",
            split,
            creation_time,
            int(loader_cfg["batch_size"]),
            num_workers,
            bool(loader_cfg.get("pin_memory", True)),
            persistent_workers if num_workers > 0 else False,
            prefetch_factor if num_workers > 0 else "n/a",
        )
        if sampler is not None:
            logger.info(
                "Enabled hard-case weighted sampling for %s | oversample_factor=%.2f | priority_factor=%.2f | ambiguous_sample_weight=%.2f",
                split,
                float(loader_cfg.get("hard_case_oversample_factor", 1.0)),
                float(loader_cfg.get("hard_case_priority_factor", 1.0)),
                float(loader_cfg.get("ambiguous_sample_weight", 1.0)),
            )
    return data_loader


class CUDAPrefetcher:
    """Move the next batch to CUDA on a separate stream."""

    def __init__(self, data_loader, device: torch.device, channels_last: bool = False) -> None:
        if device.type != "cuda":
            raise ValueError("CUDAPrefetcher requires a CUDA device.")
        self.data_loader = data_loader
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream(device=device)
        self.iterator = None
        self.next_batch = None
        self.next_timing = _empty_batch_timing()

    def __iter__(self):
        self.iterator = iter(self.data_loader)
        self.next_batch = None
        self._preload()
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(device=self.device)
        current_stream.wait_stream(self.stream)
        batch = self.next_batch
        batch_timing = self.next_timing
        _record_stream(batch, current_stream)
        self._preload()
        return batch, batch_timing

    def _preload(self) -> None:
        fetch_start = time.perf_counter()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            self.next_timing = _empty_batch_timing()
            return
        fetch_time = time.perf_counter() - fetch_start

        transfer_start = time.perf_counter()
        with torch.cuda.stream(self.stream):
            self.next_batch = move_batch_to_device(
                batch=batch,
                device=self.device,
                non_blocking=True,
                channels_last=self.channels_last,
            )
        transfer_time = time.perf_counter() - transfer_start
        self.next_timing = {
            "fetch_time": fetch_time,
            "transfer_time": transfer_time,
            "data_time": fetch_time + transfer_time,
        }


def iterate_batches(
    data_loader,
    device: torch.device,
    use_cuda_prefetch: bool,
    channels_last: bool = False,
):
    """Yield `(batch, batch_timing)` pairs for training and evaluation loops."""

    if use_cuda_prefetch and device.type == "cuda":
        yield from CUDAPrefetcher(
            data_loader=data_loader,
            device=device,
            channels_last=channels_last,
        )
        return

    iterator = iter(data_loader)
    while True:
        fetch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        fetch_time = time.perf_counter() - fetch_start
        transfer_start = time.perf_counter()
        batch = move_batch_to_device(
            batch=batch,
            device=device,
            non_blocking=True,
            channels_last=channels_last,
        )
        transfer_time = time.perf_counter() - transfer_start
        yield batch, {
            "fetch_time": fetch_time,
            "transfer_time": transfer_time,
            "data_time": fetch_time + transfer_time,
        }


def _record_stream(batch, stream: torch.cuda.Stream) -> None:
    if torch.is_tensor(batch):
        batch.record_stream(stream)
        return
    if isinstance(batch, dict):
        for value in batch.values():
            _record_stream(value, stream)
        return
    if isinstance(batch, (list, tuple)):
        for value in batch:
            _record_stream(value, stream)


def _empty_batch_timing() -> dict[str, float]:
    return {"fetch_time": 0.0, "transfer_time": 0.0, "data_time": 0.0}
