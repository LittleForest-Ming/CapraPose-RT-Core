"""COCO-style dairy-goat keypoint dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from caprapose_rt.constants import KEYPOINT_NAMES, NUM_KEYPOINTS, SKELETON_EDGES
from caprapose_rt.datasets.transforms import GoatPoseTransform


class COCODairyGoatDataset(Dataset):
    """Load a dairy-goat keypoint dataset stored in COCO-style JSON."""

    def __init__(
        self,
        ann_file: str | Path,
        image_root: str | Path,
        input_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        sigma: float,
        bbox_scale_factor: float,
        is_train: bool,
        flip_prob: float = 0.0,
        split: str = "train",
        cache_annotations: bool = True,
        cache_images: bool | str = False,
        cache_dir: str | Path | None = None,
        hard_subset_manifest: str | Path | None = None,
        review_manifest: str | Path | None = None,
        review_supervision: dict[str, Any] | None = None,
        annotation_issue_manifest: str | Path | None = None,
        annotation_issue_policy: dict[str, Any] | None = None,
        logger=None,
    ) -> None:
        self.ann_file = _normalize_path(ann_file)
        self.image_root = _normalize_path(image_root)
        self.split = split
        self.is_train = is_train
        self.cache_annotations = bool(cache_annotations)
        self.cache_images = cache_images
        self.cache_dir = (
            _normalize_path(cache_dir) if cache_dir else self.ann_file.parent / ".caprapose_cache"
        )
        self.hard_subset_manifest = (
            _normalize_path(hard_subset_manifest) if hard_subset_manifest else None
        )
        self.review_manifest = _normalize_path(review_manifest) if review_manifest else None
        self.review_supervision = self._normalize_review_supervision(review_supervision)
        self.annotation_issue_manifest = (
            _normalize_path(annotation_issue_manifest) if annotation_issue_manifest else None
        )
        self.annotation_issue_policy = self._normalize_annotation_issue_policy(annotation_issue_policy)
        self.logger = logger or logging.getLogger("caprapose_rt.dataset")
        self.transform = GoatPoseTransform(
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma=sigma,
            bbox_scale_factor=bbox_scale_factor,
            is_train=is_train,
            flip_prob=flip_prob,
        )
        self.annotation_load_time = 0.0
        self.cache_warmup_time = 0.0
        self.image_cache_bytes = 0
        self._image_cache: dict[str, np.ndarray] = {}
        self._joint_name_to_index = {
            joint_name: joint_index for joint_index, joint_name in enumerate(KEYPOINT_NAMES)
        }
        self.supervision_stats = {
            "hard_case_samples": 0,
            "reviewed_samples": 0,
            "ambiguous_samples": 0,
            "priority_samples": 0,
            "ignored_joint_targets": 0,
            "confirmed_joint_targets": 0,
            "difficult_joint_targets": 0,
            "ambiguous_joint_targets": 0,
            "flagged_annotation_issue_samples": 0,
            "excluded_annotation_issue_samples": 0,
        }

        annotation_start = time.perf_counter()
        self.samples = self._load_samples()
        self._attach_supervision_metadata()
        self._apply_annotation_issue_policy()
        self.annotation_load_time = time.perf_counter() - annotation_start
        self.logger.info(
            "Initialized %s dataset with %d samples in %.2fs",
            self.split,
            len(self.samples),
            self.annotation_load_time,
        )
        if self.hard_subset_manifest or self.review_manifest:
            self.logger.info(
                "Supervision metadata for %s | hard_cases=%d | reviewed=%d | ambiguous=%d | priority=%d | ignored_joint_targets=%d",
                self.split,
                int(self.supervision_stats["hard_case_samples"]),
                int(self.supervision_stats["reviewed_samples"]),
                int(self.supervision_stats["ambiguous_samples"]),
                int(self.supervision_stats["priority_samples"]),
                int(self.supervision_stats["ignored_joint_targets"]),
            )
            if int(self.supervision_stats["flagged_annotation_issue_samples"]) > 0:
                self.logger.info(
                    "Annotation-issue metadata for %s | flagged=%d | excluded=%d",
                    self.split,
                    int(self.supervision_stats["flagged_annotation_issue_samples"]),
                    int(self.supervision_stats["excluded_annotation_issue_samples"]),
                )

        if self.cache_images == "ram":
            self._warmup_image_cache()
        elif self.cache_images not in {False, "ram"}:
            raise ValueError("cache_images must be False or 'ram'.")

    def _load_samples(self) -> list[dict[str, Any]]:
        if not self.ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        cache_path = self._annotation_cache_path()
        if self.cache_annotations:
            cached_samples = self._try_load_annotation_cache(cache_path)
            if cached_samples is not None:
                self.logger.info(
                    "Loaded annotation cache for %s from %s",
                    self.split,
                    cache_path,
                )
                return cached_samples

        with self.ann_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self._validate_category_metadata(payload.get("categories", []))

        images = {image["id"]: image for image in payload["images"]}
        samples: list[dict[str, Any]] = []
        for annotation in payload["annotations"]:
            if annotation.get("iscrowd", 0):
                continue

            keypoints = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
            if keypoints.shape[0] != NUM_KEYPOINTS:
                raise ValueError(
                    f"Expected {NUM_KEYPOINTS} keypoints, got {keypoints.shape[0]} "
                    f"in annotation {annotation['id']}."
                )

            if annotation["image_id"] not in images:
                raise KeyError(
                    f"Missing image metadata for image_id={annotation['image_id']} "
                    f"in annotation {annotation['id']}."
                )

            image_info = images[annotation["image_id"]]
            image_path = self._resolve_image_path(image_info["file_name"])
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Resolved image path does not exist for annotation {annotation['id']}: "
                    f"{image_path}"
                )
            samples.append(
                {
                    "id": int(annotation["id"]),
                    "image_id": int(annotation["image_id"]),
                    "file_name": image_info["file_name"],
                    "image_path": str(image_path),
                    "width": int(image_info["width"]),
                    "height": int(image_info["height"]),
                    "bbox": np.asarray(annotation["bbox"], dtype=np.float32),
                    "area": float(
                        annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3])
                    ),
                    "keypoints_xy": keypoints[:, :2].astype(np.float32),
                    "visibility": (keypoints[:, 2] > 0).astype(np.float32),
                }
            )

        if self.cache_annotations:
            self._write_annotation_cache(cache_path, samples)
        return samples

    def _validate_category_metadata(self, categories: list[dict[str, Any]]) -> None:
        if not categories:
            self.logger.warning(
                "No category metadata found in %s. Proceeding with the CapraPose-RT schema.",
                self.ann_file,
            )
            return

        dataset_keypoints = categories[0].get("keypoints", [])
        if not dataset_keypoints:
            return

        if all(isinstance(value, str) for value in dataset_keypoints):
            if list(dataset_keypoints) != KEYPOINT_NAMES:
                raise ValueError(
                    "Keypoint order mismatch between dataset and CapraPose-RT constants."
                )
            dataset_skeleton = categories[0].get("skeleton", [])
            if dataset_skeleton and [tuple(edge) for edge in dataset_skeleton] != list(SKELETON_EDGES):
                raise ValueError(
                    "Skeleton mismatch between dataset category metadata and "
                    "the CapraPose-RT canonical schema."
                )
            return

        if all(isinstance(value, (int, float)) for value in dataset_keypoints):
            numeric_ids = [int(value) for value in dataset_keypoints]
            expected_ids = list(range(NUM_KEYPOINTS))
            if numeric_ids != expected_ids:
                self.logger.warning(
                    "Dataset category keypoints metadata is non-standard in %s: %s. "
                    "CapraPose-RT will trust the annotation tensor shape (%d keypoints) "
                    "and use its own named topology constants.",
                    self.ann_file,
                    numeric_ids,
                    NUM_KEYPOINTS,
                )
            return

        self.logger.warning(
            "Unsupported category keypoint metadata in %s: %s",
            self.ann_file,
            dataset_keypoints,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def get_sampling_weights(
        self,
        hard_case_oversample_factor: float = 1.0,
        hard_case_priority_factor: float = 1.0,
        ambiguous_sample_weight: float = 1.0,
    ) -> np.ndarray:
        """Build per-sample weights for conservative hard-case oversampling."""

        weights = np.ones(len(self.samples), dtype=np.float32)
        for index, sample in enumerate(self.samples):
            weight = 1.0
            if sample.get("is_hard_case", False):
                weight *= max(float(hard_case_oversample_factor), 1.0)
            if sample.get("hard_case_priority", False):
                weight *= max(float(hard_case_priority_factor), 1.0)
            if sample.get("ambiguous", False):
                weight *= max(float(ambiguous_sample_weight), 0.0)
            weights[index] = max(weight, 1e-6)
        return weights

    def _resolve_image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path
        return self.image_root / path

    def _annotation_cache_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(self.ann_file.resolve()).encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{self.split}_{self.ann_file.stem}_{digest}.pkl"

    def _annotation_signature(self) -> dict[str, Any]:
        stat = self.ann_file.stat()
        return {
            "ann_file": str(self.ann_file.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _try_load_annotation_cache(self, cache_path: Path) -> list[dict[str, Any]] | None:
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("rb") as handle:
                cached_payload = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - defensive cache recovery path
            self.logger.warning("Failed to load annotation cache %s: %s", cache_path, exc)
            return None

        if cached_payload.get("signature") != self._annotation_signature():
            self.logger.info("Annotation cache is stale for %s; rebuilding.", self.ann_file)
            return None
        return cached_payload.get("samples")

    def _write_annotation_cache(self, cache_path: Path, samples: list[dict[str, Any]]) -> None:
        payload = {
            "signature": self._annotation_signature(),
            "samples": samples,
        }
        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(
            "Wrote annotation cache for %s to %s",
            self.split,
            cache_path,
        )

    def _warmup_image_cache(self) -> None:
        warmup_start = time.perf_counter()
        unique_image_paths = list(dict.fromkeys(sample["image_path"] for sample in self.samples))
        log_interval = max(1, len(unique_image_paths) // 10)
        self.logger.info(
            "Warming RAM image cache for %s with %d unique images",
            self.split,
            len(unique_image_paths),
        )
        for index, image_path in enumerate(unique_image_paths, start=1):
            cached_array = self._load_image_array(Path(image_path))
            self._image_cache[image_path] = cached_array
            self.image_cache_bytes += cached_array.nbytes
            if index % log_interval == 0 or index == len(unique_image_paths):
                self.logger.info(
                    "RAM image cache warmup %s: %d/%d images (%.2f MiB)",
                    self.split,
                    index,
                    len(unique_image_paths),
                    self.image_cache_bytes / (1024**2),
                )
        self.cache_warmup_time = time.perf_counter() - warmup_start
        self.logger.info(
            "Finished RAM image cache warmup for %s in %.2fs",
            self.split,
            self.cache_warmup_time,
        )

    @staticmethod
    def _load_image_array(image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image_handle:
            return np.asarray(image_handle.convert("RGB"), dtype=np.uint8)

    def _get_image(self, sample: dict[str, Any]) -> Image.Image:
        image_path = sample["image_path"]
        if self.cache_images == "ram" and image_path in self._image_cache:
            return Image.fromarray(self._image_cache[image_path], mode="RGB")

        image_array = self._load_image_array(Path(image_path))
        if self.cache_images == "ram":
            self._image_cache[image_path] = image_array
            self.image_cache_bytes += image_array.nbytes
        return Image.fromarray(image_array, mode="RGB")

    def _attach_supervision_metadata(self) -> None:
        hard_subset_entries = self._load_manifest_entries(self.hard_subset_manifest)
        review_entries = self._load_manifest_entries(self.review_manifest)
        annotation_issue_entries = self._load_manifest_entries(self.annotation_issue_manifest)

        for sample in self.samples:
            annotation_id = int(sample["id"])
            hard_entry = hard_subset_entries.get(annotation_id, {})
            review_entry = review_entries.get(annotation_id, {})
            annotation_issue_entry = annotation_issue_entries.get(annotation_id, {})
            joint_status = self._parse_joint_status_map(review_entry.get("joint_status", {}))
            ignored_joint_names = self._parse_joint_name_list(
                review_entry.get("ignore_for_endpoint_loss", [])
            )
            for joint_name in ignored_joint_names:
                joint_status.setdefault(joint_name, "ambiguous_ignore")

            joint_loss_weights = np.ones((NUM_KEYPOINTS, 1), dtype=np.float32)
            difficult_joint_count = 0
            ambiguous_joint_count = 0
            confirmed_joint_count = 0
            if self.review_supervision["enabled"]:
                for joint_name, status in joint_status.items():
                    joint_index = self._joint_name_to_index.get(joint_name)
                    if joint_index is None:
                        continue
                    if status == "confirmed":
                        joint_loss_weights[joint_index, 0] *= float(
                            self.review_supervision["confirmed_joint_weight"]
                        )
                        confirmed_joint_count += 1
                    elif status == "difficult_but_valid":
                        joint_loss_weights[joint_index, 0] *= float(
                            self.review_supervision["difficult_joint_weight"]
                        )
                        difficult_joint_count += 1
                    elif status == "ambiguous_ignore":
                        joint_loss_weights[joint_index, 0] = float(
                            self.review_supervision["ambiguous_joint_weight"]
                        )
                        ambiguous_joint_count += 1

            sample["is_hard_case"] = bool(hard_entry)
            sample["hard_case_category"] = str(hard_entry.get("category", ""))
            sample["failing_joints"] = self._parse_joint_name_list(
                hard_entry.get("failing_joints", [])
            )
            sample["reviewed"] = _to_bool(review_entry.get("reviewed", False))
            sample["difficult"] = _to_bool(review_entry.get("difficult", False)) or (
                difficult_joint_count > 0
            )
            sample["ambiguous"] = _to_bool(review_entry.get("ambiguous", False)) or (
                ambiguous_joint_count > 0
            )
            sample["hard_case_priority"] = _to_bool(
                review_entry.get("hard_case_priority", False)
            )
            sample["joint_loss_weights"] = joint_loss_weights
            sample["review_joint_status"] = joint_status
            sample["reviewed_joint_names"] = sorted(joint_status.keys())
            sample["annotation_issue"] = bool(annotation_issue_entry)
            sample["annotation_issue_type"] = str(
                annotation_issue_entry.get("issue_type", annotation_issue_entry.get("issue_category", ""))
            )
            sample["exclude_from_dataset"] = self._should_exclude_annotation_issue(
                annotation_issue_entry
            )
            sample["annotation_issue_notes"] = str(annotation_issue_entry.get("notes", ""))
            if sample["is_hard_case"]:
                self.supervision_stats["hard_case_samples"] += 1
            if sample["reviewed"]:
                self.supervision_stats["reviewed_samples"] += 1
            if sample["ambiguous"]:
                self.supervision_stats["ambiguous_samples"] += 1
            if sample["hard_case_priority"]:
                self.supervision_stats["priority_samples"] += 1
            if sample["annotation_issue"]:
                self.supervision_stats["flagged_annotation_issue_samples"] += 1
            self.supervision_stats["ignored_joint_targets"] += len(
                [name for name in ignored_joint_names if name in self._joint_name_to_index]
            )
            self.supervision_stats["confirmed_joint_targets"] += confirmed_joint_count
            self.supervision_stats["difficult_joint_targets"] += difficult_joint_count
            self.supervision_stats["ambiguous_joint_targets"] += ambiguous_joint_count

    def _apply_annotation_issue_policy(self) -> None:
        if not self.annotation_issue_policy["exclude_flagged_samples"]:
            return

        original_count = len(self.samples)
        self.samples = [
            sample for sample in self.samples if not bool(sample.get("exclude_from_dataset", False))
        ]
        excluded = original_count - len(self.samples)
        self.supervision_stats["excluded_annotation_issue_samples"] = excluded
        if excluded > 0:
            self.logger.warning(
                "Excluded %d flagged %s samples using annotation_issue_manifest=%s",
                excluded,
                self.split,
                self.annotation_issue_manifest,
            )

    def _load_manifest_entries(
        self,
        manifest_path: Path | None,
    ) -> dict[int, dict[str, Any]]:
        if manifest_path is None:
            return {}
        if not manifest_path.exists():
            self.logger.warning("Supervision manifest not found: %s", manifest_path)
            return {}

        if manifest_path.suffix.lower() == ".csv":
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            with manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                rows = payload.get("samples", payload.get("entries", []))
            elif isinstance(payload, list):
                rows = payload
            else:
                raise ValueError(f"Unsupported manifest payload in {manifest_path}")

        manifest_entries: dict[int, dict[str, Any]] = {}
        for row in rows:
            if "annotation_id" not in row:
                continue
            row_split = str(row.get("split", "")).strip().lower()
            if row_split and row_split != str(self.split).strip().lower():
                continue
            manifest_entries[int(row["annotation_id"])] = dict(row)
        return manifest_entries

    def _should_exclude_annotation_issue(self, annotation_issue_entry: dict[str, Any]) -> bool:
        if not annotation_issue_entry:
            return False
        if _to_bool(annotation_issue_entry.get("exclude_from_dataset", False)):
            return True

        split_specific_field = {
            "train": "exclude_from_training",
            "val": "exclude_from_validation",
            "test": "exclude_from_test",
        }.get(self.split)
        if split_specific_field and _to_bool(annotation_issue_entry.get(split_specific_field, False)):
            return True

        if self.split in {"val", "test"} and _to_bool(
            annotation_issue_entry.get("exclude_from_evaluation", False)
        ):
            return True
        return False

    @staticmethod
    def _parse_joint_name_list(values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            if not values.strip():
                return []
            return [value.strip() for value in values.split(",") if value.strip()]
        if isinstance(values, list):
            return [str(value).strip() for value in values if str(value).strip()]
        return []

    @staticmethod
    def _normalize_review_supervision(
        review_supervision: dict[str, Any] | None,
    ) -> dict[str, float | bool]:
        cfg = {
            "enabled": False,
            "confirmed_joint_weight": 1.0,
            "difficult_joint_weight": 1.0,
            "ambiguous_joint_weight": 0.0,
        }
        if review_supervision is None:
            return cfg
        cfg.update(review_supervision)
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "confirmed_joint_weight": float(cfg.get("confirmed_joint_weight", 1.0)),
            "difficult_joint_weight": float(cfg.get("difficult_joint_weight", 1.0)),
            "ambiguous_joint_weight": float(cfg.get("ambiguous_joint_weight", 0.0)),
        }

    @staticmethod
    def _normalize_annotation_issue_policy(
        annotation_issue_policy: dict[str, Any] | None,
    ) -> dict[str, bool]:
        cfg = {
            "exclude_flagged_samples": False,
        }
        if annotation_issue_policy is None:
            return cfg
        cfg.update(annotation_issue_policy)
        return {
            "exclude_flagged_samples": bool(cfg.get("exclude_flagged_samples", False)),
        }

    @staticmethod
    def _parse_joint_status_map(values: Any) -> dict[str, str]:
        valid_statuses = {"confirmed", "difficult_but_valid", "ambiguous_ignore"}
        if values is None:
            return {}
        parsed: dict[str, str] = {}
        if isinstance(values, dict):
            items = values.items()
        elif isinstance(values, list):
            items = []
            for value in values:
                if isinstance(value, dict):
                    joint_name = value.get("joint")
                    status = value.get("status")
                    if joint_name is not None and status is not None:
                        items.append((joint_name, status))
                elif isinstance(value, str) and ":" in value:
                    joint_name, status = value.split(":", maxsplit=1)
                    items.append((joint_name, status))
        elif isinstance(values, str):
            stripped = values.strip()
            if not stripped:
                return {}
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    return COCODairyGoatDataset._parse_joint_status_map(json.loads(stripped))
                except json.JSONDecodeError:
                    return {}
            items = []
            for value in stripped.split(";"):
                if ":" not in value:
                    continue
                joint_name, status = value.split(":", maxsplit=1)
                items.append((joint_name, status))
        else:
            return {}

        for joint_name, status in items:
            normalized_joint = str(joint_name).strip()
            normalized_status = str(status).strip()
            if normalized_joint and normalized_status in valid_statuses:
                parsed[normalized_joint] = normalized_status
        return parsed

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self._get_image(sample)

        transformed = self.transform(
            image=image,
            keypoints=sample["keypoints_xy"].copy(),
            visibility=sample["visibility"].copy(),
            bbox=sample["bbox"],
        )

        return {
            "image": transformed["image"],
            "heatmaps": transformed["heatmaps"],
            "target_weight": transformed["target_weight"],
            "keypoints": transformed["keypoints"],
            "visibility": transformed["visibility"],
            "keypoints_image": torch.from_numpy(sample["keypoints_xy"].copy()),
            "bbox": torch.from_numpy(sample["bbox"].copy()),
            "crop_box": transformed["crop_box"],
            "area": torch.tensor(sample["area"], dtype=torch.float32),
            "image_id": sample["image_id"],
            "annotation_id": sample["id"],
            "file_name": sample["file_name"],
            "joint_loss_weights": torch.from_numpy(sample["joint_loss_weights"].copy()),
            "is_hard_case": torch.tensor(float(sample.get("is_hard_case", False)), dtype=torch.float32),
            "reviewed": torch.tensor(float(sample.get("reviewed", False)), dtype=torch.float32),
            "ambiguous": torch.tensor(float(sample.get("ambiguous", False)), dtype=torch.float32),
            "hard_case_priority": torch.tensor(
                float(sample.get("hard_case_priority", False)),
                dtype=torch.float32,
            ),
            "hard_case_category": sample.get("hard_case_category", ""),
            "original_size": torch.tensor(
                [sample["width"], sample["height"]],
                dtype=torch.float32,
            ),
        }


def _normalize_path(path_like: str | Path) -> Path:
    return Path(os.path.expandvars(str(path_like))).expanduser()


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False
