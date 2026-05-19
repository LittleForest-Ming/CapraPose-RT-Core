"""Dataset builders."""

from .coco_goat import COCODairyGoatDataset


def build_dataset(
    split: str,
    dataset_cfg: dict,
    logger=None,
) -> COCODairyGoatDataset:
    """Build a dataset for the requested split."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    ann_file = dataset_cfg["train_ann"] if split == "train" else dataset_cfg["val_ann"]
    image_root = (
        dataset_cfg["train_image_root"]
        if split == "train"
        else dataset_cfg["val_image_root"]
    )

    if split == "test":
        ann_file = dataset_cfg.get("test_ann", dataset_cfg["val_ann"])
        image_root = dataset_cfg.get("test_image_root", dataset_cfg["val_image_root"])

    return COCODairyGoatDataset(
        ann_file=ann_file,
        image_root=image_root,
        input_size=tuple(dataset_cfg["input_size"]),
        heatmap_size=tuple(dataset_cfg["heatmap_size"]),
        sigma=float(dataset_cfg["sigma"]),
        bbox_scale_factor=float(dataset_cfg["bbox_scale_factor"]),
        is_train=split == "train",
        flip_prob=float(dataset_cfg.get("flip_prob", 0.0)),
        split=split,
        cache_annotations=bool(dataset_cfg.get("cache_annotations", True)),
        cache_images=dataset_cfg.get("cache_images", False),
        cache_dir=dataset_cfg.get("cache_dir"),
        hard_subset_manifest=dataset_cfg.get("hard_subset_manifest"),
        review_manifest=dataset_cfg.get("review_manifest"),
        review_supervision=dataset_cfg.get("review_supervision"),
        annotation_issue_manifest=dataset_cfg.get("annotation_issue_manifest"),
        annotation_issue_policy=dataset_cfg.get("annotation_issue_policy"),
        logger=logger,
    )


__all__ = ["COCODairyGoatDataset", "build_dataset"]
