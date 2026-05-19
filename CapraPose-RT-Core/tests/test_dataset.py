import json

from PIL import Image

from caprapose_rt.constants import KEYPOINT_NAMES
from caprapose_rt.datasets.coco_goat import COCODairyGoatDataset


def test_dataset_parsing(tmp_path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "sample.jpg"
    Image.new("RGB", (128, 128), color=(255, 255, 255)).save(image_path)

    keypoints = []
    for idx in range(len(KEYPOINT_NAMES)):
        keypoints.extend([16 + idx * 2, 16 + idx * 2, 2])

    ann_path = tmp_path / "ann.json"
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": "sample.jpg",
                "width": 128,
                "height": 128,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 8, 96, 96],
                "area": 9216,
                "iscrowd": 0,
                "num_keypoints": len(KEYPOINT_NAMES),
                "keypoints": keypoints,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "dairy_goat",
                "keypoints": KEYPOINT_NAMES,
                "skeleton": [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [2, 5],
                    [5, 6],
                    [6, 7],
                    [2, 8],
                    [8, 9],
                    [9, 10],
                    [3, 11],
                    [11, 12],
                    [12, 13],
                    [3, 14],
                    [14, 15],
                    [15, 16],
                ],
            }
        ],
    }
    ann_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = COCODairyGoatDataset(
        ann_file=ann_path,
        image_root=image_root,
        input_size=(256, 256),
        heatmap_size=(64, 64),
        sigma=2.5,
        bbox_scale_factor=1.25,
        is_train=False,
        cache_annotations=True,
        cache_images="ram",
        cache_dir=tmp_path / "cache",
    )

    sample = dataset[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["heatmaps"].shape == (len(KEYPOINT_NAMES), 64, 64)
    assert sample["keypoints"].shape == (len(KEYPOINT_NAMES), 2)
    assert sample["visibility"].shape == (len(KEYPOINT_NAMES),)
    assert dataset.cache_warmup_time >= 0.0
    assert len(dataset._image_cache) == 1
    assert any((tmp_path / "cache").iterdir())
