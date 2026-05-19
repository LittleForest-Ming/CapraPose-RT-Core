import json

from tools.normalize_dataset_annotations import _normalize_payload


def test_normalize_payload_rewrites_category_metadata(tmp_path) -> None:
    ann_path = tmp_path / "train.json"
    payload = {
        "info": {"description": "raw"},
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 32, 32],
                "area": 1024,
                "iscrowd": 0,
                "num_keypoints": 17,
                "keypoints": [1.0, 2.0, 2] * 17,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "GoatPosture",
                "keypoints": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16],
                "skeleton": [[0, 1]],
            }
        ],
    }
    ann_path.write_text(json.dumps(payload), encoding="utf-8")

    normalized, summary = _normalize_payload(
        payload=payload,
        ann_path=ann_path,
        split="train",
        category_name="dairy_goat_pose",
    )

    assert summary["num_images"] == 1
    assert summary["num_annotations"] == 1
    assert normalized["categories"][0]["keypoints"][0] == "head_0"
    assert normalized["categories"][0]["keypoints"][2] == "body_0"
    assert normalized["categories"][0]["keypoints"][5] == "forelimb_left_0"
    assert normalized["categories"][0]["keypoints"][8] == "forelimb_right_0"
    assert len(normalized["categories"][0]["keypoints"]) == 17
    assert normalized["categories"][0]["caprapose_schema_version"] == "1.1"
    assert normalized["caprapose_normalization"]["category_metadata_replaced"] is True
