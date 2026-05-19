from caprapose_rt.paper.common import build_prediction_index, select_annotation_ids


def test_select_annotation_ids_is_deterministic() -> None:
    available_ids = [10, 20, 30, 40, 50, 60]
    first = select_annotation_ids(available_ids, num_samples=3, seed=123)
    second = select_annotation_ids(available_ids, num_samples=3, seed=123)
    assert first == second
    assert len(first) == 3


def test_build_prediction_index_parses_eval_payload() -> None:
    payload = {
        "metrics": {"mAP": 0.5},
        "predictions": [
            {"annotation_id": 1, "keypoints": [[0.0, 0.0]] * 17, "confidence": [1.0] * 17},
            {"annotation_id": 2, "keypoints": [[1.0, 1.0]] * 17, "confidence": [0.5] * 17},
        ],
    }
    index, metrics = build_prediction_index(payload)
    assert sorted(index.keys()) == [1, 2]
    assert metrics["mAP"] == 0.5
