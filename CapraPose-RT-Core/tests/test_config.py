from pathlib import Path

from caprapose_rt.config import load_config
from caprapose_rt.constants import KEYPOINT_NAMES


def test_config_loading() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    baseline_cfg = load_config(repo_root / "configs" / "baseline_rtmpose.py")
    decoder_cfg = load_config(repo_root / "configs" / "caprapose_rt_decoder.py")
    decoder_refine_cfg = load_config(
        repo_root / "configs" / "caprapose_rt_decoder_refine.py"
    )
    full_cfg = load_config(repo_root / "configs" / "caprapose_rt.py")

    assert baseline_cfg["model"]["decoder"]["enabled"] is False
    assert baseline_cfg["model"]["refinement"]["enabled"] is False
    assert baseline_cfg["device"].startswith("cuda")
    assert baseline_cfg["gpu_required"] is True
    assert baseline_cfg["amp"]["enabled"] is True
    assert baseline_cfg["dataset"]["num_keypoints"] == len(KEYPOINT_NAMES)
    assert "dataset_root" in baseline_cfg["dataset"]
    assert "normalized_root" in baseline_cfg["dataset"]
    assert "raw_train_ann" in baseline_cfg["dataset"]
    assert "test_ann" in baseline_cfg["dataset"]
    assert baseline_cfg["dataset"]["cache_annotations"] is True
    assert baseline_cfg["train"]["pin_memory"] is True
    assert baseline_cfg["train"]["cuda_prefetch"] is True
    normalized_work_dir = baseline_cfg["work_dir"].replace("\\", "/")
    assert normalized_work_dir.startswith("outputs/experiments/")
    assert baseline_cfg["protocol"]["name"] == "formal_round1"
    assert "short_run" in baseline_cfg["protocol"]
    assert "full_run" in baseline_cfg["protocol"]
    assert baseline_cfg["artifacts"]["checkpoints_dir"] == "checkpoints"
    assert baseline_cfg["artifacts"]["evaluation_dir"] == "evaluation"
    assert baseline_cfg["artifacts"]["qualitative_dir"] == "qualitative"
    assert decoder_cfg["model"]["decoder"]["enabled"] is True
    assert decoder_cfg["model"]["refinement"]["enabled"] is False
    assert decoder_refine_cfg["model"]["decoder"]["enabled"] is True
    assert decoder_refine_cfg["model"]["refinement"]["enabled"] is True
    assert full_cfg["model"]["decoder"]["enabled"] is True
    assert full_cfg["model"]["refinement"]["enabled"] is True
    assert full_cfg["loss"]["structural_weight"] > baseline_cfg["loss"]["structural_weight"]
    assert decoder_refine_cfg["loss"]["structural_weight"] == 0.0
    assert full_cfg["loss"]["structural_weight"] > decoder_refine_cfg["loss"]["structural_weight"]
    assert baseline_cfg["experiment_name"] == "baseline_rtmpose_m_goat17"
    assert decoder_cfg["experiment_name"] == "caprapose_rt_decoder_goat17"
    assert decoder_refine_cfg["experiment_name"] == "caprapose_rt_decoder_refine_goat17"
    assert full_cfg["experiment_name"] == "caprapose_rt_full_goat17"
