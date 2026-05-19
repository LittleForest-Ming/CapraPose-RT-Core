# CapraPose-RT Core

This folder is a lightweight public GitHub package for the core CapraPose-RT model skeleton. It is intentionally separated from the full local research repository.

## What Is Included

- Canonical 17-keypoint dairy-goat 2D pose schema.
- Lightweight visual encoder used by the CapraPose-RT pipeline.
- Heatmap keypoint prediction head.
- Public placeholder for the latent part-aware hierarchical decoder.
- Adaptive topology refinement module.
- Topology-consistent structural loss definitions.
- Minimal random-input forward example.
- Template config with placeholders instead of private experiment settings.

## What Is Not Included

- No trained weights, checkpoints, `.pt`, `.pth`, or exported engines.
- No dataset files or private annotation paths.
- No experiment output directories, logs, metrics, visualizations, or paper tables.
- No comparison-model training code for YOLO, HRNet, RTMPose-CSPNeXt, ViTPose, GRMPose, or other baselines.
- No private full decoder implementation. The decoder file in this package is an interface-compatible placeholder.

## Repository Structure

```text
CapraPose-RT-Core/
  caprapose_rt/
    schema.py
    constants.py
    models/
      backbones/
      heads/
        latent_part_decoder.py      # placeholder only
      modules/
      losses/
      pose_estimator.py
    utils/
  configs/
    caprapose_core_template.py      # placeholder parameters and paths
  examples/
    minimal_forward.py
```

## Decoder Placeholder

`caprapose_rt/models/heads/latent_part_decoder.py` keeps the public class name:

```python
LatentPartAwareHierarchicalDecoder
```

It also keeps the expected constructor arguments and forward return structure. However, it only projects features and returns zero-valued placeholder part/joint tokens. This is deliberate: the file marks where the latent part-aware hierarchical decoder belongs without publishing private experiment-specific details.

## Quick Smoke Test

Install minimal dependencies:

```bash
pip install -r requirements.txt
```

Run a random-input forward pass:

```bash
python examples/minimal_forward.py
```

Expected output shape pattern:

```text
heatmaps: (2, 17, 64, 64)
refined_coords: (2, 17, 2)
joint_features: (2, 17, 128)
decoder placeholder: True
```

## Using Your Own Data

Start from:

```text
configs/caprapose_core_template.py
```

Replace the placeholders such as:

```text
<PATH_TO_TRAIN_COCO_JSON>
<PATH_TO_IMAGE_ROOT>
<DECODER_HIDDEN_CHANNELS>
<REFINEMENT_STEP_SIZE>
```

with your own dataset paths and chosen public hyperparameters. This release does not include the private training protocol, review registries, cleaned-split decisions, or paper experiment settings.

## Intended Scope

This package is for understanding and extending the core 2D dairy-goat pose-estimation framework:

1. Efficient visual encoding.
2. Latent part-aware hierarchical decoding interface.
3. Adaptive topology refinement.
4. Topology-consistent structural learning.

It is not a full reproduction bundle for the private paper experiments.
