# Dataset, model, and media notice

This repository is an academic course project. It does not grant broader rights to third-party datasets, software, model components, or media.

## KITTI-derived model

- Dataset: [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/)
- Copyright: Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago
- Dataset terms stated by KITTI: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 and academic use only
- Citation: A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite,” CVPR 2012.

The raw KITTI images and labels are not redistributed here. `models/best.pt` is a model trained from KITTI-derived labels and is provided only for this academic assignment. Reuse must comply with the KITTI attribution, non-commercial, share-alike, and academic-use conditions.

## Ultralytics YOLO

This project depends on the separately installed `ultralytics` package and a YOLOv8 pretrained checkpoint. Ultralytics documents AGPL-3.0 and Enterprise licensing options for its software and AI models. Users are responsible for choosing and complying with the license appropriate to their use: <https://docs.ultralytics.com/>.

## Tracking video

- Source: [Video Of Cityscape During Daytime](https://www.pexels.com/video/video-of-cityscape-during-daytime-4772894/)
- Creator: German Korb
- License: [Pexels License](https://www.pexels.com/license/)

Pexels permits free use and modification under its stated conditions. The unmodified source video is not redistributed. `videos/kitti3_tracking_5s.mp4` is a five-second excerpt materially modified with model-generated bounding boxes, class names, confidence values, and tracking IDs. No endorsement by the creator, depicted people, brands, or Pexels is implied.

## Course materials and credentials

The instructor’s lecture PDFs, downloaded source data, authentication tokens, and local environment files are excluded through `.gitignore` and are not part of the public repository.
