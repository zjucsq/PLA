# PLA

This is the official code for our ICLR2023 paper: "Video Scene Graph Generation from Single-Frame Weak Supervision" [openreview link](https://openreview.net/pdf?id=KLrGlNoxzb4)

![](poster.png)

## Installation

- First use the environment.yaml to create the basic env.

- Then install this repository [scene_graph_benchmark](https://github.com/microsoft/scene_graph_benchmark).

## Data Preparing

### The pretrained detector

Download from this repository [VinVL](https://github.com/pzzhang/VinVL).

### The Action Genome dataset

Download from this repository [ActionGenome](https://github.com/JingweiJ/ActionGenome).

### The bboxes and corresponding features of each frame in AG

Use `lib/extract_bbox_features.py` to extract these features.

### The weak-annotation

- (1) Annotation which keeps objects with confidence greater than 0.2. ([here](https://mega.nz/file/IEwyhSjA#fiHU_IX_KbTuISqmSOKOQx-ZLYak7ELIrirF6cLNZ0Y))
- (2) From Annotation (1), only keeps the middle frame for each video. ([here](https://mega.nz/file/5YQ3kKiT#RTszlqCz2VFxBE0ZSVa7F-Tl-JO3SOL0BS7lXJHzcho))
- (3) From Annotation (2), annotated by the model-free strategy with $\eta=0.5$. ([here](https://mega.nz/file/9J42zZYC#C6kec5hUYwXTBDUvVMKxFjfuAVE2lQKjp1ZltuBIKiU))
- You can also assign annotation with different hyperparemeter by `lib/genarate_predicate_pseudo_label.py`.

## Train

```
python train.py --cfg demo.yml
```

## Test

```
python test.py --cfg demo.yml
```

## Model weights of PLA

- Model trained by the middle frame ([here](https://mega.nz/file/oNxzXTIS#3S6frVh3WCjzCZIWKPdVFqNq-DxI9-_Mw91b7t3MGh4))
- Final model ([here](https://mega.nz/file/NAhFEZwT#qwTNGPRWmWdL_AsDd0VpBwWLTVLOANp5Q7AjM-nh-DE))

## Citations

Please consider citing this project in your publications if it helps your research.

```
@inproceedings{chen2023video,
  title={Video scene graph generation from single-frame weak supervision},
  author={Chen, Siqi and Xiao, Jun and Chen, Long},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```