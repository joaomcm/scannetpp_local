# ScanNet++ Toolkit

Table of Contents
=================

* [ScanNet++ Toolkit](#scannet-toolkit)
   * [Requirements](#requirements)
   * [DSLR](#dslr)
      * [Undistortion: convert fisheye images to pinhole with OpenCV](#undistortion-convert-fisheye-images-to-pinhole-with-opencv)
      * [Downscale the DSLR images](#downscale-the-dslr-images)
      * [Undistortion: convert fisheye images to pinhole with COLMAP](#undistortion-convert-fisheye-images-to-pinhole-with-colmap)
      * [Render Depth for DSLR and iPhone](#render-depth-for-dslr-and-iphone)
      * [Render Semantics (coming soon)](#render-semantics-coming-soon)
   * [iPhone](#iphone)
      * [Extract RGB frames, masks and depth frames](#extract-rgb-frames-masks-and-depth-frames)
   * [Semantics](#semantics)
      * [Prepare 3D Semantics Training Data](#prepare-3d-semantics-training-data)
      * [Visualize training data](#visualize-training-data)
      * [Prepare Semantic/Instance Ground Truth files for](#prepare-semanticinstance-ground-truth-files-for-evaluation)
      * [3D Semantic Segmentation Evaluation](#3d-semantic-segmentation-evaluation)
      * [3D Instance Segmentation Evaluation](#3d-instance-segmentation-evaluation)
   * [Novel View Synthesis](#novel-view-synthesis)
      * [Novel View Synthesis Evaluation (DSLR)](#novel-view-synthesis-evaluation-dslr)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## Requirements
```
conda create -n scannetpp python=3.10
conda activate scannetpp
pip install -r requirements.txt
```

## DSLR

### Undistortion: convert fisheye images to pinhole with OpenCV
The official undistortion script that generates the undistortion ground-truth in the benchmark. This will generate the undistorted images, mask, and the respective transforms.json file for NeRF training.

Insert `data_root` in `dslr/configs/undistort.yml` and run:
```
python -m dslr.undistort dslr/configs/undistort.yml
```
Additionally, the user can specify the input and output path in the config files.

### Downscale the DSLR images
If you need to downscale the DSLR images to reduce the memory overhead during NeRF training, you can run the following script. The configuration is similar to the undistortion script.

```
python -m dslr.downscale dslr/configs/downscale.yml

```


### Undistortion: convert fisheye images to pinhole with COLMAP
User could also use COLMAP to undistort DSLR images (and masks) based on COLMAP so that the output images are pinhole camera models. However, **the result here is different from the ones generated by OpenCV**.

You will need [COLMAP](https://colmap.github.io/) installed to run this script.

Insert `data_root` and `output_dir` in `dslr/configs/undistort_colmap.yml` and run:
```
python -m dslr.undistort_colmap dslr/configs/undistort_colmap.yml
```
The output will be saved in `output_dir` with the following structure:
```
output_dir/SCENE_ID
├── colmap
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
├── images
├── masks
└── nerfstudio/transforms.json
```

### Render Depth for DSLR and iPhone

Install the python package from https://github.com/liu115/renderpy in addtion to the requirements.

```
python -m common.render common/configs/render.yml
```
The output will be saved in `output_dir` with the following structure:
```
output_dir/SCENE_ID/[dslr, iphone]
├── render_rgb
└── render_depth
```
The rendered depth maps are single-channel uint16 png, where the unit is mm and 0 means invalid depth.

### Render Semantics (coming soon)

## iPhone
### Extract RGB frames, masks and depth frames
```
python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
```


## Semantics

### Prepare 3D Semantics Training Data
Scripts to sample points on the mesh and map 1.5k+ raw labels to the benchmark classes.

Configure the paths in `semantic/configs/prepare_training_data.yml`


Then run
```
python -m semantic.prep.prepare_training_data semantic/configs/prepare_training_data.yml
```

### Visualize training data

Configure the PTH data dir, scene list and required outputs in `semantic/configs/viz_pth_data.yml`
```
python -m semantic.viz.viz_pth_data semantic/configs/viz_pth_data.yml
```

### Prepare Semantic/Instance Ground Truth Files for Evaluation

Configure the PTH data dir, scene list and required outputs in `semantic/configs/prepare_semantic_gt.yml`

Then run 
```
python -m semantic.prep.prepare_semantic_gt semantic/configs/prepare_semantic_gt.yml
```

### 3D Semantic Segmentation Evaluation
For this you need to prepare the semantic ground truth and predictions in the following format
- one file per scene named `<scene_id>.txt`, where each line contains the
label(s) for the corresponding vertex in the mesh. You can specify either a single label
or multiple comma-separate labels in each line. Each line should have the same number of labels, i.e
each file should be an `N x 1` or `N x 3` array for 1 and 3 predictions respectively.

Configure the paths to GT, predictions, label list and downloaded data in `semantic/configs/eval_semantic.yml`

Then run
```
python -m semantic.eval.eval_semantic semantic/configs/eval_semantic.yml
```

### 3D Instance Segmentation Evaluation

See `semantic/eval/eval_instance.py` for details on the input formats.

Configure the paths to GT, predictions, label list and downloaded data in `semantic/configs/eval_instance.yml`

Then run
```
python -m semantic.eval.eval_instance semantic/configs/eval_instance.yml
```

## Novel View Synthesis
### Novel View Synthesis Evaluation (DSLR)
The evaluation script here is the same that runs on the benchmark server. Therefore, it's highly encouraged to run the evaluation script before submitting the results (on the val set) to the benchmark server.


```
python -m eval.nvs --data_root DATA_ROOT --split SPLIT_FILE --pred_dir PRED_DIR
```

The PRED_DIR should have the following structure:
```
SCENE_ID0/
├── DSC00001.JPG
├── DSC00002.JPG
├── ...
SCENE_ID1/
├── ...
```

NOTE:
The evaluation script here is the same that runs on the benchmark server. Therefore, it's highly encouraged to run the evaluation script before submitting the results (on the val set) to the benchmark server.
