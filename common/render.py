import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm
try:
    import renderpy
except ImportError:
    print("renderpy not installed. Please install renderpy from https://github.com/liu115/renderpy")
    sys.exit(1)

from common.utils.colmap import read_model, write_model, Image
from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list
from joblib import Parallel,delayed
import pandas as pd
import numpy as np
import open3d as o3d
import json

class scanentpp_gt_getter:
    def __init__(self,class_equivalence_dir):
        self.class_equivalence_df = pd.read_excel(class_equivalence_dir).iloc[:,:4]
    def get_gt_point_cloud_and_labels(self,segments_anno,segments,semantics_path):
        # scene_dir = '{}/{}'.format(self.root_dir,scene_name)
        # semantics_path = scene_dir+'/scans/mesh_aligned_0.05_semantic.ply'
        # segments_ano = scene_dir + '/scans/segments_anno.json'
        # segments = scene_dir + '/scans/segments.json'        
        with open(segments_anno,'r') as f:
            segments_ano_dict = json.load(f)
        with open(segments,'r') as f:
            segments_dict = json.load(f)        
        
        mesh = o3d.io.read_triangle_mesh(semantics_path)
        
        tmp = np.array(segments_dict['segIndices'])
        (np.argsort(tmp)-np.arange(tmp.shape[0])).sum()
        mesh_df = pd.DataFrame({'idx':tmp,'class':tmp.shape[0]*[None]})
        segments_ano_dict['segGroups']
        for dct in segments_ano_dict['segGroups']:
            mesh_df.loc[dct['segments'],'class'] = dct['label']
        merge = pd.merge(mesh_df,self.class_equivalence_df,how = 'left',left_on = 'class',right_on = 'Scannet++ class')
        points = np.array(mesh.vertices)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        classes = merge.loc[:,'Segformer class Index'].values
        classes = np.nan_to_num(classes,150).astype(int)
        # classes = classes.reshape((classes.shape[0], 1))
        return pcd,classes




def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process, specify any one
    if cfg.get('scene_list_file'):
        scene_ids = read_txt_list(cfg.scene_list_file)
    elif cfg.get('scene_ids'):
        scene_ids = cfg.scene_ids
    elif cfg.get('splits'):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / 'splits' / f'{split}.txt'
            scene_ids += read_txt_list(split_path)

    output_dir = cfg.get("output_dir")
    if output_dir is None:
        # default to data folder in data_root
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)

    render_devices = []
    if cfg.get("render_dslr", False):
        render_devices.append("dslr")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")

    # go through each scene
    Parallel(n_jobs = 6)(delayed(render_this_scene)(scene_id,cfg,save_rgb = False) for scene_id in scene_ids)
    # for scene_id in tqdm(scene_ids, desc="scene"):
    #     scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
    #     render_engine = renderpy.Render()
    #     render_engine.setupMesh(str(scene.scan_mesh_path))
    #     for device in render_devices:
    #         if device == "dslr":
    #             cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
    #         else:
    #             cameras, images, points3D = read_model(scene.iphone_colmap_dir, ".txt")
    #         assert len(cameras) == 1, "Multiple cameras not supported"
    #         camera = next(iter(cameras.values()))

    #         fx, fy, cx, cy = camera.params[:4]
    #         params = camera.params[4:]
    #         camera_model = camera.model
    #         render_engine.setupCamera(
    #             camera.height, camera.width,
    #             fx, fy, cx, cy,
    #             camera_model,
    #             params,      # Distortion parameters np.array([k1, k2, k3, k4]) or np.array([k1, k2, p1, p2])
    #         )

    #         near = cfg.get("near", 0.05)
    #         far = cfg.get("far", 20.0)
    #         rgb_dir = Path(cfg.output_dir) / scene_id / device / "render_rgb"
    #         depth_dir = Path(cfg.output_dir) / scene_id / device / "render_depth"
    #         rgb_dir.mkdir(parents=True, exist_ok=True)
    #         depth_dir.mkdir(parents=True, exist_ok=True)
    #         for image_id, image in tqdm(images.items(), f"Rendering {device} images"):
    #             world_to_camera = image.world_to_camera
    #             rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
    #             rgb = rgb.astype(np.uint8)
    #             # Make depth in mm and clip to fit 16-bit image
    #             depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
    #             imageio.imwrite(rgb_dir / image.name, rgb)
    #             depth_name = image.name.split(".")[0] + ".png"
    #             imageio.imwrite(depth_dir / depth_name, depth)


def render_this_scene(scene_id,cfg,save_rgb = False,render_devices = ['iphone']):
        # for scene_id in tqdm(scene_ids, desc="scene"):
    scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
    gt_getter = scanentpp_gt_getter('/home/motion/scannetpp/mapping_to_top_100.xlsx')
    render_engine = renderpy.Render()
    render_engine.setupMesh(str(scene.scan_mesh_path))
    pcd,gt_labels = gt_getter.get_gt_point_cloud_and_labels(scene.scan_anno_json_path,scene.scan_mesh_segs_path,scene.scan_sem_mesh_path)
    for device in render_devices:
        if device == "dslr":
            cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
        else:
            cameras, images, points3D = read_model(scene.iphone_colmap_dir, ".txt")
        assert len(cameras) == 1, "Multiple cameras not supported"
        camera = next(iter(cameras.values()))

        fx, fy, cx, cy = camera.params[:4]
        params = camera.params[4:]
        camera_model = camera.model
        render_engine.setupCamera(
            camera.height, camera.width,
            fx, fy, cx, cy,
            camera_model,
            params,      # Distortion parameters np.array([k1, k2, k3, k4]) or np.array([k1, k2, p1, p2])
        )

        near = cfg.get("near", 0.05)
        far = cfg.get("far", 20.0)
        if(save_rgb):
            rgb_dir = Path(cfg.output_dir) / scene_id / device / "render_rgb"
            rgb_dir.mkdir(parents=True, exist_ok=True)

        semantic_dir = Path(cfg.output_dir) / scene_id / device / "gt_semantics"

        depth_dir = Path(cfg.output_dir) / scene_id / device / "render_depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        semantic_dir.mkdir(parents=True,exist_ok = True)
        for image_id, image in tqdm(images.items(), f"Rendering {device} images"):
            world_to_camera = image.world_to_camera
            rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
            all_nulls = np.all(vert_indices == -1,axis = 2)
            rendered_labels = gt_labels[vert_indices]
            # janky fast mode with default to 0
            rendered_label = rendered_labels[:,:,0]
            last_2_agreement = rendered_labels[:,:,1] == rendered_labels[:,:,2]
            rendered_label[last_2_agreement] = rendered_labels[:,:,1][last_2_agreement]
            rendered_label[all_nulls] = 150
            # rendered_labels = vert_

            # Make depth in mm and clip to fit 16-bit image
            depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
            depth_name = image.name.split(".")[0] + ".png"
            imageio.imwrite(depth_dir / depth_name, depth)

            imageio.imwrite(semantic_dir / depth_name,rendered_label.astype(np.uint8))
            if(save_rgb):
                rgb = rgb.astype(np.uint8)
                imageio.imwrite(rgb_dir / image.name, rgb)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
