import cv2
import torch
import trimesh

from Utils import nvdiffrast_render
import numpy as np
from PIL import Image
import numpy as np

import os.path as osp
import argparse


def calc_xyz_bp_fast(depth, R, T, K):
    Kinv = np.linalg.inv(K)
    height, width = depth.shape
    # ProjEmb = np.zeros((height, width, 3)).astype(np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    ProjEmb = (
        np.einsum(
            "ijkl,ijlm->ijkm",
            R.T.reshape(1, 1, 3, 3),
            depth.reshape(height, width, 1, 1)
            * np.einsum("ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1))
            - T.reshape(1, 1, 3, 1),
        ).squeeze()
        * mask.reshape(height, width, 1)
    )
    return ProjEmb



K = np.array(
            [[567.53720406, 0.0, 312.66570357], [0.0, 569.36175922, 257.1729701], [0.0, 0.0, 1.0]]
)
H = 480
W = 640

output_dir = './weights/tmp'
ob_in_cams = np.load('./weights/obj_poses_level1.npy')
ob_in_cams[:,:3,3] = np.array([0,0,0.5])
ob_in_cams_tensor = torch.from_numpy(ob_in_cams).cuda().to(torch.float32)



mesh = trimesh.load('./demo_data/tm2/mesh/tm2.ply')

start_num = 81

rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=ob_in_cams_tensor[start_num:,:,:], context='cuda',
                                                 get_normal=False, glctx=None, mesh_tensors=None,mesh=mesh,
                                             output_size=None, bbox2d=None,
                                             use_light=True, extra={})


rgb_r = rgb_r.cpu().numpy()
depth_r = depth_r.cpu().numpy()


for i in range(len(rgb_r)):
    rgb = rgb_r[i]
    depth = depth_r[i]

    ob_in_cam =ob_in_cams[i]

    xyz_np = calc_xyz_bp_fast(depth, ob_in_cam[:3,:3], ob_in_cam[:3,3], K)

    rgb = Image.fromarray(np.uint8(rgb*255.0))
    rgb.save(osp.join(output_dir, f"{i:06d}.png"))
    np.save(osp.join(output_dir, f"{i:06d}.npy"), xyz_np)

