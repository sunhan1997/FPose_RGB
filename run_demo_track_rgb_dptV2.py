# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import cv2

from estimater import *
from datareader import *
import argparse
from ultralytics import YOLO
import cv2
# import pyrealsense2 as rs
import open3d as o3d
import cv2
import numpy as np

from dpt_infer import Dpt

engine_path = r"/home/sunh/6D_ws/Fpose_rgb/feature2/depth_anything_v2_vitl.engine"
grayscale = False
# 加载并运行模型
model = Dpt(engine_path)



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mp1/mesh/mp1.ply')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mp1')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=3)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  debug = args.debug

  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  i = 1
  for index in range(1,137):

    #读取图像
    depth = cv2.imread('./demo_data/mp1/depth/{:06d}-depth.png'.format(index),-1)/ 1000.
    color = cv2.imread('./demo_data/mp1/rgb/{:06d}-color.png'.format(index))

    if i == 1 :
      pose = np.array([[0.     ,    1.   ,      0.    ,     0.11223628-0.05],
                      [0.     ,    0.    ,     1.         ,0.02427196],
                      [1.     ,    0.    ,     0.    ,     0.5039429 ],
                      [0.    ,     0.       ,  0.   ,      1.        ]])

      est.pose_last =np.array([pose])
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                          is_input_rgb=True)
      cv2.imshow('1', vis[..., ::-1])
      cv2.waitKey(0)
      i = 2
    else:

      time_start = time.time()


      rgb = color.copy()
      depth = model.run(rgb)
      depth = cv2.resize(depth, (640, 480))
      depth_show = np.uint8(depth)
      depth_show = cv2.applyColorMap(depth_show, cv2.COLORMAP_JET)
      cv2.imshow('a', depth_show)
      cv2.waitKey(1)

      print('****------------------------------------------------------------------------------********: {}'.format(time.time()-time_start))

      color = reader.get_color_real(color)
      depth = reader.get_depth_real(depth)
      pose,vis_refine = est.track_one_demo(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)


      print('***********************************************************************************: {}'.format(time.time()-time_start))
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.imshow('vis_refine', vis_refine)
      cv2.waitKey(1)
      i = i+1








