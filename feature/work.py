from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import cv2
import torch
from torchvision.transforms import Compose
import numpy as np
import open3d as o3d

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

encoder = 'vitl' # or 'vitb', 'vits'
depth_anything = DepthAnything(model_configs[encoder]).cuda()
depth_anything.load_state_dict(torch.load(f'/home/sunh/6D_ws/Fpose_rgb/feature/checkpoints/depth_anything_vitl14.pth'))



transform = Compose([
    Resize(
        width=640,
        height=480,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

for i in range(100):
    color_image = cv2.cvtColor( cv2.imread('/home/sunh/6D_ws/other_network/megapose6d/local_data/examples/mp6d/data/0000/{:06d}-color.png'.format(i)),cv2.COLOR_BGR2RGB )
    # color_image = cv2.cvtColor(cv2.imread('/home/sunh/Project/Depth-Anything-V2/data/images/color_image-1.png'),cv2.COLOR_BGR2RGB)
    image = color_image/ 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).cuda()

    # depth shape: 1xHxW
    features, depth = depth_anything(image)

    depth = depth[0].cpu().detach().numpy()
    depth_image = np.uint8(depth)
    depth_image = cv2.resize(depth_image,(640,480))
    depth = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    cv2.imwrite('/home/sunh/6D_ws/other_network/cnos-main/Depth-Anything/example/{:06d}.png'.format(i), depth_image)
    cv2.imshow('a', depth)
    cv2.waitKey()

    color_image = o3d.io.read_image('/home/sunh/6D_ws/other_network/megapose6d/local_data/examples/mp6d/data/0000/{:06d}-color.png'.format(i))
    # color_image = o3d.io.read_image('/home/sunh/Project/Depth-Anything-V2/data/images/color_image-1.png')
    # depth_image = o3d.io.read_image('/home/sunh/6D_ws/other_network/cnos-main/Depth-Anything/example/{:06d}.png'.format(i))
    depth_image = o3d.io.read_image('/home/sunh/6D_ws/other_network/cnos-main/Depth_Anything/example/000000.png')
    o3d_inter = o3d.camera.PinholeCameraIntrinsic(640, 480, 572.4114,  573.57043,
                                                  325.2611, 242.04899)
    # o3d_inter = o3d.camera.PinholeCameraIntrinsic(640, 480, 387.41, 386.95,
    #                                               318.957,245.277)
    # 点云生成和显示
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1.0 / 1,
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_inter)
    o3d.visualization.draw_geometries([pcd])





# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
#
# import cv2
# import torch
# from torchvision.transforms import Compose
# import numpy as np
# import open3d as o3d
#
#
# depth_anything = DepthAnything.from_pretrained('checkpoints/depth_anything_vitl14', local_files_only=True).cuda()
#
# transform = Compose([
#     Resize(
#         width=28*14,
#         height=28*14,
#         resize_target=False,
#         keep_aspect_ratio=True,
#         ensure_multiple_of=14,
#         resize_method='lower_bound',
#         image_interpolation_method=cv2.INTER_CUBIC,
#     ),
#     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     PrepareForNet(),
# ])
#
# color_image = cv2.cvtColor(cv2.imread('//home/sunh/Project/Depth-Anything-V2/data/imgs/camera_screenshot_15.07.2024.png'   ),
#                      cv2.COLOR_BGR2RGB)
# image = color_image/ 255.0
# image = transform({'image': image})['image']
# image = torch.from_numpy(image).unsqueeze(0).cuda()
#
# # depth shape: 1xHxW
# features, depth = depth_anything(image)
#
# depth = depth[0].cpu().detach().numpy()
# depth_image = np.uint8(depth)
# depth_image = cv2.resize(depth_image,(640,480))
# depth = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
# # cv2.imwrite('/home/sunh/6D_ws/other_network/cnos-main/Depth-Anything/example/{:06d}.png'.format(i), depth_image)
# cv2.imshow('a', depth)
# cv2.waitKey()


