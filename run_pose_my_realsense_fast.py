import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import geomloss
import os
from plyfile import PlyData
from feature.depth_anything.dpt import DepthAnything
import pyrealsense2 as rs
import cv2
import numpy as np

from MY_Utils import *
from Utils import *


#相机配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
#相机深度参数，包括精度以及 depth_scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 3)
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 8  # 8 meter
clipping_distance = clipping_distance_in_meters / depth_scale
#color和depth对齐
align_to = rs.stream.color
align = rs.align(align_to)
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Start Detection >>>>>>>>>>>>>>>>>>>>>>>>>>>>')



torch.backends.cudnn.enabled = False
from dpt_infer import Dpt
engine_path = r"/home/sunh/6D_ws/Fpose_rgb/feature2/depth_anything_v2_vitl.engine"
grayscale = False
# 加载并运行模型
model = Dpt(engine_path)


############################################## set for dinov2 #################################
# 设置补丁(patch)的高度和宽度
patch_h = 28
patch_w = 28
# 特征维度
feat_dim = 1024

# 定义图像转换操作
transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)),  # 调整图像大小
    T.ToTensor(),  # 转换为张量
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
])

from torchvision.transforms import Compose
from feature.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
transform_dapth = Compose([
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

#  load model
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

encoder = 'vitl' # or 'vitb', 'vits'
dinov2_vits14 = DepthAnything(model_configs[encoder]).cuda()
dinov2_vits14.load_state_dict(torch.load(f'/home/sunh/6D_ws/Fpose_rgb/feature/checkpoints/depth_anything_vitl14.pth'))
dinov2_vits14.eval()

############################################## set for dinov2 #################################


tamplate_number = 162
############################################## get the features of the template #################################
ref_features = torch.zeros(tamplate_number, 784,1024).cuda() # 162 242
for i in range(tamplate_number):
    ref_tensor = torch.zeros(1, 3, patch_h * 14, patch_w * 14).cuda()
    ref_img_path = './weights/tmp/{:06d}.png'.format(i)
    # 打开图像并转换为RGB模式
    ref_img = Image.open(ref_img_path).convert('RGB')
    bbox = ref_img.getbbox()
    ref_img = np.array(ref_img)
    ref_img = ref_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    ref_img = Image.fromarray(ref_img)
    # 对图像进行转换操作，并将其存储在imgs_tensor的第一个位置
    ref_tensor[0] = transform(ref_img)[:3]

    with torch.no_grad():
        features_dict,_ = dinov2_vits14(ref_tensor)
        ref_feature = features_dict[0][0]  # 12 12 24
        ref_features[i] = ref_feature

############################################## get the features of the template #################################

####### intrinsic_matrix and read CAD model
intrinsic_matrix = np.array(
    [[597.119, 0.0, 325.671], [0.0, 597.461, 236.537], [0.0, 0.0, 1.0]]     ## L515
)

## start
model_path = os.path.join('./demo_data/tm2/mesh/tm2.ply')
ply = PlyData.read(model_path)
pt_cld_data = ply.elements[0].data
all_pose = []





################ FoundationPose ######################
from estimater import *
from datareader import *
device = 'cuda:0'
iteration = 5
refiner = PoseRefinePredictor()
refiner.model.to(device)

mesh = trimesh.load('./demo_data/tm2/mesh/tm2.ply')
glctx = dr.RasterizeCudaContext(device)

mesh,model_center = reset_object(mesh)
mesh_tensors = make_mesh_tensors(mesh)
for k in mesh_tensors:
    mesh_tensors[k] = mesh_tensors[k].to(device)

tf_to_center = get_tf_to_centered_mesh(model_center)
diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox_show = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

scorer = ScorePredictor()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                     refiner=refiner, debug_dir=f'{code_dir}/debug', debug=0, glctx=glctx)
################ FoundationPose ######################



# 定义全局变量
drawing = False  # 是否正在绘制
start_x, start_y = -1, -1  # 开始点
end_x, end_y = -1, -1  # 结束点

# 鼠标回调函数
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        # 画出矩形
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow('image', img)

obj_i= 1
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    #读取图像
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    #读取内参
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics



    start = time.time()

    rgb = color_image
    color = rgb.copy()
    depth = model.run(color)
    depth = cv2.resize(depth, (640, 480))
    depth_show = np.uint8(depth)
    depth_show = cv2.applyColorMap(depth_show, cv2.COLORMAP_JET)


    if obj_i  == 1:
        img = color_image.copy()
        if img is None:
            print("Error: Could not load image.")
            exit()
        # 创建窗口
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_rectangle)
        # 显示图像
        cv2.imshow('image', img)
        # 等待用户完成选择
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 按 'q' 或 'Esc' 键退出
                break
        # 生成 mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 255, -1)


        start = time.time()


        mask_test = mask

        ys, xs = np.nonzero(mask_test > 0)
        tar_bbox = calc_2d_bbox(xs, ys, (640, 480))
        x1, y1, w1, h1 = tar_bbox

        tar_tensor = torch.zeros(1, 3, patch_h * 14, patch_w * 14).cuda()

        img = rgb.copy()
        img_cv2 = img[int(y1):int(y1 + h1), int(x1): int(x1 + w1)]


        cv2.imshow('img_cv2', img_cv2)
        cv2.waitKey(0)

        img_tensor = Image.fromarray(img_cv2)
        tar_tensor[0] = transform(img_tensor)[:3]


        features_dict = dinov2_vits14(tar_tensor)
        target_feature = features_dict[0]  # 12 12 24
        target_feature = target_feature[0][0][0]



        score_list = []
        for i in range(0, tamplate_number):
            ref_feature = ref_features[i]
            score = OTLoss(target_feature, ref_feature)
            score = score.detach().cpu().numpy()
            score_list.append(score)
        score_idx_5 = np.argsort(np.array(score_list))[:3]

        img_h, img_w, _ = img_cv2.shape


        match_idx = 0
        color_ref = cv2.imread(
            './weights/tmp/{:06d}.png'.format(score_idx_5[match_idx]))
        coor_ref = np.load(
            './weights/tmp/{:06d}.npy'.format(score_idx_5[match_idx]),allow_pickle=True)
        ys, xs = np.nonzero(color_ref[:, :, 0] > 0)
        ref_bbox = calc_2d_bbox(xs, ys, (640, 480))  ## linemod
        x, y, w, h = ref_bbox
        coor_ref = coor_ref[int(y):int(y + h), int(x): int(x + w)]
        color_ref_patch = color_ref[int(y):int(y + h), int(x): int(x + w)]



        mapping_2d = []
        mapping_3d = []

        coor_ref = cv2.resize(coor_ref, (w1, h1), interpolation=cv2.INTER_NEAREST)  # w h
        img[int(y1):int(y1 + h1), int(x1): int(x1 + w1)] = coor_ref

        y_all, x_all = np.where(coor_ref[:, :, 0] > 0)
        for i_coor in range(len(y_all)):
            x, y = x_all[i_coor], y_all[i_coor]
            coord_3d = coor_ref[y, x]
            coord_3d = coord_3d
            mapping_2d.append((x + x1, y + y1))
            mapping_3d.append(coord_3d)


        ############################ PnP Ransac #####################
        try:
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(mapping_3d, dtype=np.float32),
                                                          np.array(mapping_2d, dtype=np.float32), intrinsic_matrix,
                                                          distCoeffs=None,
                                                          iterationsCount=50, reprojectionError=1.0,
                                                          flags=cv2.SOLVEPNP_P3P)
            R, _ = cv2.Rodrigues(rvecs)
            pred_pose = np.concatenate([R, tvecs.reshape((3, 1))], axis=-1)

            img_show = create_bounding_box(color, pred_pose, pt_cld_data, intrinsic_matrix, color=(0, 0, 255))  # red

            cv2.imshow('color_ref_patch', color_ref_patch)
            cv2.imshow('img', img)
            cv2.imshow('img_show', img_show)

            cv2.waitKey(0)

            print(pred_pose)
            pred_pose[:3,3] = pred_pose[:3,3] #/1000.0

        except Exception as e:
            print('PNP error')


        pred_pose_four = np.identity(4)
        pred_pose_four[:3,:4] = pred_pose
        model_center_np = np.identity(4)
        model_center_np[:3, 3] = np.array(model_center)
        pred_pose_four = pred_pose_four @ model_center_np

        print(pred_pose_four)


        color = rgb.copy()
        color_show = rgb.copy()
        xyz_map = depth2xyzmap(depth, intrinsic_matrix)

        pose = est.register_rgb(K=intrinsic_matrix, rgb=rgb, depth=depth, ob_mask=mask_test, iteration=5, guess_pose = pred_pose_four)
        center_pose = np.dot( pose, np.linalg.inv(model_center_np))
        img = create_bounding_box(rgb, center_pose[:3,:4], pt_cld_data, intrinsic_matrix, color=(0, 255, 0))  # green


        cv2.imshow('img_refine', img)
        cv2.waitKey(0)

        obj_i = 2


    else:

        pose, vis = est.track_one(rgb=color, depth=depth, K=intrinsic_matrix, iteration=2)
        center_pose = np.dot( pose, np.linalg.inv(model_center_np))
        print("********************************************************************************************: ", time.time()-start)
        img = create_bounding_box(rgb, center_pose[:3,:4], pt_cld_data, intrinsic_matrix, color=(0, 255, 0))  # green
        cv2.imshow('a', depth_show)
        cv2.imshow('color_ref_patch', color_ref_patch)
        cv2.imshow('img', img)
        cv2.waitKey(1)



