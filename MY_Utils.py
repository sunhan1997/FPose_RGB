import torch
import cv2
import geomloss
import numpy as np



############################################## set OT Loss #################################
p = 2
entreg = .1 # entropy regularization factor for Sinkhorn
OTLoss = geomloss.SamplesLoss(
    loss='sinkhorn', p=p,
    # 对于p=1或p=2的情形
    cost=geomloss.utils.distances if p==1 else geomloss.utils.squared_distances,
    blur=entreg**(1/p), backend='tensorized')
############################################## set OT Loss #################################

############################################## some def #################################
def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]

def create_bounding_box(img, pose, pt_cld_data, intrinsic_matrix,color=(0,0,255)):
    "Create a bounding box around the object"
    # 8 corner points of the ptcld data
    min_x, min_y, min_z = np.min(pt_cld_data['x']), np.min(pt_cld_data['y']),np.min(pt_cld_data['z'])
    max_x, max_y, max_z = np.max(pt_cld_data['x']), np.max(pt_cld_data['y']),np.max(pt_cld_data['z'])

    corners_3D = np.array([[max_x, min_y, min_z],
                           [max_x, min_y, max_z],
                           [min_x, min_y, max_z],
                           [min_x, min_y, min_z],
                           [max_x, max_y, min_z],
                           [max_x, max_y, max_z],
                           [min_x, max_y, max_z],
                           [min_x, max_y, min_z]])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)

    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), color, 1)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), color, 1)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[4]), color, 1)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[2]), color, 1)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[5]), color, 1)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[3]), color, 1)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[6]), color, 1)
    img = cv2.line(img, tuple(coord_2D[3]), tuple(coord_2D[7]), color, 1)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[7]), color, 1)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[5]), color, 1)
    img = cv2.line(img, tuple(coord_2D[5]), tuple(coord_2D[6]), color, 1)
    img = cv2.line(img, tuple(coord_2D[6]), tuple(coord_2D[7]), color, 1)

    return img
############################################## some def #################################



################ fast rcnn ######################
def reset_object(mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    model_center = (min_xyz + max_xyz) / 2
    if mesh is not None:
        mesh_ori = mesh.copy()
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - model_center.reshape(1, 3)

    return mesh,model_center
def get_tf_to_centered_mesh(model_center):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(model_center, device='cuda', dtype=torch.float)
    return tf_to_center




