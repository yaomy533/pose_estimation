
import numpy as np
import math
import cv2, os


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2],
                        # [0.07,0,0],
                        # [0,0.07,0],
                        # [0,0,0.07],
                        # [0,0,0]
                        ]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def align_rotation(sRT):
    """ Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """
    s = np.cbrt(np.linalg.det(sRT[:3, :3]))
    R = sRT[:3, :3] / s
    T = sRT[:3, 3]

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                      [0.0,            1.0,  0.0           ],
                      [theta_y/r_norm, 0.0,  theta_x/r_norm]])
    rotation = R @ s_map
    aligned_sRT = np.identity(4, dtype=np.float32)
    aligned_sRT[:3, :3] = s * rotation
    aligned_sRT[:3, 3] = T
    return aligned_sRT


def draw_bboxes(img, img_pts, color, gt=False, width=2):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, width)
    # draw pillars in minor darker color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, width)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, width)

    # if gt:
    #     img = cv2.line(img, tuple(img_pts[11]), tuple(img_pts[8]), (255*0.6,0,0), 2)
    #     img = cv2.line(img, tuple(img_pts[11]), tuple(img_pts[9]), (0,255*0.6,0), 2)
    #     img = cv2.line(img, tuple(img_pts[11]), tuple(img_pts[10]), (0,0,255*0.6), 2)
    # else:
    #     img = cv2.line(img, tuple(img_pts[11]), tuple(img_pts[8]), (255,0,0), 2)
    #     img = cv2.line(img, tuple(img_pts[11]), tuple(img_pts[9]), (0,255,0), 2)
    #     img = cv2.line(img, tuple(img_pts[11]), tuple(img_pts[10]), (0,0,255), 2)


    return img


def draw_detections(img, out_dir, data_name, intrinsics, pred_sRT=None, pred_size=None, pred_class_ids=None,
                    gt_sRT=None, gt_size=None, gt_class_ids=None):
    """ Visualize pose predictions.
    """
    out_path = os.path.join(out_dir, data_name)

    img = img.copy()    
    # darw ground truth - GREEN color
    if gt_sRT is not None:
        for i in range(gt_sRT.shape[0]):
            if gt_class_ids[i] in [0,1,3]:
                sRT = align_rotation(gt_sRT[i, :, :])
            else:
                sRT = gt_sRT[i, :, :]
            bbox_3d = get_3d_bbox(gt_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (0, 255, 0), gt=True, width=2 if gt_class_ids[i]==4 else 1 )
    # darw prediction - RED color
    if pred_sRT is not None:
        for i in range(pred_sRT.shape[0]):
            # if pred_class_ids[i] != 4 : continue

            if pred_class_ids[i] in [0,1,3]:
                sRT = align_rotation(pred_sRT[i, :, :])
            else:
                sRT = pred_sRT[i, :, :]
            bbox_3d = get_3d_bbox(pred_size[i, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, sRT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            img = draw_bboxes(img, projected_bbox, (255, 0, 0), width=2 if gt_class_ids[i]==4 else 1 )
            # img = draw_bboxes(img, projected_bbox, (0, 255, 0))

    if pred_sRT is not None or gt_sRT is not None:
        cv2.imwrite(out_path, img[:, :, (2, 1, 0)])
    # cv2.imshow('vis', img)
    # cv2.waitKey(0)
