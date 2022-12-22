import os
import yaml
import numpy as np

from PIL import Image
from easydict import EasyDict
import pdb

from dataloader.pc_dataset import SemanticKITTI
from dataloader.dataset import point_image_dataset_semkitti

import torch

dataset_dir = '/scratch/parth.shah/kitti_odometry/dataset/sequences/'
print("Dataset directory - ", dataset_dir, "\n")

sequence = "01"
instance = "000000"

pcd_file = dataset_dir + sequence +  "/velodyne/" + instance +".bin"
print("PCD filename - ", pcd_file)
raw_pcd = np.fromfile(pcd_file, dtype=np.float32).reshape((-1, 4))
print("PCD shape - " ,raw_pcd.shape)
print()

img_file = dataset_dir + sequence +  "/image_2/" + instance +".png"
print("Image filename - ", img_file)
img = Image.open(img_file)
print()

# label_file =  dataset_dir + sequence +  "/labels/" + instance +".label"
# print("Label filename - ", label_file)
print("Labels are zero arrays")
# annotated_data = np.expand_dims(np.zeros_like(raw_pcd[:, 0], dtype=int), axis=1)
# instance_label = np.expand_dims(np.zeros_like(raw_pcd[:, 0], dtype=int), axis=1)
with open('./config/label_mapping/semantic-kitti.yaml', 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
learning_map = semkittiyaml['learning_map']

annotated_data = np.fromfile(pcd_file.replace('velodyne', 'labels')[:-3] + 'label',
                                dtype=np.uint32).reshape((-1, 1))
instance_label = annotated_data >> 16
annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
annotated_data = np.vectorize(learning_map.__getitem__)(annotated_data)

print()

calib_file = dataset_dir + sequence + "/calib.txt"
print("Calib path - ", calib_file)
calib = SemanticKITTI.read_calib(calib_file)
proj_matrix = np.matmul(calib["P2"], calib["Tr"])

print("Creating pc_data_dict")
data_dict = {}
data_dict['xyz'] = raw_pcd[:, :3]
data_dict['labels'] = annotated_data.astype(np.uint8)
data_dict['instance_label'] = instance_label
data_dict['signal'] = raw_pcd[:, 3:4]
data_dict['origin_len'] = len(raw_pcd)
data_dict['img'] = img
data_dict['proj_matrix'] = proj_matrix


print("Working on data")
xyz = data_dict['xyz']
labels = data_dict['labels']
instance_label = data_dict['instance_label'].reshape(-1)
sig = data_dict['signal']
origin_len = data_dict['origin_len']

ref_pc = xyz.copy()
ref_labels = labels.copy()
ref_index = np.arange(len(ref_pc))


min_volume_space = [-50, -50, -4]
max_volume_space = [50, 50, 2]

mask_x = np.logical_and(xyz[:, 0] > min_volume_space[0], xyz[:, 0] < max_volume_space[0])
mask_y = np.logical_and(xyz[:, 1] > min_volume_space[1], xyz[:, 1] < max_volume_space[1])
mask_z = np.logical_and(xyz[:, 2] > min_volume_space[2], xyz[:, 2] < max_volume_space[2])
mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

xyz = xyz[mask]
ref_pc = ref_pc[mask]
labels = labels[mask]
instance_label = instance_label[mask]
ref_index = ref_index[mask]
sig = sig[mask]
point_num = len(xyz)

image = data_dict['img']
proj_matrix = data_dict['proj_matrix']

keep_idx = xyz[:, 0] > 0
points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
img_points = (proj_matrix @ points_hcoords.T).T
img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
keep_idx_img_pts = point_image_dataset_semkitti.select_points_in_frustum(img_points, 0, 0, *image.size)
keep_idx[keep_idx] = keep_idx_img_pts

img_points = np.fliplr(img_points)
points_img = img_points[keep_idx_img_pts]
img_label = labels[keep_idx]

point2img_index = np.arange(len(labels))[keep_idx]
feat = np.concatenate((xyz, sig), axis=1)

bottom_crop = [480, 320]
left = int(np.random.rand() * (image.size[0] + 1 - bottom_crop[0]))
right = left + bottom_crop[0]
top = image.size[1] - bottom_crop[1]
bottom = image.size[1]

# update image points
keep_idx = points_img[:, 0] >= top
keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

# crop image
image = image.crop((left, top, right, bottom))
points_img = points_img[keep_idx]
points_img[:, 0] -= top
points_img[:, 1] -= left

img_label = img_label[keep_idx]
point2img_index = point2img_index[keep_idx]

img_indices = points_img.astype(np.int64)

image = np.array(image, dtype=np.float32, copy=False) / 255.

print("Creating final data_dict_2")
data_dict_2 = {}
data_dict_2 = {}
data_dict_2['point_feat'] = feat
data_dict_2['point_label'] = labels
data_dict_2['ref_xyz'] = ref_pc
data_dict_2['ref_label'] = ref_labels
data_dict_2['ref_index'] = ref_index
data_dict_2['mask'] = mask
data_dict_2['point_num'] = point_num
data_dict_2['origin_len'] = origin_len
data_dict_2['root'] = dataset_dir

data_dict_2['img'] = image
data_dict_2['img_indices'] = img_indices
data_dict_2['img_label'] = img_label
data_dict_2['point2img_index'] = point2img_index

print(data_dict_2.keys())


import network.arch_2dpass as model_file
from main import load_yaml

print("Loading Configs")
config_file = "./config/2DPASS-semantickitti.yaml"
checkpoint_file = "./pretrained/semantickitti/best_model.ckpt"
configs = load_yaml(config_file)
configs["checkpoint"] = checkpoint_file

configs["submit_to_server"] = False
configs["baseline_only"] = False
configs["pretrain2d"] = False
configs = EasyDict(configs)

print()
print("Loading Model")
my_model = model_file.get_model(configs)
my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs, strict=(not configs.pretrain2d))
my_model.cuda()

from dataloader.dataset import collate_fn_default

data_dict_final = collate_fn_default([data_dict_2])

for key in data_dict_final.keys():
    if type(data_dict_final[key]) == torch.Tensor:
        data_dict_final[key] = data_dict_final[key].cuda()

print("Running forward function")
# pdb.set_trace()

data_dict_copy = data_dict_final.copy()
ret_dict = my_model(data_dict_final)
print("Returned values")
print(ret_dict.keys())

logits = ret_dict['logits']
predicted_labels = torch.argmax(logits, dim=1)
dataset_labels = ret_dict['labels']

print("Dataset")
print(dataset_labels.bincount())


print("Predicted")
print(predicted_labels.bincount())

pdb.set_trace()

