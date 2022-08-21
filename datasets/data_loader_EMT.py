import os
import random
import numpy as np
from shutil import copyfile, move
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from datasets import data_transform as tr


def read_npy(root_path, split='train'):
    files = []

    files_root = os.path.join(root_path, split)

    for file_name in os.listdir(files_root):
        file_path = os.path.join(files_root, file_name)
        files.append(file_path)
    return files


def load_data_npy(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    # img = Image.fromarray(data['img'][:])
    img = Image.fromarray(np.uint8(data['img'][:])).convert('RGB')
    # img = Image.fromarray(np.uint8(data['img']))
    mask = np.array(data['masks'][:], dtype=bool)
    # use the masks instead of outlines
    mask[mask > 0] = 1
    mask = Image.fromarray(np.uint8(mask))
    return img, mask


# def read_img(root_path, split='test'):
#     images = []
#     masks = []
#
#     image_root = os.path.join(root_path, split + '/images')
#     gt_root = os.path.join(root_path, split + '/labels')
#
#     for image_name in os.listdir(image_root):
#         image_path = os.path.join(image_root, image_name)
#         label_path = os.path.join(gt_root, image_name)
#
#         images.append(image_path)
#         masks.append(label_path)
#
#     return images, masks


# def load_data_img(img_path, mask_path):
#     img = Image.open(img_path).convert('RGB')
#     mask = Image.open(mask_path)
#     mask = np.array(mask)
#     mask[mask > 0] = 1  # 这里我把255转到了1
#     mask = Image.fromarray(np.uint8(mask))
#     return img, mask


# class DataLoaderIMG(data.Dataset):
#     def __init__(self, args, split='test'):
#         self.args = args
#         self.root = self.args.root_path
#         self.split = split
#         self.images, self.masks = read_img(self.root, self.split)
#
#     def transform_tr(self, sample):
#         composed_transforms = transforms.Compose([
#             tr.RandomHorizontalFlip(),
#             tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.img_size),
#             tr.RandomGaussianBlur(),
#             tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#             tr.ToTensor()
#         ])
#         return composed_transforms(sample)
#
#     def transform_val(self, sample):
#         composed_transforms = transforms.Compose([
#             tr.FixScaleCrop(crop_size=self.args.img_size),
#             tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#             tr.ToTensor()
#         ])
#         return composed_transforms(sample)
#
#     def __getitem__(self, index):
#         img, mask = load_data_img(self.images[index], self.labels[index])
#         if self.split == "train":
#             sample = {'image': img, 'label': mask}
#             return self.transform_tr(sample)
#         elif self.split == 'val':
#             img_name = os.path.split(self.images[index])[1]
#             sample = {'image': img, 'label': mask}
#             sample_ = self.transform_val(sample)
#             sample_['case_name'] = img_name[0:-4]
#             return sample_
#         # return sample
#
#     def __len__(self):
#         assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
#         return len(self.images)


class LoadData(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.root = self.args.root_path
        self.split = split
        # self.images, self.labels = read_own_data(self.root, self.split)
        self.files = read_npy(self.root, self.split)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.img_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.img_size),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def __getitem__(self, index):
        # img, mask = own_data_loader(self.images[index], self.labels[index])
        img, mask = load_data_npy(self.files[index])
        if self.split == "train":
            sample = {'image': img, 'label': mask}
            return self.transform_tr(sample)
        elif self.split == 'val':
            # img_name = os.path.split(self.images[index])[1]
            img_name = os.path.split(self.files[index])[1]
            sample = {'image': img, 'label': mask}
            sample_ = self.transform_val(sample)
            sample_['case_name'] = img_name[0:-4]
            return sample_
        # return sample

    def __len__(self):
        return len(self.files)
        # assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        # return len(self.images)
