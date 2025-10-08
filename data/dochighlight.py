import os
import glob
import random
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class DocHighlightDataset(Dataset):
    def __init__(self, base_dir='/media/data2/xhw/final', split='train', spatial_transform=False, patch_size=(512,512)):
        self.base_dir = base_dir
        self.split = split
        self.spatial_transform = spatial_transform
        self.patch_size = patch_size
        folder_path = os.path.join(self.base_dir, self.split)
        self.degraded_images = sorted(glob.glob(os.path.join(folder_path, "*_in.*")))

    def __len__(self):
        return len(self.degraded_images)
    
    def __getitem__(self, idx):
        degraded_path = self.degraded_images[idx]
        gt_path = degraded_path.replace('_in', '_gt')
        degraded_image, gt_image = self.load_images(degraded_path, gt_path)
        degraded_image, gt_image = self.apply_transforms(degraded_image, gt_image)
        degraded_image = transforms.ToTensor()(degraded_image)
        gt_image = transforms.ToTensor()(gt_image)
        if self.split != 'test':
            degraded_image, gt_image = self.extract_random_patch(degraded_image, gt_image)
        if self.split == 'test':
            image_name = os.path.basename(degraded_path)
            return degraded_image, gt_image, image_name
        return degraded_image, gt_image

    def load_images(self, degraded_path, gt_path):
        degraded_image = cv2.imread(degraded_path)
        degraded_image = cv2.cvtColor(degraded_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.imread(gt_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        return degraded_image, gt_image
    
    def apply_spatial_transform(self, degraded_image, gt_image):
        transform_functions = {
            'none': lambda img: img,
            'rot90': lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            'rot180': lambda img: cv2.rotate(img, cv2.ROTATE_180),
            'rot270': lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }
        spatial_transform_type = random.choice(list(transform_functions.keys()))
        degraded_image = transform_functions[spatial_transform_type](degraded_image)
        gt_image = transform_functions[spatial_transform_type](gt_image)
        return degraded_image, gt_image

    def apply_transforms(self, degraded_image, gt_image):
        if self.spatial_transform:
            degraded_image, gt_image = self.apply_spatial_transform(degraded_image, gt_image)
        return degraded_image, gt_image

    def extract_random_patch(self, degraded_image, gt_image):
        _, h, w = degraded_image.shape
        patch_h, patch_w = self.patch_size
        if h < patch_h:
            pad_height = patch_h - h
            degraded_image = transforms.functional.pad(degraded_image, (0, 0, 0, pad_height), padding_mode='constant', fill=1)
            gt_image = transforms.functional.pad(gt_image, (0, 0, 0, pad_height), padding_mode='constant', fill=1)
            h = patch_h 
        if w < patch_w:
            pad_width = patch_w - w
            degraded_image = transforms.functional.pad(degraded_image, (0, pad_width, 0, 0), padding_mode='constant', fill=1)
            gt_image = transforms.functional.pad(gt_image, (0, pad_width, 0, 0), padding_mode='constant', fill=1)
            w = patch_w
        
        top = random.randint(0, h - patch_h)
        left = random.randint(0, w - patch_w)
        degraded_patch = degraded_image[:, top:top + patch_h, left:left + patch_w]
        gt_patch = gt_image[:, top:top + patch_h, left:left + patch_w]
        return degraded_patch, gt_patch

if __name__ == "__main__":
    dataset = DocHighlightDataset(split='test')
    path = 'test_images'
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(dataset)):
        result = dataset[i]
        if dataset.split == 'test':
            degraded_patch, gt_patch, image_name = result
        else:
            degraded_patch, gt_patch = result
        degraded_image_np = degraded_patch.permute(1, 2, 0).numpy() * 255
        gt_image_np = gt_patch.permute(1, 2, 0).numpy() * 255
        if degraded_image_np.shape != gt_image_np.shape:
            print(i)
            print(degraded_image_np.shape)
            print(gt_image_np.shape)

