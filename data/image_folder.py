"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import glob


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

"""
def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
"""

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for path in sorted(glob.iglob(os.path.join(dir, '**'), recursive=True)):
        if is_image_file(os.path.basename(path)):
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def mix_make_dataset(dir, max_dataset_size=float("inf")):
    cropped_images = []
    whole_images = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    # cropped
    cropped_dir = os.path.join(dir, 'cropped')
    for path in sorted(glob.iglob(cropped_dir + '/*')):
        if is_image_file(os.path.basename(path)):
            cropped_images.append(path)

    # whole
    whole_dir = os.path.join(dir, 'whole')
    for path in sorted(glob.iglob(whole_dir + '/*')):
        if is_image_file(os.path.basename(path)):
            whole_images.append(path)

    images = cropped_images + whole_images   # Always First half is cropped, back half is whole.

    total_num_cropped = len(cropped_images)
    total_num_whole = len(whole_images)
    return images[:min(max_dataset_size, len(images))], total_num_cropped, total_num_whole 


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
