# The following code is borrwed and modified as necessary
# from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
import cv2
import torch
import random
import numbers
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import randperm
from torch._utils import _accumulate
from torchvision import datasets, transforms
import torchvision.transforms.functional as F


class CustomDataset(Dataset):
    """ Load Customer Dataset
    Args:
        csv_file (String): CSV file name containing annotation data.
                           [annotation_type, filename, x1, y1, x2, y2, center_x, center_y, width, height]
        root_dir (String): root directory in which image files are kept
        transform (List Object): list of transforms to use on the input data
        idx (int): index of dataset for which to fetch data
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotation_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_data)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, self.annotation_data.iloc[idx, 1])
        image = Image.open(img_name)
        bbox = np.array(self.annotation_data.iloc[idx, 2:]).astype(
            float).reshape(-1, 2)
        sample = {'image': image, 'bbox': bbox}

        if self.transform:
            if type(self.transform) is not list:
                self.transform = [self.transform]
            for idx in range(len(self.transform)):
                sample = self.transform[idx](sample)

        return sample


class ToTensor(object):
    """Convert ndarrays or PIL image in sample to Tensors."""

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = F.to_tensor(image)
        return {'image': image, 'bbox': torch.from_numpy(bbox)}


class Resize(object):
    """Resize the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        w, h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = F.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        bbox[0] = np.round(bbox[0] * np.array([new_w / w, new_h / h]), 0)
        bbox[1] = np.round(bbox[1] * np.array([new_w / w, new_h / h]), 0)
        bbox[3] = np.abs([bbox[0, 0]-bbox[1, 0], bbox[0, 1]-bbox[1, 1]])
        bbox[2] = np.array([bbox[0, 0]+bbox[3, 0]/2, bbox[0, 1]+bbox[3, 1]/2])

        return {'image': img, 'bbox': bbox}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            sample (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        image, bbox = sample['image'], sample['bbox']
        img = F.normalize(image, self.mean, self.std)
        return {'image': img, 'bbox': bbox}


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        _, h = image.size

        if random.random() < self.p:
            image = F.vflip(image)
            if bbox[2][0] > 0 and bbox[2][1] > 0:
                bbox[0][1] = h-1-bbox[0][1]-bbox[3][1]
                bbox[1][1] = h-1-bbox[1][1]+bbox[3][1]
                bbox[2][1] = h-1-bbox[2][1]

        return {'image': image, 'bbox': bbox}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        w, _ = image.size

        if random.random() < self.p:
            image = F.hflip(image)
            if bbox[2][0] > 0 and bbox[2][1] > 0:
                bbox[0][0] = w-1-bbox[0][0]-bbox[3][0]
                bbox[1][0] = w-1-bbox[1][0]+bbox[3][0]
                bbox[2][0] = w-1-bbox[2][0]

        return {'image': image, 'bbox': bbox}


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(
                    "{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        tforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            tforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            tforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            tforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            tforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(tforms)
        transform = transforms.Compose(tforms)

        return transform

    def __call__(self, sample):
        """
        Args:
            sample (List): List of Input image and bounding box

        Returns:
            List: Color jittered image and original bounding box.
        """
        image, bbox = sample['image'], sample['bbox']
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        image = transform(image)

        return {'image': image, 'bbox': bbox}


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + \
            " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx].tolist()]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
