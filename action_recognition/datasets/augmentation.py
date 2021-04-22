import random

import torch
from torchvision import transforms
from torchvision.transforms import _transforms_video as transforms_video


class RandomVerticalFlipVideo:
    """
    Flip the video clip along the vertical direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip: torch.Tensor):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if random.random() < self.p:
            clip = clip.flip(-2)
        return clip

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class VideoClipResize:  # pylint: disable=too-few-public-methods
    # https://github.com/facebookresearch/ClassyVision/blob/master/classy_vision/dataset/transforms/util_video.py#L108
    def __init__(self, size: int, interpolation_mode: str = "bilinear"):
        self.interpolation_mode = interpolation_mode
        self.size = size

    def __call__(self, clip):
        # clip size: C x T x H x W
        if not min(clip.size()[2], clip.size()[3]) == self.size:
            new_h, new_w = self._get_rescaled_size(self.size, clip.size()[2], clip.size()[3])
            clip = torch.nn.functional.interpolate(
                clip, size=(new_h, new_w), mode=self.interpolation_mode
            )
        return clip

    @staticmethod
    def _get_rescaled_size(scale, h, w):
        if h < w:
            new_h = scale
            new_w = int(scale * w / h)
        else:
            new_w = scale
            new_h = int(scale * h / w)
        return new_h, new_w

    def __repr__(self):
        return f"VideoClipResize({self.size})"


imagenet_transform_dict = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

kinetics400_transform_dict = {
    "mean": [0.43216, 0.394666, 0.37645],
    "std": [0.22803, 0.22145, 0.216989]
}


def default_transformation_2D(split, size=224):
    return {
        "train": transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180, expand=True),
            transforms.Resize(size),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_transform_dict["mean"], imagenet_transform_dict["std"]),
        ]),
        "valid": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_transform_dict["mean"], imagenet_transform_dict["std"]),
        ]),
        "test": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_transform_dict["mean"], imagenet_transform_dict["std"]),
        ]),
    }[split]


def default_transformation_3D(split, size=224):
    return {
        "train": transforms.Compose([
            transforms_video.ToTensorVideo(),
            transforms_video.RandomResizedCropVideo(size),
            RandomVerticalFlipVideo(),
            transforms_video.RandomHorizontalFlipVideo(),
            transforms_video.NormalizeVideo(kinetics400_transform_dict["mean"], kinetics400_transform_dict["std"]),
        ]),
        "valid": transforms.Compose([
            transforms_video.ToTensorVideo(),
            VideoClipResize(size),  # not square
            transforms_video.CenterCropVideo(size),
            transforms_video.NormalizeVideo(kinetics400_transform_dict["mean"], kinetics400_transform_dict["std"]),
        ]),
        "test": transforms.Compose([
            transforms_video.ToTensorVideo(),
            VideoClipResize(size),  # not square
            transforms_video.CenterCropVideo(size),
            transforms_video.NormalizeVideo(kinetics400_transform_dict["mean"], kinetics400_transform_dict["std"]),
        ]),
    }[split]
