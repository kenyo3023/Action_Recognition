from typing import Dict, Any, List

import cv2
import numpy as np

import torch
from torch.utils.data.dataset import IterableDataset

from action_recognition.datasets.augmentation import default_transformation_2D, default_transformation_3D


class Video_Instance:
    def __init__(self, path: str):
        """
        Handles video reading with cv2.VideoCapture

        :param path: video's absolute path, or 0 for camera input
        :type path: str
        :param half_precision: fp16 inferencing, defaults to False
        :type half_precision: bool, optional
        """
        self.path = path
        self._info: Dict[str, Any] = {}
        self.cap = None
        self.dtype = np.float32

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def get_frames(self):
        """
        A generator generates/iterates frames
        """
        self.cap = cv2.VideoCapture(self.path)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = frame[:, :, ::-1]  # BGR -> RGB
            yield frame
        self.close()

    @property
    def info(self) -> Dict[str, Any]:
        if self._info:
            return self._info
        self._info = {}
        cap = cv2.VideoCapture(self.path)
        if cap.isOpened():
            self._info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._info['fps'] = cap.get(cv2.CAP_PROP_FPS)
            self._info['w'], self._info['h'] = cap.get(3), cap.get(4)
        else:
            raise RuntimeError(f"Video {self.path} cannot capture")
        cap.release()
        return self._info


class Video_2D_Inference(IterableDataset):  # pylint: disable=abstract-method, too-few-public-methods
    def __init__(self, videos: List[Video_Instance]):
        self.videos = videos
        self.transform = default_transformation_2D('test')

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.videos = self.videos[worker_info.id::worker_info.num_workers]  # pylint: disable=no-member

        for instance in self.videos:
            buff = None
            for idx, frame in enumerate(instance.get_frames()):
                frame = self.transform(frame)
                if buff is not None:
                    yield buff
                buff = frame, instance.path, idx, False
            yield buff[:-1] + (True, )


class Video_3D_Inference(IterableDataset):  # pylint: disable=abstract-method, too-few-public-methods
    def __init__(self, videos: List[Video_Instance], window_size: int):
        self.videos = videos
        self.transform = default_transformation_3D('test')
        self.window_size = window_size
        self.window_step = 1

    def sample_batch(self, instance: Video_Instance):
        ret = None
        fill_idx = 0
        next_yield = self.window_size
        roll_size = self.window_size - self.window_step
        for frame in instance.get_frames():
            frame = torch.as_tensor(frame.copy())
            frame = self.transform(frame.unsqueeze(0)).squeeze(1)  # C, T, H, W
            if ret is None:
                ret = torch.empty((roll_size * 2, *frame.shape), dtype=frame.dtype)
            assert 0 <= fill_idx < ret.shape[0]
            ret[fill_idx] = frame
            fill_idx += 1
            next_yield -= 1
            if next_yield == 0:
                yield ret[fill_idx - self.window_size:fill_idx].transpose(0, 1)  # T, C, H, W -> C, T, H, W
                next_yield = self.window_step
            if fill_idx == ret.shape[0]:
                ret[:roll_size] = ret[-roll_size:]
                fill_idx = roll_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.videos = self.videos[worker_info.id::worker_info.num_workers]  # pylint: disable=no-member
        for instance in self.videos:
            buff = None
            for idx, segment in enumerate(self.sample_batch(instance)):
                # segment = self.transform(segment)
                if buff is not None:
                    yield buff
                buff = segment, instance.path, idx, False
            yield buff[:-1] + (True, )
