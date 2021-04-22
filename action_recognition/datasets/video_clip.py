import os

import cv2
import torch
from tqdm import tqdm
from torchvision.datasets.video_utils import VideoClips


class Video_Frame_Clips(VideoClips):
    @staticmethod
    def vid2framedir(video_path):
        return os.path.join(os.path.dirname(video_path), 'jpg', os.path.basename(video_path).split('.')[0])

    def __init__(self, video_paths, frame_mode=False, **kwargs):
        self.frame_mode = frame_mode
        super().__init__(video_paths, **kwargs)

    def _init_from_metadata(self, metadata):
        super()._init_from_metadata(metadata)

        if not self.frame_mode and all(os.path.exists(self.vid2framedir(f)) for f in tqdm(self.video_paths)):
            self.frame_mode = True
            self.video_pts = [torch.as_tensor(list(range(len(pts)))) for pts in self.video_pts]

    def subset(self, indices):
        video_paths = [self.video_paths[i] for i in indices]
        video_pts = [self.video_pts[i] for i in indices]
        video_fps = [self.video_fps[i] for i in indices]
        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps
        }
        return type(self)(video_paths,
                          clip_length_in_frames=self.num_frames,
                          frames_between_clips=self.step,
                          frame_rate=self.frame_rate,
                          _precomputed_metadata=metadata, num_workers=self.num_workers,
                          _video_width=self._video_width,
                          _video_height=self._video_height,
                          _video_min_dimension=self._video_min_dimension,
                          _audio_samples=self._audio_samples,
                          _audio_channels=self._audio_channels,
                          frame_mode=self.frame_mode)

    def read_video_by_frame(self, video_path, start, end):
        dirname = self.vid2framedir(video_path)
        fnames = [os.path.join(dirname, f'img_{i:06d}.jpg') for i in range(start, end + 1)]
        images = [cv2.imread(fn) for fn in fnames]
        if any(im is None for im in images):
            print(video_path, start, end)
        images = [torch.from_numpy(img[:, :, ::-1].copy()) for img in images]  # list[np.array H, W, C]
        video = torch.stack(images)  # T, H, W, C
        return video

    def get_clip(self, idx):
        if self.frame_mode:
            video_idx, clip_idx = self.get_clip_location(idx)
            video_path = self.video_paths[video_idx]
            clip_pts = self.clips[video_idx][clip_idx]
            start_pts = clip_pts[0].item()
            end_pts = clip_pts[-1].item()
            video = self.read_video_by_frame(video_path, start_pts, end_pts)
            return video, None, None, video_idx
        return super().get_clip(idx)
