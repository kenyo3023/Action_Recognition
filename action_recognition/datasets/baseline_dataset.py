import os
import logging
import json
import math
import random
import pprint
from typing import Callable, Optional, List, Dict, Any

# import tqdm.auto as tqdm
import wandb
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler, random_split, SubsetRandomSampler
# from torchvision import transforms
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.samplers import RandomClipSampler  # , UniformClipSampler

from action_recognition.datasets.augmentation import default_transformation_2D, default_transformation_3D
from action_recognition.datasets.video_annotation_dataset import \
    extract_metadata_from_annotations, read_annotations, read_tuple
from action_recognition.datasets.breakfast import get_breakfast_dataset
from action_recognition.datasets.mpii import get_mpii_cooking_dataset
from action_recognition.datasets.video_clip import Video_Frame_Clips

logger = logging.getLogger(__name__)


# from classy_vision.dataset.classy_video_dataset import MaxLengthClipSampler
class MaxLengthClipSampler(Sampler):
    def __init__(self, clip_sampler, num_samples=None):
        super().__init__(None)
        self.clip_sampler = clip_sampler
        self.num_samples = num_samples

    def __iter__(self):
        num_samples = len(self)
        n = 0
        for clip in self.clip_sampler:
            if n < num_samples:
                yield clip
                n += 1
            else:
                break

    def __len__(self):
        full_size = len(self.clip_sampler)
        if self.num_samples is None:
            return full_size

        return min(full_size, self.num_samples)


def BalancedSampler(labels: List, num_samples=None, log_weight=False):
    if log_weight:
        label2weight = {a: math.log(labels.count(a)) / labels.count(a) for a in set(labels)}
    else:
        label2weight = {a: 1.0 / labels.count(a) for a in set(labels)}

    weights = [label2weight[lab] for lab in labels]
    sampler = WeightedRandomSampler(weights, num_samples)
    return sampler


def BalancedClipSampler(video_clips: VideoClips, clip_labels: List[int], num_samples=None, log_weight=False):
    assert len(video_clips.clips) == len(clip_labels)
    vc_labels = [clip_labels[video_clips.get_clip_location(idx)[0]] for idx in range(video_clips.num_clips())]
    if num_samples is None:
        num_samples = len(video_clips.video_paths)
    return BalancedSampler(vc_labels, num_samples, log_weight)


def BalancedPathSampler(video_clips: VideoClips, clip_labels: List[int], num_samples=None, log_weight=False):
    assert len(video_clips.clips) == len(clip_labels)
    vc_labels = []
    for idx in range(video_clips.num_clips()):
        vidx, _ = video_clips.get_clip_location(idx)
        vc_labels.append((clip_labels[vidx], video_clips.video_paths[vidx]))

    if num_samples is None:
        num_samples = len(video_clips.video_paths)
    return BalancedSampler(vc_labels, num_samples, log_weight)


def DownsampleClipSampler(video_clips: VideoClips, labels: List[int]):
    vc_labels = [labels[video_clips.get_clip_location(idx)[0]] for idx in range(video_clips.num_clips())]
    cnt = min(vc_labels.count(a) for a in set(labels))
    indices = []
    for a in set(labels):
        indices += random.sample([i for i, c in enumerate(vc_labels) if c == a], cnt)
    return SubsetRandomSampler(indices)


class MouseClipDataset(Dataset):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
            self,
            metadata: Dict[str, List],
            labels: List[int],
            dataset_root: Optional[str],
            output_type: str,
            frames_per_clip: int,
            step_between_clips: int,
            num_sample_per_clip: int,
            frame_rate: Optional[int],
            transform: Optional[List[Callable]] = None,
            split_data: Optional[Dict[str, List[str]]] = None,
            in_memory: bool = False,
    ):
        assert output_type in ['video', 'random_frame']
        self.dataset_root = dataset_root
        self.output_type = output_type
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_sample_per_clip = num_sample_per_clip
        self.frame_rate = frame_rate
        self.split_data = split_data

        # self.video_clips = VideoClips(
        self.video_clips = Video_Frame_Clips(
            metadata['video_paths'],
            clip_length_in_frames=frames_per_clip,
            frames_between_clips=step_between_clips,
            frame_rate=frame_rate,
            _precomputed_metadata=metadata,
        )
        self.all_metadata = metadata
        assert len(labels) == len(metadata['video_paths'])
        self.labels = labels
        self.in_memory = in_memory
        self.transform = transform

        self.cache: Dict[int, Any] = {}
        # if in_memory:
        #     for idx, item in enumerate(self):
        #         self.cache[idx] = item

        info = [len(pts) for pts in self.video_clips.video_pts]
        logger.info("frame_distribution: min %s, max %s", min(info), max(info))
        logger.info("dataset length: %s", len(self))
        logger.info("video count: %s", len(self.video_clips.video_paths))
        ds_labels = [self.labels[self.video_clips.get_clip_location(idx)[0]] for idx in range(len(self))]
        logger.info('class distribution: %s', {a: ds_labels.count(a) for a in set(self.labels)})
        logger.info('frame mode: %s', self.video_clips.frame_mode)

    @classmethod
    def from_wandb_artifact(
            cls,
            artifact_name: str, split_by='random',
            mix_clip=0, extract_groom=False, no_valid=False,
            exclude_5min=False, exclude_2_mouse=False, exclude_fpvid=True, exclude_2_mouse_valid=False,
            **kwargs):
        artifact = wandb.run.use_artifact(artifact_name, type='dataset')
        data_dir = artifact.download(root='./output')

        with open(os.path.join(data_dir, 'annotation_list.json'), 'r') as f:
            dataset_dict = json.load(f)

        metadata_path = os.path.join(data_dir, 'metadata.pth')
        metadata = torch.load(metadata_path)

        class2idx = {cn: int(idx) for idx, cn in read_tuple(os.path.join(data_dir, 'mapping.txt'), n=1)}
        if extract_groom:
            metadata, labels = extract_metadata_from_annotations(metadata, dataset_dict)
            labels = [class2idx[lab] for lab in labels]
        else:
            labels = [class2idx[os.path.dirname(p).split('/')[-1]] for p in metadata['video_paths']]

        split_file = os.path.join(data_dir, 'split_metadata.json')
        df = pd.read_json(split_file, orient='records')
        assert split_by in ['random', 'new_random'] + \
            [f'cage_{n}' for n in range(df['cage'].nunique())] + \
            [f'src_{n}' for n in range(df['src'].nunique())]

        exclude_list = []
        if exclude_2_mouse_valid:
            df = df[~df['exclude_2_mouse']]
        if exclude_5min:
            exclude_list += df[df['5min']]['video_path'].to_list()
        if exclude_2_mouse:
            exclude_list += df[df['exclude_2_mouse']]['video_path'].to_list()
        if exclude_fpvid:
            exclude_list += df[df['fp_vid']]['video_path'].to_list()

        if split_by.startswith('cage') or split_by.startswith('src'):
            key, split = split_by.split('_')
            split = int(split)
            testset = df[df[key] == split]['video_path'].to_list()
            assert testset, 'testing no dataset'

            if no_valid:
                validset = testset
            else:
                validset = df[df[key] == (split + 1) % (df[key].max() + 1)]['video_path'].to_list()
                now_split = split
                while not validset:
                    now_split += 1
                    validset = df[df[key] == (now_split + 1) % (df[key].max() + 1)]['video_path'].to_list()

            trainset = df[~df['video_path'].isin(validset + testset + exclude_list)]['video_path'].to_list()
            assert trainset, ('empty training dataset', exclude_list, validset, testset)
            if mix_clip > 0:
                def get_id(path):
                    return int(path.split('.')[-2].split('_')[-1])
                trainset += [p for p in validset if get_id(p) < mix_clip]
                trainset += [p for p in testset if get_id(p) < mix_clip]
            split_data = {'train': trainset, 'valid': validset, 'test': testset}
        else:
            split_data = {
                key: df[df['random_split'] == key]['video_path'].to_list()
                for key in ['train', 'valid', 'test']
            }

        return cls(metadata=metadata, labels=labels, dataset_root=None, split_data=split_data, **kwargs)

    @classmethod
    def from_ds_folder(cls, dataset_root, metadata_path=None, extract_groom=False, **kwargs):
        # make dataset
        classes = list(sorted(list_dir(dataset_root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        items = make_dataset(dataset_root, class_to_idx, extensions=(".mp4",))
        logger.info('class2idx: %s', class_to_idx)
        samples = [s[0] for s in items]
        labels = [s[1] for s in items]

        metadata = None
        if metadata_path is not None and os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
        metadata = VideoClips(
            samples,  # for computing timestamps
            _precomputed_metadata=metadata,
            num_workers=8,
        ).metadata
        if metadata_path is not None and metadata is None:
            torch.save(metadata, metadata_path)

        if extract_groom:  # special case: groom under dataset_root/groom
            groom_folder = f'{dataset_root}/groom'
            annots = read_annotations(groom_folder)
            extracted_metadata, extracted_labels = extract_metadata_from_annotations(metadata, annots)
            # extracted_paths = set(item['video_paths'] for item in extracted_metadata)
            extracted_labels = [class_to_idx[a] for a in extracted_labels]
            logger.info('Extracted groom video count: %s', len(extracted_labels))
            for vid_path, vid_pts, vid_fps, lab in zip(metadata['video_paths'],
                                                       metadata['video_pts'],
                                                       metadata['video_fps'],
                                                       labels):
                if 'not_groom' not in vid_path:
                    continue
                extracted_metadata['video_paths'].append(vid_path)
                extracted_metadata['video_pts'].append(vid_pts)
                extracted_metadata['video_fps'].append(vid_fps)
                extracted_labels.append(lab)
            metadata, labels = extracted_metadata, extracted_labels
        return cls(metadata=metadata, labels=labels, dataset_root=dataset_root, **kwargs)

    @classmethod
    def from_annotation_list(cls, dataset_root, **kwargs):  # fine-tune I3D
        if dataset_root == './data/breakfast':
            dataset_dict, meta_dict = get_breakfast_dataset()
        elif dataset_root == './data/mpii':
            dataset_dict, meta_dict = get_mpii_cooking_dataset()
        else:
            raise NotImplementedError('Name', dataset_root)
        metadata_path = os.path.join(meta_dict['video_root'], '..', 'metadata.pth')

        metadata = None
        if metadata_path is not None and os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
        else:
            logger.info("Calculating metadata to %s", metadata_path)
        metadata = VideoClips(
            [item['video_path'] for item in dataset_dict],  # for computing timestamps
            _precomputed_metadata=metadata,
            num_workers=8,
        ).metadata
        if metadata_path is not None:
            torch.save(metadata, metadata_path)

        class2idx = {cn: int(idx) for idx, cn in read_tuple(meta_dict['mapping_file'], n=1)}
        new_meta, labels = extract_metadata_from_annotations(metadata, dataset_dict, label_set=list(class2idx.keys()))
        labels = [class2idx[lab] for lab in labels]

        n_splits = meta_dict['n_splits']
        for split in range(n_splits):
            split_file = os.path.join(meta_dict['video_root'], '..', f'cv_{split}_split.json')
            if os.path.exists(split_file):
                continue
            valid_split = (split + 1) % n_splits
            test_paths = [ds_dict['video_path'] for ds_dict in dataset_dict if ds_dict['split'] == split]
            valid_paths = [ds_dict['video_path'] for ds_dict in dataset_dict if ds_dict['split'] == valid_split]
            train_paths = [
                ds_dict['video_path'] for ds_dict in dataset_dict if ds_dict['split'] not in [split, valid_split]]
            with open(split_file, 'w') as f:
                json.dump({
                    'train': train_paths,
                    'valid': valid_paths,
                    'test': test_paths,
                }, f)

        return cls(metadata=new_meta, labels=labels, dataset_root=meta_dict['video_root'], **kwargs)

    def get_new_split(self, split_by):
        if self.split_data is None:
            if split_by == 'new_random':
                split_ratio = [int(len(self.video_clips.video_paths) * r) for r in (0.7, 0.2)]
                split_ratio += [len(self.video_clips.video_paths) - sum(split_ratio)]
                splits = [list(subset) for subset in random_split(self.video_clips.video_paths, split_ratio)]
                self.split_data = dict(zip(['train', 'valid', 'test'], splits))
            logger.info("New Split:\n%s", pprint.pformat(self.split_data))
        return self.split_data

    def get_split(self, split, split_by, transform_size, overwrite_dict: Dict[str, Any]) -> "MouseClipDataset":
        if split_by in ["new_random", "random"] or split_by.startswith('cage_') or split_by.startswith('src_'):
            split_data = self.get_new_split(split_by)
        else:
            split_fname = f'{split_by}_split.json'
            assert self.dataset_root is not None
            split_path = os.path.join(self.dataset_root, '..', split_fname)
            if not os.path.exists(split_path):
                raise FileNotFoundError(f'{split_fname} not found at {self.dataset_root}/..')
            with open(split_path, 'r') as f:
                split_data = json.load(f)
        indices = sorted([
            idx for idx, p in enumerate(self.video_clips.video_paths)
            if p in split_data[split]
        ])
        metadata = self.video_clips.subset(indices).metadata
        labels = [self.labels[idx] for idx in indices]

        if self.output_type == "random_frame":
            transform = default_transformation_2D(split, transform_size)
        elif self.output_type == "video":
            transform = default_transformation_3D(split, transform_size)

        ds = {
            "dataset_root": self.dataset_root,
            "output_type": self.output_type,
            "frames_per_clip": self.frames_per_clip,
            "step_between_clips": self.step_between_clips,
            "frame_rate": self.frame_rate,
            "num_sample_per_clip": self.num_sample_per_clip,
            "in_memory": split in ['valid'],
        }
        ds.update(overwrite_dict)

        return type(self)(metadata=metadata, labels=labels, transform=transform, **ds)  # type: ignore

    def __getitem__(self, idx):
        if self.in_memory and idx in self.cache:
            return self.cache[idx]
        video, _audio, _info, video_idx = self.video_clips.get_clip(idx)
        label = self.labels[video_idx]
        if self.output_type == 'random_frame':
            # video: Tensor[T, H, W, C]
            random_idx = random.randint(0, video.size(0) - 1)
            # img: [C, H, W]
            ret = video[random_idx].numpy()
        elif self.output_type == 'video':
            ret = video
        else:
            raise NotImplementedError(f"output_type {self.output_type}")

        if self.transform is not None:
            ret = self.transform(ret)
        if self.in_memory:
            self.cache[idx] = (ret, label, video_idx)
        return ret, label, video_idx

    def __len__(self):
        return self.video_clips.num_clips()

    def get_sampler(self, split, sampler_config=None, num_samples: Optional[int] = None):
        if split == "train":
            sampler = self.build_train_sampler(sampler_config, num_samples=num_samples)
        else:
            sampler = self.build_test_sampler(sampler_config)
        if num_samples is not None and sampler is not None:
            sampler = MaxLengthClipSampler(sampler, num_samples)
        return sampler

    def build_train_sampler(self, config, num_samples):
        if num_samples is None:
            return RandomClipSampler(self.video_clips, self.num_sample_per_clip)
        cfg = {
            "log_weight": False,
            "balance_label": True,
            "balance_video": False,
            "balance_clip": False,
            "balance_src": False,
        }
        if config:
            cfg.update(config)
        vc_labels = []
        for idx in range(self.video_clips.num_clips()):
            vidx, _ = self.video_clips.get_clip_location(idx)
            ans = []
            if cfg['balance_label']:
                ans.append(self.labels[vidx])
            if cfg['balance_video']:
                path = self.video_clips.video_paths[vidx]
                ans.append(path)
            if cfg['balance_clip']:
                path = self.video_clips.video_paths[vidx]
                src = '_'.join(path.split('/')[-1].split('.')[0].split('_')[:-1])
                ans.append(src)
            if cfg['balance_src']:
                path = self.video_clips.video_paths[vidx]
                src = '_'.join(path.split('/')[-1].split('.')[0].split('_')[:-2])
                ans.append(src)
            assert ans
            vc_labels.append(tuple(ans))

        if num_samples is None:
            num_samples = len(self.video_clips.video_paths)
        return BalancedSampler(vc_labels, num_samples, cfg['log_weight'])

    def build_test_sampler(self, config):
        cfg = {"downsample": False, "balance_label": False}
        if config:
            cfg.update(config)
        if cfg["downsample"]:
            return DownsampleClipSampler(self.video_clips, self.labels)
        if cfg['balance_label']:
            return BalancedClipSampler(self.video_clips, self.labels)
        return None
