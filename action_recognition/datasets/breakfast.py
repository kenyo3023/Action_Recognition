import os
import logging
# from typing import Callable, Optional, List

from action_recognition.datasets.video_annotation_dataset import read_tuple


logger = logging.getLogger(__name__)


split_name = [
    ['P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15'],
    ['P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28'],
    ['P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40', 'P41'],
    ['P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49', 'P50', 'P51', 'P52', 'P53', 'P54']
]


def get_split_by_path(p):
    # gt_root/class/PXX_src_PXX_class.txt
    pid = os.path.basename(p).split('_')[0]
    for i, splits in enumerate(split_name):
        if pid in splits:
            return i
    raise ValueError(f"Split not found: {p}")


def get_breakfast_dataset(**kwargs):
    # [act, start (0-base), end (0-base, not included)]

    metadata = {
        # "video_root": './data/breakfast/Breakfast_Final/vid',
        "video_root": './data/breakfast/Breakfast_Final/mp4',
        "gt_root": './data/breakfast/Breakfast_Final/lab_raw',
        "mapping_file": './data/breakfast/mapping_bf.txt',
        "n_splits": len(split_name),
    }
    metadata.update({k: kwargs.get(k, metadata[k]) for k in metadata})
    dataset_dict = []
    for dirpath, _, fnames in os.walk(metadata['video_root']):
        if 'stereo02' in dirpath:
            continue
        for fname in fnames:
            # if not fname.endswith('.avi'):
            if not fname.endswith('.mp4'):
                continue
            vid_path = os.path.join(dirpath, fname)

            pid, lab = os.path.splitext(fname)[0].split('_')
            gt_path = os.path.join(metadata['gt_root'], pid, f'{pid}_{lab}.coarse')
            if not os.path.exists(gt_path):
                logger.debug("video file not exist for %s: %s", vid_path, gt_path)
                continue

            gt = []
            for ts, act in read_tuple(gt_path):
                st, en = ts.split('-')
                gt.append((act, int(st) - 1, int(en)))

            dataset_dict.append({
                "video_path": vid_path,
                "annotations": gt,
                "video_len": int(en),
                "split": get_split_by_path(gt_path)
            })
    return dataset_dict, metadata
