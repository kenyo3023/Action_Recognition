import os
from os.path import join
import glob
import logging
import random
import json
import subprocess
from collections import defaultdict

import cv2
from torch.utils.data.dataset import random_split
from torchvision.datasets import DatasetFolder
from torchvision.datasets.utils import extract_archive

from action_recognition.utils import mp_tqdm_worker
from action_recognition.utils.utils import safe_dir


logger = logging.getLogger(__name__)


@mp_tqdm_worker
def cmd_worker(cmd, **_kwargs):
    subprocess.run(cmd.split(), check=True)


def prepare_mouse_clipped_dataset(zip_path="./data/clipped_database.zip"):
    """Prepare extracted dataset into DatasetFolder format"""

    # assume dataset is downloaded under ./data
    root = './data/clipped_database'
    preprocessed_root = join(root, 'preprocessed')
    if not os.path.exists(root):
        logger.info("Extracting file")
        extract_archive(zip_path, "./data")

    label_name = {
        'drink': ['drink', 'd'],
        'eat': ['eat', 'e'],
        'groom': ['groomback', 'groom', 'gb', 'g'],
        'hang': ['hang', 'ha'],
        'head': ['head', 'he'],
        'rear': ['rear', 'r'],
        'rest': ['rest', 'rs'],
        'walk': ['walk', 'w'],
    }
    label_map = {v: k for k in label_name for v in label_name[k]}
    for cls_name in label_name:
        dirname = join(preprocessed_root, cls_name)
        if os.path.exists(dirname):
            logging.info("Class dir already exists, aborting preprocess")
            return
        os.mkdir(dirname)
        logger.info('mkdir %s', dirname)
    cmds = []
    for rt, _, files in list(os.walk(root)):
        for fname in files:
            name, ext = os.path.splitext(fname)
            if ext != '.mpg':
                logger.info('Found File %s, ignored', join(rt, fname))
                continue
            _, label, _ = name.split('_')
            label = label_map[label]
            src, dst = join(rt, fname), join(preprocessed_root, label, f'{name}.mp4')
            # logger.info('Transcoding file from %s to %s', src, dst)
            # shutil.move(src, dst)
            cmd = f'ffmpeg -loglevel panic -i {src} {dst}'
            cmds.append({"cmd": cmd})

    # mp_tqdm(cmd_worker, cmds, process_cnt=16)

    # for rt, dirs, _ in os.walk(root):
    #     for folder in dirs:
    #         dirname = join(rt, folder)
    #         print(dirname)
    #         if not os.listdir(dirname):
    #             os.rmdir(dirname)
    #             logger.info('Removing %s', dirname)


def gen_train_valid_test_split(root, by="random", ratio=(0.5, 0.2, 0.3)):
    assert by in ['random']
    seed = random.random()
    meta_data = {
        "ratio": ratio,
        "seed": seed,
    }
    random.seed(seed)
    all_data = DatasetFolder(root, loader=lambda x: x, extensions=(".mp4",))
    # video_clips = {}
    if by == "random":
        ratio = list(int(r * len(all_data)) for r in ratio)
        ratio[0] += len(all_data) - sum(ratio)
        split = random_split(all_data, ratio)
        for spl, ds in enumerate(split):
            meta_data[f'split_{spl}'] = list(p for p, _ in ds)
        if len(split) == 2:
            meta_data['train'] = meta_data.pop('split_0')
            meta_data['test'] = meta_data.pop('split_1')
        elif len(split) == 3:
            meta_data['train'] = meta_data.pop('split_0')
            meta_data['valid'] = meta_data.pop('split_1')
            meta_data['test'] = meta_data.pop('split_2')

    with open(join(root, f"{by}_split.json"), 'w') as f:
        json.dump(meta_data, f)


def get_video_len(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=nb_frames", "-of",
            "default=noprint_wrappers=1:nokey=1", video_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True)
    return int(result.stdout)


def cut_by_annotation(root='./data/mouse_video/', annotation_fn='annotation.json'):
    # cut video into clips
    with open(join(root, annotation_fn), 'r') as f:
        annot = json.load(f)

    assert all(k in d for k in ['video_path', 'annotation', 'unit'] for d in annot)
    preprocessed_root = join(root, 'crop_preprocessed')
    safe_dir(preprocessed_root)
    labels = ["groom", "not_groom"]

    for lab in labels:
        class_path = join(preprocessed_root, lab)
        safe_dir(class_path)

    video_annot = defaultdict(list)
    for data in annot:
        assert data['unit'] == 'sec' and 'fps' in data, data
        p = data['cropped_path']
        ann = data['annotation']
        video_annot[p].extend(ann)

    cmd_list = []
    for p in video_annot:
        vid_len = int(get_video_len(p))
        others = [True] * vid_len
        for idx, (start, end) in enumerate(video_annot[p]):
            local_fn = f'{os.path.basename(p).split(".")[0]}_{idx:05d}.mp4'
            local_clipfile_name_path = join(preprocessed_root, "groom", local_fn)
            dur = end - start + 1
            if not os.path.exists(local_clipfile_name_path):
                assert dur > 0, (start, end, local_clipfile_name_path)
                cmd = (f"ffmpeg -loglevel warning -i {p} -y "
                       f"-ss 00:{start//60}:{start%60} -t 00:{dur//60}:{dur%60} {local_clipfile_name_path}")
                cmd_list.append(cmd)
                for i in range(start, end + 1):
                    others[i] = False
        # not groom
        now_st = -1
        other_idx = 0
        for i in range(1, len(others)):
            if others[i] and not others[i - 1] or i == 1:
                now_st = i
            if not others[i] and others[i - 1] or i == (len(others) - 1):
                local_fn = f'{os.path.basename(p).split(".")[0]}_{other_idx:05d}.mp4'
                other_idx += 1
                local_clipfile_name_path = join(preprocessed_root, "not_groom", local_fn)
                dur = i - now_st
                if not os.path.exists(local_clipfile_name_path):
                    cmd = (f"ffmpeg -loglevel warning -i {p} -y "
                           f"-ss 00:{now_st//60}:{now_st%60} -t 00:{dur//60}:{dur%60} {local_clipfile_name_path}")
                    cmd_list.append(cmd)

    return cmd_list


@mp_tqdm_worker
def vid_2_img(video_path, target_directory=None, shortside_len=256, prefix_str='img', **_kwargs):
    if target_directory is None:
        target_directory = os.path.join(os.path.dirname(video_path), 'jpg', os.path.basename(video_path).split('.')[0])
    safe_dir(target_directory)

    # Check if image is already processed
    for f in glob.glob(f"{target_directory}/{prefix_str}_*.jpg"):
        if os.path.getsize(f) <= 0:
            os.remove(f)
    img_cnt = len(glob.glob(f"{target_directory}/{prefix_str}_*.jpg"))
    vid_len = get_video_len(video_path)
    if img_cnt == vid_len:
        return

    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width, height = cap.get(3), cap.get(4)
            factor = max(shortside_len / width, shortside_len / height)
            width, height = int(width * factor), int(height * factor)
            width -= width % 2
            height -= height % 2
        else:
            print(f"[Error] {video_path} Bad Video")
            return

        total_count, valid_count = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            total_count += 1
            if ret:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f"{target_directory}/{prefix_str}_{valid_count:06d}.jpg", frame)
                valid_count += 1
            elif total_count != 1:
                break
        processed_cnt = len(glob.glob(f"{target_directory}/{prefix_str}_*.jpg"))
        if processed_cnt == 0:
            print(f"{target_directory} is empty!")
        elif processed_cnt != vid_len:
            print(f"Not at same length! processed frame cnt: {processed_cnt}, video n_frame: {vid_len}")

    except Exception:
        print(f"[Error] {target_directory} TRY")
