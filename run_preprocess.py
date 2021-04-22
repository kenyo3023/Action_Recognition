import os
import logging
import json
import random
from glob import glob

import pandas as pd
import wandb
import torch
from torchvision.datasets.video_utils import VideoClips

from action_recognition.experiment.config import WandbConfig
from action_recognition.datasets.preprocess import get_video_len, cut_by_annotation, cmd_worker, vid_2_img
from action_recognition.datasets.video_annotation_dataset import read_annotations
from action_recognition.utils import mp_tqdm


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # prepare_mouse_clipped_dataset()

    cmd_list = cut_by_annotation()
    if cmd_list:
        logging.info("Cutting by annotation...")
        mp_tqdm(cmd_worker, [{"cmd": c} for c in cmd_list], process_cnt=4)
    dataset_name = 'mouse_cropped'
    metadata = {
        "dataset_root": './data/mouse_video/crop_preprocessed',
        "video_clip_metadata_path": './data/mouse_video/crop_preprocessed/metadata.pth',
    }

    mp4_list = glob(f'{metadata["dataset_root"]}/**/*.mp4')
    if not os.path.exists(metadata['video_clip_metadata_path']):
        logging.info("Metadata not found, calculating VideoClip metadata...")
        vc = VideoClips(video_paths=mp4_list, num_workers=1)
        torch.save(vc.metadata, metadata['video_clip_metadata_path'])

    logging.info('Running video 2 image...')
    mp_tqdm(
        vid_2_img,
        [{'video_path': v} for v in mp4_list],
        shared={'shortside_len': 256},
        process_cnt=16)

    annot_list = read_annotations(metadata['dataset_root'] + '/groom')
    not_groom_root = os.path.join(metadata['dataset_root'], 'not_groom')
    for p in os.listdir(not_groom_root):
        if not p.endswith('.mp4'):
            continue
        vp = os.path.join(not_groom_root, p)
        vid_len = get_video_len(vp)
        annot_list += [{
            'video_path': vp,
            'annotations': [('not_groom', 0, None)],
            'video_len': vid_len,
        }]

    fp_root = os.path.join(metadata['dataset_root'], 'fp_vid')
    for p in os.listdir(fp_root):
        if not p.endswith('.mp4'):
            continue
        vp = os.path.join(fp_root, p)
        vid_len = get_video_len(vp)
        annot_list += [{
            'video_path': vp,
            'annotations': [('not_groom', 0, None)],
            'video_len': vid_len,
        }]
    # annot_list += read_annotations(metadata['dataset_root'] + '/fp_vid')

    cage_list = [
        {
            'video_path': item['video_path'],
            'cage': '_'.join(os.path.basename(item['video_path']).split('_')[:-1]),
            'src': '_'.join(os.path.basename(item['video_path']).split('_')[:-2]),
        }
        for item in annot_list
    ]
    for item in cage_list:
        item['exclude_2_mouse'] = (item['cage'] in ['V_20190917_124903_OC0_c2', 'V_20190917_124903_OC0_c3'])
        item['always_train'] = 'V_20' not in item['src']
        item['fp_vid'] = 'fp_vid' in item['src']
        item['5min'] = '/groom/V_20' not in item['video_path']

    cage_df = pd.DataFrame(cage_list)
    split_mask = ~cage_df.always_train
    cage_df.loc[split_mask, 'cage'] = pd.Categorical(cage_df[split_mask].cage).codes
    cage_df.loc[~split_mask, 'cage'] = -1
    cage_df.loc[split_mask, 'src'] = pd.Categorical(cage_df[split_mask].src).codes
    cage_df.loc[~split_mask, 'src'] = -1

    all_paths = [item['video_path'] for item in annot_list]
    assert all(os.path.isfile(p) for p in all_paths)

    split_ratio = [int(len(all_paths) * r) for r in (0.7, 0.15)]
    new_split = ['train'] * split_ratio[0] + ['valid'] * split_ratio[1] + ['test'] * (len(all_paths) - sum(split_ratio))
    random.shuffle(new_split)
    cage_df['random_split'] = new_split

    logging.info("Cage Split: %s\n", cage_df)

    # log artifact
    config = WandbConfig()
    wandb.init(
        job_type='preprocess', entity=config.wandb_repo,
        project=config.wandb_project, name=dataset_name or config.cur_time,
        group=config.wandb_group, dir=config.wandb_dir,
    )
    logging.info('Logging Dataset artifact')
    artifact = wandb.Artifact(dataset_name, type='dataset')
    with artifact.new_file('annotation_list.json') as f:
        json.dump(annot_list, f)
    with artifact.new_file('split_metadata.json') as f:
        cage_df.to_json(f, orient='records')
    with artifact.new_file('mapping.txt') as f:
        f.write('0 groom\n1 not_groom')
    artifact.add_file(metadata['video_clip_metadata_path'])
    wandb.run.log_artifact(artifact)
