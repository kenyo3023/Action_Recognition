import argparse
import logging

import wandb

from action_recognition.experiment.config import WandbConfig
from action_recognition.evaluate.evaluate_video import run_inference

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    video_paths = [
        'data/mouse_video/20190814/V_20190814_134337_OC0_c0.mp4',  # 0
        'data/mouse_video/20190906/V_20190906_123023_OC0_c0.mp4',  # 1
        'data/mouse_video/20190906/V_20190906_123023_OC0_c1.mp4',  # 2
        'data/mouse_video/20190917/V_20190917_124903_OC0_c0.mp4',  # 3
        'data/mouse_video/20190917/V_20190917_124903_OC0_c2.mp4',  # 4
        'data/mouse_video/20190917/V_20190917_124903_OC0_c3.mp4',  # 5
        'data/mouse_video/20190919/V_20190919_134212_OC0_c0.mp4',  # 6
        'data/mouse_video/20190919/V_20190919_134212_OC0_c1.mp4',  # 7
        'data/mouse_video/20190919/V_20190919_134212_OC0_c2.mp4',  # 8
        'data/mouse_video/20190919/V_20190919_134212_OC0_c3.mp4',  # 9
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default='', type=str)
    parser.add_argument("--split", default=-1, type=int)
    parser.add_argument("--run_id", default='', type=str)
    parser.add_argument("--prob_thresh", default=0.5, type=float)
    args = parser.parse_args()
    config = WandbConfig(wandb_group='inference')

    if args.group:
        config.wandb_group = args.group
    if args.split >= 0:
        video_paths = [video_paths[args.split]]  # , video_paths[(split + 2) % 10]]
    model_artifact_name = f"run_{args.run_id}_model"

    wandb.init(
        job_type='inference', entity=config.wandb_repo,
        project=config.wandb_project, name=config.cur_time,
        group=config.wandb_group, dir=config.wandb_dir,
        config=args,
    )

    run_inference(video_paths, model_artifact_name, args)
