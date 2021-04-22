import argparse
import logging

import wandb

from action_recognition.utils.mq_tqdm import mp_tqdm, cmd_worker

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_group", default='', type=str)
    parser.add_argument("--group", default='', type=str)
    parser.add_argument("--process_cnt", default=1, type=int)
    parser.add_argument("--prob_thresh", default=0.5, type=float)
    parser.add_argument("--random_split", default=False, action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    group_runs = api.runs('donny/video_classification', {'group': args.train_group}, order='+config.split_by')
    ret = {}
    for run in group_runs:
        if run.state in ['finished']:
            ret[run.config['split_by']] = run.id

    cmd_format = 'pipenv run python run_inference.py --group {gp} --split {sp} --run_id {rid} --prob_thresh {pt}'
    if args.random_split:
        cmds = [
            cmd_format.format(
                gp=f'inference_{args.train_group}' if not args.group else args.group,
                pt=args.prob_thresh, sp=-1, rid=ret['random'])
        ]
    else:
        cmds = [
            cmd_format.format(
                gp=f'inference_{args.train_group}' if not args.group else args.group,
                pt=args.prob_thresh, sp=split.split('_')[-1], rid=run_id)
            for split, run_id in ret.items() if split != 'random'
        ]

    logging.info('Running cmds:\n%s', '\n'.join(cmds))

    mp_tqdm(cmd_worker, [{'cmd': c} for c in cmds], process_cnt=args.process_cnt)
