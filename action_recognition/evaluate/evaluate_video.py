import os
import json
import logging
from typing import Dict, Tuple, List, Union, Any

# import seaborn as sns
import torch
import wandb
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision.models.video.resnet import VideoResNet
from tqdm import tqdm
from sklearn.metrics import classification_report

from action_recognition.datasets.inference_dataset import Video_Instance, Video_2D_Inference, Video_3D_Inference
from action_recognition.datasets.video_annotation_dataset import read_annotations
from action_recognition.models.i3d import I3D
from action_recognition.utils.utils import safe_dir
# from action_recognition.experiment.config import

logger = logging.getLogger(__name__)


def inference_video(
        model: torch.nn.Module, gpu_id: int,
        data_queue: mp.Queue, result_queue: mp.Queue,
        # dataset: Union[Video_2D_Inference, Video_3D_Inference],
        # batch_size: int, num_worker: int,
):
    model = model.eval().cuda(device=gpu_id)
    with torch.no_grad():
        # for data, fn, idx, done in DataLoader(dataset, batch_size=batch_size, num_workers=num_worker):
        while True:
            data, fn, idx, done = data_queue.get()
            out = model(data.cuda(device=gpu_id)).detach().cpu()
            result_queue.put((out, fn, idx.clone(), done.clone()))
            del data, idx, done


def read_data(
        dataset: Union[Video_2D_Inference, Video_3D_Inference],
        batch_size: int, num_worker: int, data_queue: mp.Queue):
    mp.set_sharing_strategy('file_system')
    for item in DataLoader(dataset, batch_size=batch_size, num_workers=num_worker):
        data_queue.put(item)


def gather_result(result_queue: mp.Queue, total_video_count: int, model_artifact_name: str, total_len=None):
    fn_results: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
    fns: List[Tuple] = []
    pbar = tqdm(total=total_len, dynamic_ncols=True)
    while True:
        if len(fns) == total_video_count:
            break
        out_, fn_, idx_, done_ = result_queue.get()
        for out, fn, idx, done in zip(out_.clone(), fn_, idx_.clone(), done_.clone()):
            fn_results.setdefault(fn, list())
            fn_results[fn].append((idx, out.clone()))
            pbar.update(1)
            if done:
                output = [t for _, t in sorted(fn_results.pop(fn))]
                output_path = fn2outfn(fn, model_artifact_name)
                safe_dir(output_path, with_filename=True)
                torch.save(output, output_path)
                fns.append((fn, output_path))
                pbar.set_description(f'Done: {output_path}')
        del out_, fn_, idx_, done_
    pbar.close()
    return fns


def fn2outfn(fn, model_artifact_name):
    return os.path.join('output', model_artifact_name, f'{os.path.basename(fn)}.pkl')


def run_inference(video_paths: List[str], model_artifact_name: str, args):
    is3D = start_inferencing(video_paths, model_artifact_name)

    fns = [(p, fn2outfn(p, model_artifact_name)) for p in video_paths]

    logger.info(fns)
    for fn, out_path in fns:
        artifact = wandb.Artifact(os.path.basename(fn), type='result', metadata={'model': model_artifact_name})
        artifact.add_file(out_path)
        wandb.run.use_artifact(artifact)
        plot_output(out_path, fn, args.prob_thresh, window_size=16 if is3D else 1)


def start_inferencing(video_paths: List[str], model_artifact_name: str):
    artifact = wandb.run.use_artifact(model_artifact_name + ':latest', type='model')
    inf_vid = [p for p in video_paths if not os.path.exists(fn2outfn(p, model_artifact_name))]

    model = torch.load(artifact.get_path('best_valid_Accuracy_model.pth').download(), map_location='cpu')
    is3D = isinstance(model, (I3D, VideoResNet))
    logger.info("Inferencing model: %s", model.__class__.__name__)

    batch_size = 16 if is3D else 64
    vid_instances = [Video_Instance(p) for p in inf_vid]
    vid_dataset = Video_3D_Inference(vid_instances, 16) if is3D else Video_2D_Inference(vid_instances)

    ctx = mp.get_context('spawn')
    data_queue = ctx.Queue(maxsize=4 * torch.cuda.device_count())
    result_queue = ctx.Queue()
    model_worker = [
        ctx.Process(target=inference_video, args=(
            model, gpu_id, data_queue, result_queue,
            # vid_dataset, batch_size, 0
        ))
        for gpu_id in range(torch.cuda.device_count())
    ]
    data_worker = ctx.Process(target=read_data, args=(vid_dataset, batch_size, 0, data_queue))

    data_worker.start()
    for p in model_worker:
        p.start()

    gather_result(
        result_queue, len(inf_vid), model_artifact_name,
        total_len=sum(v.info['frame_count'] for v in vid_instances))

    data_worker.join()
    for p in model_worker:
        p.terminate()

    return is3D


def plot_output(out_path, vid_fn, prob_thresh=0.5, window_size=16):
    data = torch.load(out_path)
    data = torch.stack(data)[:, :2]  # [L, 2]
    y_prob = torch.nn.functional.softmax(data, dim=1)
    # y_pred = (torch.argmax(data, dim=1, keepdim=True) == 0).long()
    y_pred = (y_prob >= prob_thresh).long()[:, :1]  # [L, 2]

    # anno = load_annot(vid_fn)
    anno = load_fine_annot(vid_fn)
    y_gt = torch.zeros(data.shape[0], 1).long()
    for s, e in anno:
        y_gt[s - window_size + 1:e - window_size + 1] = 1

    labels = ["no groom", "groom"]

    wandb.sklearn.plot_confusion_matrix(y_gt, y_pred, labels=labels)
    report = classification_report(y_gt, y_pred, target_names=labels, output_dict=True)
    report = expand_nested_str_dict(report)

    report.update({
        "ROC": wandb.plots.ROC(y_gt, y_prob, labels=labels),
        "PR_curve": wandb.plots.precision_recall(y_gt, y_prob, labels=labels),
    })
    wandb.log(report)


def load_annot(fn):
    with open('./data/mouse_video/annotation.json') as f:
        annot = json.load(f)
    for d in annot:
        if d['cropped_path'] == fn:
            return [(s * d['fps'], e * d['fps']) for s, e in d['annotation']]
    return None


def load_fine_annot(fn):
    vid_root = 'data/mouse_video/crop_preprocessed/groom'
    annotation_list = read_annotations(vid_root)

    coarse_annot = load_annot(fn)
    supvid_fns = [
        f'{vid_root}/' + os.path.basename(fn).split('.')[0] + f'_{i:05d}.mp4'
        for i in range(len(coarse_annot))]
    ret = []
    for (ss, _es), supvid_fn in zip(coarse_annot, supvid_fns):
        fine_anno = [item for item in annotation_list if item['video_path'] == supvid_fn]
        if fine_anno:
            fine_anno = fine_anno[0]['annotations']
            for act, sf, ef in fine_anno:
                if act != 'groom':
                    continue
                ret.append((ss + sf, ss + ef + 1))
    return ret


def expand_nested_str_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    ret = {}
    for k, v in input_dict.items():
        if isinstance(v, dict):
            ret_dict = expand_nested_str_dict(v)
            for nk, nv in ret_dict.items():
                ret[f'{k}.{nk}'] = nv
        else:
            ret[k] = v
    return ret
