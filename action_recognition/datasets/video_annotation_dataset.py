import os
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


def extract_metadata_from_annotations(metadata, annotation_list, label_set=None):
    # make new metadata by annotation
    new_meta = {
        "video_paths": [],
        "video_pts": [],
        "video_fps": [],
    }
    labels = []
    for item in annotation_list:
        paths = item['video_path']
        idx = metadata['video_paths'].index(paths)
        pts, fps = metadata['video_pts'][idx], metadata['video_fps'][idx]
        if len(pts) != item['video_len']:
            logger.debug(
                'Annotation length %s does not match with video legth %s on video %s',
                item['video_len'], len(pts), paths,
            )
        for act, start, end in item['annotations']:
            start = start or 0
            end = end or len(pts)
            if not start < end <= len(pts):
                logger.debug("annotation (%s, %s, %s) not match with frame_cnt (%s)", act, start, end, len(pts))
            if label_set and act not in label_set:
                continue
            new_meta['video_paths'].append(paths)
            new_meta['video_pts'].append(pts[start:end])
            new_meta['video_fps'].append(fps)
            labels.append(act)
    return new_meta, labels


def read_annotations(video_folder) -> List[Dict]:
    annotation_list = []
    for video_path in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_path)
        if not video_path.endswith('mp4'):
            continue
        item = {'video_path': video_path}
        annotation_file = os.path.splitext(video_path)[0] + '.xml'
        if not os.path.isfile(annotation_file):
            continue
        item['annotations'], item['video_len'] = read_from_vatic_xml(annotation_file)
        annotation_list.append(item)
    return annotation_list


def read_from_vatic_xml(xml_path) -> Tuple[List[Tuple[str, int, int]], int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ed_frame = root.findtext('object/endFrame')
    assert ed_frame is not None, xml_path
    annot_vid_len = int(ed_frame) if ed_frame is not None else -1
    start, end = None, None
    ret: List[Tuple[str, int, int]] = []
    for el in root.iterfind('object/polygon'):
        tt = el.findtext('t')
        frame = int(tt) if tt is not None else -1
        if el.findtext('pt/l') == '1':
            if not ret and frame != 0:
                ret.append(('not_groom', 0, frame))
            if end is not None:
                assert start is not None
                ret.append(('groom', start, end))
                ret.append(('not_groom', end, frame))
                end = None
            start = frame
        else:
            end = frame + 1
    if end is not None:
        assert start is not None
        ret.append(('groom', start, end))
        ret.append(('not_groom', end, annot_vid_len))
        end = None
    return ret, annot_vid_len


def read_tuple(path, n=-1):
    with open(path, 'r') as f:
        return list(line.strip().split(maxsplit=n) for line in f.readlines())
