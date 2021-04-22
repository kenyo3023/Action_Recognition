import os
import logging

import pandas as pd

# from ASF.datasets.data_utils import read_tuple


logger = logging.getLogger(__name__)

testing_subjects = [8, 10, 16, 17, 18, 19, 20]


def get_mpii_cooking_dataset(**kwargs):
    # [act, start (0-base), end (0-base, not included)]
    metadata = {
        # "video_root": './data/mpii/videos/videos',
        "video_root": './data/mpii/videos/mp4',
        "gt_root": './data/mpii/detectionGroundtruth-1-0.csv',
        "mapping_file": './data/mpii/mapping_mpii.txt',
        "n_splits": len(testing_subjects),
    }
    metadata.update(kwargs)
    df = pd.read_csv(
        metadata['gt_root'],
        names=["subject", 'file_name', 'start_frame', 'end_frame', 'activity_category_id', 'activity_category_name'])
    df['start_frame'] -= 1

    gb_df = df.groupby(['file_name', 'subject'])
    new_df = gb_df['activity_category_name'].apply(list).to_frame(name='activity_category_name')
    new_df['start_frame'] = gb_df['start_frame'].apply(list)
    new_df['end_frame'] = gb_df['end_frame'].apply(list)

    dataset_dict = []
    for row in new_df.reset_index().itertuples():
        gt = list(zip(row.start_frame, row.activity_category_name, row.start_frame, row.end_frame))
        gt = [g[1:] for g in sorted(gt)]
        dataset_dict.append({
            # "video_path": os.path.join(metadata['video_root'], f'{row.file_name}.avi'),
            "video_path": os.path.join(metadata['video_root'], f'{row.file_name}.mp4'),
            "annotations": gt,
            "video_len": gt[-1][-1],
            "split": testing_subjects.index(row.subject) if row.subject in testing_subjects else len(testing_subjects),
        })
    return dataset_dict, metadata
