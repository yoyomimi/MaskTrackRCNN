import numpy as np
import random
import torch
from pycocotools.coco import COCO
import pycocotools.mask as rletools
from torch.utils.data import Dataset
from .pipelines import Compose, to_tensor
from .registry import DATASETS
import os
from mmcv.parallel import DataContainer as DC

TRAINVAL_SEQ = {'MOTS20-02': (1920,1080), 'MOTS20-05': (640,480), 'MOTS20-09': (1920,1080), 'MOTS20-11':(1920,1080)}
# TRAINVAL_SEQ = {'MOTS20-05': (640,480)}
TEST_SEQ = {'MOTS20-01': (1920,1080), 'MOTS20-06': (640,480), 'MOTS20-07': (1920,1080), 'MOTS20-12': (1920,1080)}
TEST_SEQ = {'MOTS20-05': (640,480)}

class SegmentedObject:
    def __init__(self, mask, class_id, track_id):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id % 1000

@DATASETS.register_module
class MOTSDataset(Dataset):

    CLASSES = ('person')

    def __init__(self,
                 data_root,
                 pipeline,
                 test_mode=False):
        super().__init__()
        self.data_root = data_root
        self.test_mode = test_mode

        self.pipeline = Compose(pipeline)

        self.img_infos, self.vid_infos, self.idx_vid_infos = self.load_annotations(self.data_root)
        
        if not self.test_mode:
            self._set_group_flag()

    def load_txt(self, path):
        objects_per_frame = {}
        track_ids_per_frame = {}  # To check that no frame contains two objects with same id
        combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                fields = line.split(" ")

                frame = int(fields[0])
                if frame not in objects_per_frame:
                    objects_per_frame[frame] = []
                if frame not in track_ids_per_frame:
                    track_ids_per_frame[frame] = set()
                if int(fields[1]) in track_ids_per_frame[frame]:
                    assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
                else:
                    track_ids_per_frame[frame].add(int(fields[1]))

                class_id = int(fields[2])
                if not (class_id == 1 or class_id == 2 or class_id == 10):
                    assert False, "Unknown object class " + fields[2]

                mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
                if frame not in combined_mask_per_frame:
                    combined_mask_per_frame[frame] = mask
                elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
                    assert False, "Objects with overlapping masks in frame " + fields[0]
                else:
                    combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask],
                                                                    intersect=False)
                objects_per_frame[frame].append(SegmentedObject(
                    mask,
                    class_id,
                    int(fields[1])
                ))

        return objects_per_frame
   
    def load_annotations(self, data_root):
        img_infos = []
        vid_infos = {}
        idx_vid_infos = []
        if not self.test_mode:
            seq_id = 0
            for seq in TRAINVAL_SEQ.keys():
                seq_id += 1 # from 1
                if seq_id not in vid_infos.keys():
                    vid_infos[seq_id] = {}
                object_per_frame = self.load_txt(path=os.path.join(data_root, seq, 'gt', 'gt.txt'))
                for k,v in object_per_frame.items():
                    frame_id = int(k) - 1
                    img_info = {
                        'filename': os.path.join(data_root, seq, 'img1', str(k).zfill(6)+'.jpg'),
                        'height': TRAINVAL_SEQ[seq][1],
                        'width': TRAINVAL_SEQ[seq][0],
                        'vid': seq_id,
                        'frame_id': frame_id
                    }
                    ann_info = {
                        'bboxes': None,
                        'masks': [],
                        'mask_ignore': None,
                        'labels': [],
                        'ids': [],
                    }
                    for i in v:
                        if i.class_id == 10:
                            ann_info['mask_ignore'] = i.mask
                        else:
                            ann_info['masks'].append(i.mask)
                            ann_info['labels'].append(1)
                            ann_info['ids'].append(i.track_id + seq_id * 1000)
                    img_infos.append((img_info, ann_info))
                    
                    # from 0
                    if frame_id not in vid_infos[seq_id].keys():
                        vid_infos[seq_id][frame_id] = len(img_infos) - 1
                    idx_vid_infos.append((seq_id, frame_id))
        else:
            for seq in TEST_SEQ.keys():
                object_per_frame = self.load_txt(path=os.path.join(data_root, seq, 'gt', 'gt.txt'))
                for k,v in object_per_frame.items():
                    img_info = {
                        'filename': os.path.join(data_root, seq, 'img1', str(k).zfill(6)+'.jpg'),
                        'height': TRAINVAL_SEQ[seq][1],
                        'width': TRAINVAL_SEQ[seq][0],
                    }
                    img_infos.append(img_info)
        return img_infos, vid_infos, idx_vid_infos

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i][0]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
                
    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = self.idx_vid_infos[idx]
        frame_num = len(self.vid_infos[vid])
        sample_range = range(frame_num)
        valid_samples = []
        for i in sample_range:
            # check if the frame id is valid
            ref_idx = (vid, i)
            if i != frame_id and ref_idx in self.idx_vid_infos:
                valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        ref_vid, ref_frame_id = random.choice(valid_samples)
        return self.vid_infos[ref_vid][ref_frame_id]

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data, ref_data = self.prepare_train_img(idx)
            if (data is None) or (ref_data is None):
                idx = self._rand_another(idx)
                continue
            data['ref_img'] = ref_data['img']
            data['ref_bboxes'] = ref_data['gt_bboxes']
            ref_ids = np.array(ref_data['gt_ids'])
            ids = data['gt_ids']
            for i in range(len(ids)):
                ref_id_index = np.where(ref_ids == ids[i])[0]
                if len(ref_id_index) == 0:
                    ids[i] = 0
                else:
                    ids[i] = ref_id_index[0] + 1
            data['gt_ids'] = DC(to_tensor(ids))
            
            return data

    def pre_pipeline(self, results):
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['id_fields'] = []
        return results

    def prepare_train_img(self, idx):
        ref_idx = self.sample_ref(idx)
        img_info, ann_info = self.img_infos[idx]
        ref_img_info, ref_ann_info = self.img_infos[ref_idx]
        results = dict(img_info=img_info, ann_info=ann_info)
        ref_results = dict(img_info=ref_img_info, ann_info=ref_ann_info)
        results = self.pre_pipeline(results)
        ref_results = self.pre_pipeline(ref_results)
        return self.pipeline(results), self.pipeline(ref_results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        results = self.pre_pipeline(results)
        return self.pipeline(results)

    def __len__(self):
        return len(self.img_infos)