import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str

@PIPELINES.register_module
class MOTSLoadImageFromFile(object):

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_info']['filename']
        img = mmcv.imread(filename, self.color_type)
        frame_id = results['img_info']['frame_id']
        video_id = results['img_info']['vid']
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['is_first'] = (frame_id == 0)
        results['video_id'] = video_id
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)

@PIPELINES.register_module
class MOTSLoadAnnotations(object):

    def __init__(self):
        pass

    def _load_bboxes(self, results):
        results['gt_bboxes'] = []
        # import matplotlib.pyplot as plt
        id = 0
        for i in results['gt_masks']:
            coord = np.where(i==1)
            min_x = float(np.min(coord[0]))
            max_x = float(np.max(coord[0]))
            min_y = float(np.min(coord[1]))
            max_y = float(np.max(coord[1]))
            # i[int(min_x):int(max_x)+1, int(min_y):int(max_y)+1] = 1
            # plt.imsave(str(id)+'.jpg', i, cmap='gray')
            results['gt_bboxes'].append([min_x, min_y, max_x, max_y])
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _load_masks(self, results):
        # print(results['ann_info']['masks'])
        results['gt_masks'] = [maskUtils.decode(i) for i in results['ann_info']['masks']] # list length=num_instance, [h, w]
        # print(len(results['gt_masks']))
        # for i in results['gt_masks']:
        #     print(i.shape)
        results['mask_fields'].append('gt_masks')
        return results

    def _load_mask_ignore(self, results):
        if results['ann_info']['mask_ignore'] is None:
            results['mask_ignore'] = [np.zeros((results['img'].shape[0], results['img'].shape[1]))]
        else:
            results['mask_ignore'] = [maskUtils.decode(results['ann_info']['mask_ignore'])] # list length=1, [h, w]
        results['mask_fields'].append('mask_ignore')
        return results

    def _load_ids(self, results):
        results['gt_ids'] = results['ann_info']['ids']
        results['id_fields'].append('gt_ids')
        return results

    def __call__(self, results):
        results = self._load_masks(results)
        results = self._load_mask_ignore(results)
        results = self._load_labels(results)
        results = self._load_bboxes(results)
        results = self._load_ids(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
