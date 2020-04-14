from __future__ import division

import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import (assign_and_sample, bbox2roi, bbox2result, bbox2result_with_id, multi_apply,
                        merge_aug_masks, bbox_overlaps)


@DETECTORS.register_module
class CascadeTrackRCNN(BaseDetector, RPNTestMixin, MaskTestMixin):

    def __init__(self,
                 num_stages,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 track_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(CascadeTrackRCNN, self).__init__()

        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [
                    bbox_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))

        if track_head is not None:
            self.track_head = nn.ModuleList()
            if not isinstance(track_head, list):
                track_head = [track_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(track_head) == self.num_stages
            for head in track_head:
                self.track_head.append(builder.build_head(head))

        if mask_head is not None:
            self.mask_roi_extractor = nn.ModuleList()
            self.mask_head = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(mask_head, list):
                mask_head = [mask_head for _ in range(num_stages)]
            assert len(mask_roi_extractor) == len(mask_head) == self.num_stages
            for roi_extractor, head in zip(mask_roi_extractor, mask_head):
                self.mask_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.mask_head.append(builder.build_head(head))
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(CascadeTrackRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()
            self.track_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      ref_img, 
                      ref_bboxes,
                      gt_masks=None,
                      mask_ignore=None,
                      gt_ids=None,
                      proposals=None):
        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_img)
        losses = dict()

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        ref_rois = bbox2roi(ref_bboxes)
        ref_bbox_img_n = [x.size(0) for x in ref_bboxes]

        for i in range(self.num_stages):
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            assign_results, sampling_results = multi_apply(
                assign_and_sample,
                proposal_list,
                gt_bboxes,
                gt_labels,
                gt_ids,
                cfg=rcnn_train_cfg)

            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]
            track_head = self.track_head[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
            ref_bbox_feats = bbox_roi_extractor(
                ref_x[:bbox_roi_extractor.num_inputs], ref_rois)
            cls_score, bbox_pred = bbox_head(bbox_feats)

            bbox_targets, (ids, id_weights) = bbox_head.get_target(sampling_results, gt_bboxes,
                                                gt_labels, rcnn_train_cfg)
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (value * lw if
                                                    'loss' in name else value)
            match_score = track_head(bbox_feats, ref_bbox_feats, 
                                          bbox_img_n, ref_bbox_img_n)
            loss_match = track_head.loss(match_score, ids, id_weights)
            for name, value in loss_match.items():
                losses['s{}.{}'.format(i, name)] = (value * lw if
                                                    'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_roi_extractor = self.mask_roi_extractor[i]
                mask_head = self.mask_head[i]
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = mask_roi_extractor(
                    x[:mask_roi_extractor.num_inputs], pos_rois)
                mask_pred = mask_head(mask_feats)
                mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                                    rcnn_train_cfg)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (value * lw
                                                        if 'loss' in name else
                                                        value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]  # bbox_targets is a tuple
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

        return losses

    def simple_test_bboxes(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        is_first = img_meta[0]['is_first']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_id_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)

        det_roi_feats = None
        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
            det_roi_feats = bbox_feats.copy()

            cls_score, bbox_pred = bbox_head(bbox_feats)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    cfg=rcnn_test_cfg)

                bbox_result = (det_bboxes, det_labels)

                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        segm_result = [
                            [] for _ in range(mask_head.num_classes - 1)
                        ]
                    else:
                        _bboxes = (det_bboxes[:, :4] * scale_factor
                                   if rescale else det_bboxes)
                        mask_rois = bbox2roi([_bboxes])
                        mask_feats = mask_roi_extractor(
                            x[:len(mask_roi_extractor.featmap_strides)],
                            mask_rois)
                        mask_pred = mask_head(mask_feats)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        
        bbox_result = (det_bboxes, det_labels)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [
                    [] for _ in range(self.mask_head[-1].num_classes - 1)
                ]
            else:
                _bboxes = (det_bboxes[:, :4] * scale_factor
                           if rescale else det_bboxes)
                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_feats = mask_roi_extractor(
                        x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                    mask_pred = self.mask_head[i](mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        det_obj_ids = None
        if det_bboxes.shape[0] == 0:
            det_obj_ids=np.array([], dtype=np.int64)
            if is_first:
                self.prev_bboxes =  None
                self.prev_roi_feats = None
                self.prev_det_labels = None
        elif is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.size(0))
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = det_roi_feats
            self.prev_det_labels = det_labels
        else:
            assert self.prev_roi_feats is not None
            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)]
            prev_bbox_img_n = [self.prev_roi_feats.size(0)]
            match_score = self.track_head[-1](det_roi_feats, self.prev_roi_feats,
                                      bbox_img_n, prev_bbox_img_n)[0]
            match_logprob = nn.functional.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1,1)).float()
            bbox_ious = bbox_overlaps(det_bboxes[:,:4], self.prev_bboxes[:,:4])
            # compute comprehensive score 
            comp_scores = self.track_head[-1].compute_comp_scores(match_logprob, 
                det_bboxes[:,4].view(-1, 1),
                bbox_ious,
                label_delta,
                add_bbox_dummy=True)
            match_likelihood, match_ids = torch.max(comp_scores, dim =1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object, 
            # add tracking features/bboxes of new object
            match_ids = match_ids.cpu().numpy().astype(np.int32)
            det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)
            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    # add new object
                    det_obj_ids[idx] = self.prev_roi_feats.size(0)
                    self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score 
                    obj_id = match_id - 1
                    match_score = comp_scores[idx, match_id]
                    if match_score > best_match_scores[obj_id]:
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        # udpate feature
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]

        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'],
                           ms_id_result)
            else:
                results = (ms_bbox_result['ensemble'],
                           ms_id_result)
        else:
            if self.with_mask:
                results = {
                    stage: (ms_bbox_result[stage], ms_segm_result[stage], ms_id_result)
                    for stage in ms_bbox_result
                }
            else:
                results = (ms_bbox_result, ms_id_result)

        return results

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        (det_bboxes, det_labels), segm_results, det_obj_ids = self.simple_test_bboxes(
            img, img_meta, proposals, rescale)
        bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids,
                                   self.bbox_head.num_classes)
        return bbox_results, segm_results

    def aug_test(self, img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError
