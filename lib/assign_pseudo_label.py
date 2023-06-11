from platform import release
import random
import torch
import os
import cv2
import numpy as np
import json
from random import choice

from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from lib.extract_bbox_features import extract_feature_given_bbox, extract_feature_given_bbox_video, extract_feature_given_bbox_base_feat


def load_feature(frame_names, union_box_feature, path='datasets/ActionGenome/dataset/ag/AG_detection_results'):
    """
    frame_names: a list of name like '001YG.mp4/000093.png'
    """
    total_paths = [os.path.join(path, f) for f in frame_names]
    dets_list = []
    feat_list = []
    base_feat_list = []
    for p in total_paths:
        dets_path = os.path.join(p, 'dets.npy')
        feat_path = os.path.join(p, 'feat.npy')
        dets = np.load(dets_path, allow_pickle=True)
        feat = np.load(feat_path)
        dets_list.append(dets)
        feat_list.append(feat)

        if union_box_feature:
            base_feat_path = os.path.join(p, 'base_feat.npy')
            base_feat = np.load(base_feat_path)
            base_feat_list.append(base_feat)
            
    return dets_list, feat_list, base_feat_list


def assign_label_to_proposals_by_dict_for_image(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict, pseudo_way):
    
    # 先遍历一遍检查人
    people_oi_idx = cls_dict[1]
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] in people_oi_idx:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)
    if len(people_conf_list) != 0:
        final_people_idx = people_conf_list.index(max(people_conf_list))
        # final_people_idx上一步是在people_cong_list的index，要转换一下
        final_people_idx = people_idx[final_people_idx]
        people_det = img_det[final_people_idx]
        people_det['class'] = 1
        people_feat = img_feat[final_people_idx]
    else:
        # print("cannot find people")
        if pseudo_way == 0:
            return [], [], [], []
        elif pseudo_way == 1:
            final_people_idx = 0
            people_det = img_det[final_people_idx]
            people_det['class'] = 1
            people_feat = img_feat[final_people_idx]
        
    # 获取gt中label列表
    gt_ag_class_list = []
    for pair_info in img_gt_annotation:
        if 'class' in pair_info:
            gt_ag_class_list.append(pair_info['class'])
    # 获取在gt中有对象的object列表
    object_idx = []
    object_det = []
    object_feat = []
    for bbox_idx, bbox_det in enumerate(img_det):
        # 排除人
        if bbox_idx == final_people_idx:
            continue
        if bbox_det['class'] in people_oi_idx:
            continue
        # 获取bbox对应的ag中类别
        bbox_ag_class_list = oi_to_ag_cls_dict[bbox_det['class']]
        # 区分train和test，train的时候要和gt比较才加入，test只要类别在ag中就加入
        # 考虑oi中类别对应多个ag中类别
        if is_train:
            bbox_ag_class_list = list(set(bbox_ag_class_list) & set(gt_ag_class_list))
            if len(bbox_ag_class_list) > 0:
                for c in bbox_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
                    object_det.append(bbox_det.copy())
                    object_feat.append(img_feat[bbox_idx])
        else:
            if len(bbox_ag_class_list) > 0:
                for c in bbox_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
                    object_det.append(bbox_det.copy())
                    object_feat.append(img_feat[bbox_idx])
    return people_det, people_feat, object_det, object_feat


def assign_label_to_proposals_by_dict_for_video(dets, feats, is_train, gt_annotation, dict_path='datasets', pseudo_way=0):

    cls_dict = np.load(os.path.join(dict_path, 'ag_to_oi_word_map_synset.npy'), allow_pickle=True).tolist()
    oi_to_ag_cls_dict = np.load(os.path.join(dict_path, 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()
    
    video_people_det = []
    video_people_feat = []
    video_object_det = []
    video_object_feat = []
    for i in range(len(dets)):
        people_det, people_feat, object_det, object_feat = assign_label_to_proposals_by_dict_for_image(dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict, pseudo_way)
        video_people_det.append(people_det)
        video_people_feat.append(people_feat)
        video_object_det.append(object_det)
        video_object_feat.append(object_feat)

    return video_people_det, video_people_feat, video_object_det, video_object_feat


def create_dis(conf, idx):
    distrubution = torch.zeros(36)
    distrubution[idx] = conf
    distrubution[torch.where(distrubution==0)] = (1-conf) / 35
    return distrubution


def create_dis_list(FINAL_SCORES_OI, PRED_LABELS_OI):
    oi_to_ag_cls_dict = np.load(os.path.join('datasets', 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()

    all_ag_id = list(range(2, 36))
    dis_ag = torch.zeros((len(FINAL_SCORES_OI), 36), device=FINAL_SCORES_OI.device)
    for i in range(len(FINAL_SCORES_OI)):
        conf = FINAL_SCORES_OI[i].item()
        # 获取bbox对应的ag中类别
        bbox_ag_class_list = oi_to_ag_cls_dict[PRED_LABELS_OI[i].item()]
        if bbox_ag_class_list != []:
            idx = random.choice(bbox_ag_class_list)
        else:
            idx = random.choice(all_ag_id)
            # 直接取概率最高的table
            # idx = object_freq[0]
        dis_ag[i] = create_dis(conf, idx-1)

    return dis_ag


def category_oi2ag(dis_oi):
    oi_to_ag_cls_dict = np.load(os.path.join('datasets', 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()

    dis_ag = torch.zeros((len(dis_oi), 36), device=dis_oi.device)
    for dis_id, one_dis in enumerate(dis_oi):
        for oi_id, mapped_ag_id_list in oi_to_ag_cls_dict.items():
            for ag_id in mapped_ag_id_list:
                dis_ag[dis_id][ag_id-1] += one_dis[oi_id]

    return dis_ag


def prepare_func(thresh=0.2):
    config_file = "configs/detector/vinvl_x152c4.yaml"
    opts = ["MODEL.WEIGHT", "models/vinvl/vinvl_vg_x152c4.pth", 
            "MODEL.ROI_HEADS.NMS_FILTER", "1",
            "MODEL.ROI_HEADS.SCORE_THRESH", str(thresh),
            "DATA_DIR", "datasets",
            "TEST.IGNORE_BOX_REGRESSION", "False"]

    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    transforms = build_transforms(cfg, is_train=False)

    return model, transforms


def convert_data(is_train, base_feat_list, video_people_det, video_people_feat, video_object_det, video_object_feat, \
    gt_annotation, frame_names, faset_rcnn_model, transforms, union_box_feature):
    # 将video_people_det, video_people_feat, video_object_det, video_object_feat转换成entry的格式

    frame_num = len(video_people_det)
    bbox_num = 0

    for idx in range(frame_num):
        if video_people_det[idx] != []:
            bbox_num += 1
            bbox_num += len(video_object_det[idx])

    # bbox_num = 0
    MyDevice = torch.device('cuda:0')
    boxes = torch.zeros((bbox_num, 5), device=MyDevice)

    labels = torch.zeros(bbox_num, dtype=torch.int64, device=MyDevice)
    # obj_labels = torch.zeros(bbox_num-frame_num, dtype=torch.int64, device=MyDevice)
    scores = torch.zeros(bbox_num, device=MyDevice)
    distribution = torch.zeros((bbox_num, 36), device=MyDevice)
    features = torch.zeros((bbox_num, 2048), device=MyDevice)
    im_idx = []
    pair_idx = []
    a_rel = []
    s_rel = []
    c_rel = []
    rel_gt = []
    box_idx = []

    bbox_cnt = 0
    for idx in range(frame_num):

        if video_people_det[idx] != []:
            people_det = video_people_det[idx]
            people_feat = video_people_feat[idx]
            object_det = video_object_det[idx]
            object_feat = video_object_feat[idx]
            
            # 构造 boxes labels scores distrubution features
            boxes[bbox_cnt][0] = idx
            boxes[bbox_cnt][1:5] = torch.Tensor(people_det['rect'])
            labels[bbox_cnt] = people_det['class']
            scores[bbox_cnt] = people_det['conf']
            distribution[bbox_cnt] = create_dis(people_det['conf'], people_det['class'] - 1)  # because '__background__' is not a label
            features[bbox_cnt] = torch.from_numpy(people_feat)

            people_bbox_idx = bbox_cnt # 记录people的序号，之后im_idx要用
            box_idx.append(idx)
            bbox_cnt += 1

            for bbox_det, bbox_feat in zip(object_det, object_feat):
                boxes[bbox_cnt][0] = idx
                boxes[bbox_cnt][1:5] = torch.Tensor(bbox_det['rect'])
                labels[bbox_cnt] = bbox_det['class']
                scores[bbox_cnt] = bbox_det['conf']
                distribution[bbox_cnt] = create_dis(bbox_det['conf'], bbox_det['class'] - 1)  # because '__background__' is not a label
                features[bbox_cnt] = torch.from_numpy(bbox_feat)
            
                # 构造 im_idx pair_idx
                '''
                img_gt_annotation = gt_annotation[idx]
                for obj_info in img_gt_annotation:
                    if 'class' in obj_info:
                        if obj_info['class'] == bbox_det['class']:
                            # 在gt中找到对应的object
                            im_idx.append(idx)
                            pair_idx.append([people_bbox_idx, bbox_cnt])
                            a_rel.append(obj_info['attention_relationship'].tolist())
                            s_rel.append(obj_info['spatial_relationship'].tolist())
                            c_rel.append(obj_info['contacting_relationship'].tolist())
                '''
                img_gt_annotation = gt_annotation[idx]
                # 注意warning：这里im_idx和pair_idx，只有training时候才筛选，testing的时候不筛选
                # testing的时候，也不需要pseudo gt了
                if is_train:
                    for obj_info in img_gt_annotation:
                        if 'class' in obj_info:
                            if obj_info['class'] == bbox_det['class']:
                                # 在gt中找到对应的object
                                im_idx.append(idx)
                                pair_idx.append([people_bbox_idx, bbox_cnt])
                                a_rel.append(obj_info['attention_relationship'].tolist())
                                s_rel.append(obj_info['spatial_relationship'].tolist())
                                c_rel.append(obj_info['contacting_relationship'].tolist())
                                if obj_info['object_source']['ar'][-1] == '1gt' and obj_info['object_source']['sr'][-1] == '1gt' and obj_info['object_source']['cr'][-1] == '1gt':
                                    rel_gt.append(True)
                                elif obj_info['object_source']['ar'][-1] == 'gt' and obj_info['object_source']['sr'][-1] == 'gt' and obj_info['object_source']['cr'][-1] == 'gt':
                                    rel_gt.append(True)
                                elif obj_info['object_source']['ar'][-1] == '1' and obj_info['object_source']['sr'][-1] == '1' and obj_info['object_source']['cr'][-1] == '1':
                                    rel_gt.append(False)
                                else:
                                    print(obj_info['object_source']['ar'][-1], obj_info['object_source']['sr'][-1], obj_info['object_source']['cr'][-1], 'Error!')
                                break
                else:
                    im_idx.append(idx)
                    pair_idx.append([people_bbox_idx, bbox_cnt])

                box_idx.append(idx)
                bbox_cnt += 1

    rel_gt = torch.tensor(rel_gt, device=MyDevice)
    box_idx = torch.tensor(box_idx, device=MyDevice)
    im_idx = torch.tensor(im_idx, device=MyDevice)
    pair_idx = torch.tensor(pair_idx, device=MyDevice).long()

    rel_num = len(pair_idx)
    if rel_num == 0:
        return None
    '''
    else:
        return {'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'distribution': distribution,
            'im_idx': im_idx,
            'pair_idx': pair_idx,
            'features': features,
            'union_feat': torch.zeros((rel_num, 2048, 7, 7), device=MyDevice),
            'spatial_masks': torch.zeros((rel_num, 2, 27, 27), device=MyDevice),
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel}
    '''
    if union_box_feature:
        # for detection union boxes

        imgs_paths = [os.path.join('datasets/ActionGenome/dataset/ag/frames', f) for f in frame_names]
        cv2_imgs = [cv2.imread(img_file) for img_file in imgs_paths]

        union_boxes = torch.cat((im_idx[:, None],
                                torch.min(boxes[:, 1:3][pair_idx[:, 0]],
                                        boxes[:, 1:3][pair_idx[:, 1]]),
                                torch.max(boxes[:, 3:5][pair_idx[:, 0]],
                                        boxes[:, 3:5][pair_idx[:, 1]])), 1)
        union_boxes_list = [union_boxes[union_boxes[:, 0] == i] for i in range(frame_num)]
        union_feat_list = []
        
        for i, union_boxes_one_image in enumerate(union_boxes_list):
            if len(union_boxes_list[i]) > 0:
                union_feat_list.append(extract_feature_given_bbox(faset_rcnn_model, transforms, cv2_imgs[i], union_boxes_list[i][:, 1:]))
                # union_feat_list.append(extract_feature_given_bbox_base_feat(faset_rcnn_model, transforms, cv2_imgs[i], union_boxes_list[i][:, 1:], base_feat_list[i]))
            else:
                union_feat_list.append(torch.Tensor([]).cuda(0))
        union_feat = torch.cat(union_feat_list)
        '''
        imgs = []
        bboxes = []
        for i, union_boxes_one_image in enumerate(union_boxes_list):
            if len(union_boxes_list[i]) > 0:
                imgs.append(cv2_imgs[i])
                bboxes.append(union_boxes_list[i][:, 1:])
        # bboxes = union_boxes_list[:][:, 1:]
        union_feat_list = extract_feature_given_bbox_video(faset_rcnn_model, transforms, cv2_imgs, bboxes)
        union_feat = union_feat_list
        '''

    else:
        union_feat = torch.zeros((rel_num, 2048, 7, 7), device=MyDevice)
        # union_feat = torch.randn(rel_num, 2048, 7, 7).cuda(0)

    if pair_idx.shape[0] == 0:
        spatial_masks = torch.zeros((rel_num, 2, 27, 27), device=MyDevice)
    else:
        pair_rois = torch.cat((boxes[pair_idx[:,0],1:],boxes[pair_idx[:,1],1:]), 1).data.cpu().numpy()
        spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5, device=MyDevice)
            
    obj_labels = labels[labels != 1]
    # obj_boxes = boxes[labels != 1]
    
    entry = {'boxes': boxes,
            'labels': labels,
            'obj_labels': obj_labels,
            'scores': scores,
            'distribution': distribution,
            'im_idx': im_idx,
            'pair_idx': pair_idx,
            'features': features,
            'union_feat': union_feat,
            'spatial_masks': spatial_masks,
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel,
            'rel_gt': rel_gt,
            'box_idx': box_idx}

    return entry


#############################################
# test the detector
#############################################

def entry_to_pred(entry):
    # convert entry to pred directly
    if entry == None:
        return {}

    entry['pred_labels'] = entry['labels']
    entry['pred_scores'] = entry['scores']
    rel_num = len(entry['attention_gt'])
    attention_distribution = torch.zeros(rel_num, 3).cuda(0)
    spatial_distribution = torch.zeros(rel_num, 6).cuda(0)
    contacting_distribution = torch.zeros(rel_num, 17).cuda(0)

    for i in range(rel_num):
        # attention_distribution[i][entry['attention_gt'][i]] = 1 / len(entry['attention_gt'][i])
        # spatial_distribution[i][entry['spatial_gt'][i]] = 1 / len(entry['spatial_gt'][i])
        # contacting_distribution[i][entry['contacting_gt'][i]] = 1 / len(entry['contacting_gt'][i])
        attention_distribution[i][entry['attention_gt'][i]] = 1
        spatial_distribution[i][entry['spatial_gt'][i]] = 1
        contacting_distribution[i][entry['contacting_gt'][i]] = 1

    entry['attention_distribution'] = attention_distribution
    entry['spatial_distribution'] = spatial_distribution
    entry['contacting_distribution'] = contacting_distribution

    return entry



#############################################
# debug
#############################################

def count_person_and_object_for_image(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict):
    """
    only use a dictionary to assign gt object labels
    TODO: using box location to match gt objects
    dict中有映射、gt中有对象保留，其他舍去（gt中同一个对象应该不会有两个）
    注意先检查人
    """
    
    has_person_img = True

    # 检查人的部分不需要区分训练和测试
    # 先遍历一遍检查人
    # 因为肯定有人所以不和gt比
    people_oi_idx = cls_dict[1]
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] in people_oi_idx:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)
    if len(people_conf_list) != 0:
        has_person_img = True
        final_people_idx = people_conf_list.index(max(people_conf_list))
        people_det = img_det[final_people_idx]
        people_det['class'] = 1
        people_feat = img_feat[final_people_idx]
    else:
        has_person_img = False
        return has_person_img, 0
        # final_people_idx = 0
        # people_det = img_det[final_people_idx]
        # people_det['class'] = 1
        # people_feat = img_feat[final_people_idx]
        
    # 获取gt中label列表
    gt_ag_class_list = []
    for pair_info in img_gt_annotation:
        if 'class' in pair_info:
            gt_ag_class_list.append(pair_info['class'])
    # 获取在gt中有对象的object列表
    object_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        # 排除人
        if bbox_idx == final_people_idx:
            continue
        if bbox_det['class'] in people_oi_idx:
            continue
        # 获取bbox对应的ag中类别
        bbox_ag_class_list = oi_to_ag_cls_dict[bbox_det['class']]
        # 区分train和test，train的时候要和gt比较才加入，test只要类别在ag中就加入
        if is_train:
            for c in bbox_ag_class_list:
                if c in gt_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
        else:
            if len(bbox_ag_class_list) > 0:
                c = choice(bbox_ag_class_list)
                bbox_det['class'] = c
                object_idx.append(bbox_idx)

    return has_person_img, len(object_idx)



def count_person_and_object_for_video(dets, feats, is_train, gt_annotation, cls_dict, oi_to_ag_cls_dict, frame_names):

    f_names = [f.split('/')[1] for f in frame_names]
    info_dict = {}
    no_person_img_cnt = 0
    with_person_img_cnt = 0
    total_rel_cnt = 0

    for i in range(len(dets)):
        has_person_img, rel_cnt = count_person_and_object_for_image(dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict)
        info_dict[f_names[i]] = (has_person_img, rel_cnt)
        if has_person_img:
            with_person_img_cnt += 1
        else:
            no_person_img_cnt += 1
        total_rel_cnt += rel_cnt

    return info_dict, no_person_img_cnt, with_person_img_cnt, total_rel_cnt
