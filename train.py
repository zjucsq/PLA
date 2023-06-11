import os
import argparse
import json
from lib.config import conf, cfg_from_file

"""------------------------------------some settings----------------------------------------"""
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/demo.yml', type=str)
args = parser.parse_args()
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
print('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
print('-------------student model setting------------------')
print(args.cfg_file)
print(conf)
with open(os.path.join(conf.save_path, "configs.json"), 'w') as f:
    json.dump(conf, f)
"""-----------------------------------------------------------------------------------------"""

os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.gpu_id)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import log_softmax, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
np.set_printoptions(precision=3)
import time
import pandas as pd
import copy
from tqdm import tqdm
torch.set_num_threads(4)
from tensorboardX import SummaryWriter

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from lib.sttran import STTran
from lib.assign_pseudo_label import prepare_func
from lib.transition_module import transition_module
from lib.ults.track import track, track_diff, track_iou, track_diff_iou
from lib.ults.init_teacher_model import init_teacher_model


AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, ws_object_bbox_path=conf.ws_object_bbox_path, remove_one_frame_video=conf.remove_one_frame_video,
                      filter_nonperson_box_frame=True, filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=conf.num_workers,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, ws_object_bbox_path=None, remove_one_frame_video=True,
                     filter_nonperson_box_frame=True, filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=conf.num_workers,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda")
# freeze the detection backbone
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, conf=conf).to(device=gpu_device)
object_detector.eval()

if conf.union_box_feature:
    faset_rcnn_model, transforms = prepare_func()
else:
    faset_rcnn_model = None
    transforms = None

model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset_train.attention_relationships),
               spatial_class_num=len(AG_dataset_train.spatial_relationships),
               contact_class_num=len(AG_dataset_train.contacting_relationships),
               obj_classes=AG_dataset_train.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               transformer_mode=conf.transformer_mode,
               is_wks=conf.is_wks,
               feat_dim=conf.feat_dim).to(device=gpu_device)
print("create student model successfully")
print('*'*50)

if conf.teacher_mode_cfg is None:
    print("Do not need to create teacher model")
    print('*'*50)
else:
    t_model = init_teacher_model(conf.teacher_mode_cfg, AG_dataset_train, gpu_device)
    print("create teacher model successfully")
    print('*'*50)

if conf.ckpt is None:
    print('Do not need to load CKPT')
    start_epoch = 0
else:
    ckpt = torch.load(conf.ckpt, map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print('CKPT {} is loaded'.format(conf.ckpt))
    left_pos = conf.ckpt.rfind('_')
    right_pos = conf.ckpt.rfind('.')
    start_epoch = int(conf.ckpt[left_pos+1:right_pos]) + 1

evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset_train.object_classes,
    AG_all_predicates=AG_dataset_train.relationship_classes,
    AG_attention_predicates=AG_dataset_train.attention_relationships,
    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
    iou_threshold=0.5,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset_train.object_classes,
    AG_all_predicates=AG_dataset_train.relationship_classes,
    AG_attention_predicates=AG_dataset_train.attention_relationships,
    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)

evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset_train.object_classes,
    AG_all_predicates=AG_dataset_train.relationship_classes,
    AG_attention_predicates=AG_dataset_train.attention_relationships,
    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
    iou_threshold=0.5,
    constraint='no')

# loss function, default Multi-label margin loss
if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    # bce_loss = nn.BCEWithLogitsLoss()
    bce_loss = nn.BCELoss()
    if conf.loss == 'KL':
        # kl_loss = nn.KLDivLoss(reduction="batchmean")
        kl_loss = nn.KLDivLoss(reduction="sum")
    elif conf.loss == 'L1':
        L1_loss = nn.L1Loss()
    elif conf.loss == 'L2':
        L2_loss = nn.MSELoss()
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()


softmax = nn.Softmax(dim=1)
sigmoid = nn.Sigmoid()

if conf.transition_module:
    trans_module = transition_module().to(device=gpu_device)
else:
    trans_module = None

# optimizer
if conf.optimizer == 'adamw':
    # optimizer = AdamW(model.parameters(), lr=conf.lr)
    if trans_module is None:
        optimizer = AdamW(model.parameters(), lr=conf.lr)
    elif trans_module is not None:
        optimizer = AdamW([{'params': model.parameters()}, {'params': trans_module.parameters(), 'lr': conf.t_lr}], lr=conf.lr)
        # optimizer = AdamW([{'params': model.parameters()}, {'params': trans_module.parameters(), 'lr': conf.lr}], lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

writer = SummaryWriter(conf.tensorboard_name)

# some parameters
tr = []

test_res = {}

save_loss = {}

save_epoch = 1000

for epoch in range(start_epoch, conf.nepoch):
    model.train()
    object_detector.is_train = True
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)

    alpha1_cnt = 0
    alpha2_cnt = 0
    loss_cnt_tr = []

    with tqdm(total=len(dataloader_train)) as t:
        for b in range(len(dataloader_train)):
            data = next(train_iter)

            if conf.is_wks:
                im_data = None
                im_info = None
                gt_boxes = None
                num_boxes = None
            else:
                im_data = copy.deepcopy(data[0].cuda(0))
                im_info = copy.deepcopy(data[1].cuda(0))
                gt_boxes = copy.deepcopy(data[2].cuda(0))
                num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_train.gt_annotations[data[4]]
            frame_names = AG_dataset_train.video_list[data[4]]

            # prevent gradients to FasterRCNN
            with torch.no_grad():
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, frame_names, faset_rcnn_model, transforms)
                t_entry = copy.deepcopy(entry)

            # pass when no relation
            # if entry['im_idx'].shape[0] == 0:
            if entry != None:

                if conf.teacher_mode_cfg is not None:
                    with torch.no_grad():
                        t_pred = t_model(t_entry)
                
                pred = model(entry)

                # rel_num*3/6/17
                attention_distribution = pred["attention_distribution"]
                spatial_distribution = pred["spatial_distribution"]
                contact_distribution = pred["contacting_distribution"]

                object_label = pred['labels']
                attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device)
                # 对于[[1]]这种形状，只squeeze一维
                if attention_label.shape[0] > 1:
                    attention_label.squeeze_()
                else:
                    attention_label.squeeze_(1)
                # attention_label: 一维tensor，每个值对应每对people-object的关系类别，整体like tensor([2, 1, 1, 2, 0, 0], device='cuda:0')
                if not conf.bce_loss:
                    # multi-label margin loss or adaptive loss
                    # spatial_label/contact_label: 二维tensor，其中每个一维tensor对应每对people-object的关系类别
                    # 每个一维tensor like tensor([ 2,  4, -1, -1, -1, -1], device='cuda:0')，-1之前的表示gt的关系
                    spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=attention_distribution.device)
                    contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=attention_distribution.device)
                    for i in range(len(pred["spatial_gt"])):
                        spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                        contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

                else:
                    # bce loss
                    # spatial_label/contact_label: 二维tensor，其中每个一维tensor对应每对people-object的关系类别
                    # 每个一维tensor like tensor([ 0,  0,  1,  0,  1,  0], device='cuda:0')，1表示gt的关系，其他表示没有gt关系
                    spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=attention_distribution.device)
                    contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=attention_distribution.device)
                    for i in range(len(pred["spatial_gt"])):
                        spatial_label[i, pred["spatial_gt"][i]] = 1
                        contact_label[i, pred["contacting_gt"][i]] = 1

                # 确定soft target和hard target的比例
                # 只需要soft target或者只需要hard target时，设定alpha=0或1即可
                if conf.temperature is None or conf.temperature == 1:
                    temperature = 1
                else:
                    temperature = conf.temperature

                losses = {}
                if conf.alpha is None or conf.alpha == 0:
                    alpha = 0
                else:
                    alpha = conf.alpha
                
                gt_rel = pred['rel_gt']
                
                if alpha != 0:
                    # 需要soft label计算蒸馏损失的情况，否则只需要hard label
                    if conf.teacher_mode_cfg is not None:                            
                        
                        student_object_distribution = pred['distribution']
                        student_attention_distribution = pred["attention_distribution"]
                        student_spatial_distribution = pred["spatial_distribution"]
                        student_contact_distribution = pred["contacting_distribution"]

                        teacher_object_distribution = t_pred['distribution']
                        teacher_attention_distribution = t_pred["attention_distribution"]
                        teacher_spatial_distribution = t_pred["spatial_distribution"]
                        teacher_contact_distribution = t_pred["contacting_distribution"]

                        if conf.label_fusion_strategy == 0:
                            fusion_spatial_distribution = teacher_spatial_distribution * alpha + spatial_label * (1-alpha)
                            fusion_contact_distribution = teacher_contact_distribution * alpha + contact_label * (1-alpha)
                        elif conf.label_fusion_strategy == 1:
                            pred_spatial_label, pred_contact_label = trans_module(teacher_spatial_distribution, teacher_contact_distribution, pred['obj_labels'])
                            pseudo_id, transition_id = track_iou(t_pred['labels'], t_pred['im_idx'], t_pred['boxes'][:, 1:5], 0.5)
                            spatial_relation_loss_pred = kl_loss(F.log_softmax(teacher_spatial_distribution[pseudo_id], dim=1), F.softmax(pred_spatial_label[transition_id], dim=1))
                            contact_relation_loss_pred = kl_loss(F.log_softmax(teacher_contact_distribution[pseudo_id], dim=1), F.softmax(pred_contact_label[transition_id], dim=1))

                            losses['spatial_relation_loss_pred'] = spatial_relation_loss_pred
                            losses['contact_relation_loss_pred'] = contact_relation_loss_pred
                            alpha1 = 2 - 2 * F.sigmoid(spatial_relation_loss_pred).detach()
                            alpha2 = 2 - 2 * F.sigmoid(contact_relation_loss_pred).detach()
                            fusion_spatial_distribution = teacher_spatial_distribution * alpha1 + spatial_label * (1-alpha1)
                            fusion_contact_distribution = teacher_contact_distribution * alpha2 + contact_label * (1-alpha2)
                            alpha1_cnt += alpha1.item()
                            alpha2_cnt += alpha2.item()

                        fusion_spatial_distribution[gt_rel] = spatial_label[gt_rel]
                        fusion_contact_distribution[gt_rel] = contact_label[gt_rel]
                        spatial_label = fusion_spatial_distribution
                        contact_label = fusion_contact_distribution

                if conf.mode == 'sgcls' or conf.mode == 'sgdet':
                    losses['object_loss'] = ce_loss(pred['distribution'], object_label)

                losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
                if not conf.bce_loss:
                    losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
                    losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)

                else:
                    if conf.loss == 'KL':
                        # print(spatial_distribution, spatial_label)
                        losses["spatial_relation_loss"] = kl_loss(F.log_softmax(spatial_distribution, dim=1), F.softmax(spatial_label, dim=1))
                        losses["contact_relation_loss"] = kl_loss(F.log_softmax(contact_distribution, dim=1), F.softmax(contact_label, dim=1))
                    elif conf.loss == 'L1':
                        losses["spatial_relation_loss"] = L1_loss(spatial_distribution, spatial_label)
                        losses["contact_relation_loss"] = L1_loss(contact_distribution, contact_label)
                    elif conf.loss == 'L2':
                        losses["spatial_relation_loss"] = L2_loss(spatial_distribution, spatial_label)
                        losses["contact_relation_loss"] = L2_loss(contact_distribution, contact_label)
                    elif conf.loss == 'BCE':
                        losses["spatial_relation_loss_BCE"] = bce_loss(spatial_distribution, spatial_label)
                        losses["contact_relation_loss_BCE"] = bce_loss(contact_distribution, contact_label)

                optimizer.zero_grad()
                loss = sum(losses.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()

                for k in losses.keys():
                    writer.add_scalar(k, losses[k], epoch * len(dataloader_train) + b)

                tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
                loss_cnt_tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

            if b % save_epoch == 0 and b >= save_epoch:
                time_per_batch = (time.time() - start) / save_epoch
                print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                    time_per_batch, len(dataloader_train) * time_per_batch / 60))

                mn = pd.concat(tr[-save_epoch:], axis=1).mean(1)
                print(mn)
                start = time.time()

            t.set_description(desc="Epoch {} ".format(epoch))
            # t.set_postfix(steps=step, loss=loss.data.item())
            t.update(1)

    # 一个epoch结束，存储这个epoch的loss
    save_loss[str(epoch)] = {}
    loss_all = pd.concat(loss_cnt_tr, axis=1).mean(1)
    for loss_name in loss_all.keys():
        save_loss[str(epoch)][loss_name] = loss_all[loss_name]
    save_loss[str(epoch)]['alpha1'] = alpha1_cnt / b
    save_loss[str(epoch)]['alpha2'] = alpha2_cnt / b
    with open(os.path.join(conf.save_path, "save_loss_{}.json".format(epoch)), 'w') as f:
        json.dump(save_loss, f)

    if trans_module is not None:
        torch.save({"state_dict": model.state_dict(), "state_dict_3": trans_module.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
    else:
        torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))

    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        with tqdm(total=len(dataloader_test)) as t:
            for b in range(len(dataloader_test)):
                data = next(test_iter)

                im_data = copy.deepcopy(data[0].cuda(0))
                im_info = copy.deepcopy(data[1].cuda(0))
                gt_boxes = copy.deepcopy(data[2].cuda(0))
                num_boxes = copy.deepcopy(data[3].cuda(0))
                gt_annotation = AG_dataset_test.gt_annotations[data[4]]
                frame_names = AG_dataset_test.video_list[data[4]]

                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, frame_names, faset_rcnn_model, transforms)

                if entry != None:
                    pred = model(entry)
                else:
                    pred = {}
                # evaluator.evaluate_scene_graph(gt_annotation, pred)
                evaluator1.evaluate_scene_graph(gt_annotation, dict(pred))
                evaluator2.evaluate_scene_graph(gt_annotation, dict(pred))
                evaluator3.evaluate_scene_graph(gt_annotation, dict(pred))
                t.update(1)
            print('-----------', flush=True)
    score = np.mean(evaluator1.result_dict[conf.mode + "_recall"][20])
    evaluator1.print_stats()

    # save res
    with_res = evaluator1.save_stats()
    semi_res = evaluator2.save_stats()
    no_res = evaluator3.save_stats()
    res = {'with': with_res, 'semi': semi_res, 'no': no_res}
    test_res['epoch' + str(epoch)] = res

    for k in with_res.keys():
        writer.add_scalar('with' + k, with_res[k], epoch)
    for k in semi_res.keys():
        writer.add_scalar('semi' + k, semi_res[k], epoch)
    for k in no_res.keys():
        writer.add_scalar('no' + k, no_res[k], epoch)

    with open(os.path.join(conf.save_path, "save_res_{}.json".format(epoch)), 'w') as f:
        json.dump(test_res, f)

    evaluator1.reset_result()
    evaluator2.reset_result()
    evaluator3.reset_result()
    scheduler.step(score)

    
