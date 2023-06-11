import os
import argparse
import json
from lib.config import conf, cfg_from_file

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/demo.yml', type=str)
args = parser.parse_args()
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
print('The results saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print(conf)

os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.gpu_id)


import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
from tqdm import tqdm
torch.set_num_threads(2)

from dataloader.action_genome import AG, cuda_collate_fn

from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.sttran import STTran
from lib.assign_pseudo_label import prepare_func


AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, ws_object_bbox_path=None, remove_one_frame_video=True,
                     filter_nonperson_box_frame=True, filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=conf.num_workers, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, conf=conf).to(device=gpu_device)
object_detector.eval()

faset_rcnn_model, transforms = prepare_func()

model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               spatial_class_num=len(AG_dataset.spatial_relationships),
               contact_class_num=len(AG_dataset.contacting_relationships),
               obj_classes=AG_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               transformer_mode=conf.transformer_mode,
               is_wks=conf.is_wks,
               feat_dim=conf.feat_dim).to(device=gpu_device)

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))
#
evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)

evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='no')

with torch.no_grad():
    with tqdm(total=len(dataloader)) as t:
        for b, data in enumerate(dataloader):

            im_data = copy.deepcopy(data[0].cuda())
            im_info = copy.deepcopy(data[1].cuda())
            gt_boxes = copy.deepcopy(data[2].cuda())
            num_boxes = copy.deepcopy(data[3].cuda())
            gt_annotation = AG_dataset.gt_annotations[data[4]]
            frame_names = AG_dataset.video_list[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, frame_names, faset_rcnn_model, transforms)

            if entry != None: 
                pred = model(entry)
            else:
                pred = {}
            evaluator1.evaluate_scene_graph(gt_annotation, dict(pred))
            evaluator2.evaluate_scene_graph(gt_annotation, dict(pred))
            evaluator3.evaluate_scene_graph(gt_annotation, dict(pred))

            # t.set_description(desc="Epoch %i"%i)
            # t.set_postfix(steps=step, loss=loss.data.item())
            t.update(1)


print('-------------------------with constraint-------------------------------')
evaluator1.print_stats()
print('-------------------------semi constraint-------------------------------')
evaluator2.print_stats()
print('-------------------------no constraint-------------------------------')
evaluator3.print_stats()

# save res
with_res = evaluator1.save_stats()
semi_res = evaluator2.save_stats()
no_res = evaluator3.save_stats()
res = {'with': with_res, 'semi': semi_res, 'no': no_res}
test_res = {}
test_res[conf.model_path] = res

with open(os.path.join(conf.save_path, "save_res_4_{}.json".format(conf.mode)), 'w') as f:
    json.dump(test_res, f)
