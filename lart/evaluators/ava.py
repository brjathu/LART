import csv
import os

import joblib
import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from phalp.configs.base import CACHE_DIR
from lart.ActivityNet.Evaluation.get_ava_performance import run_evaluation
from lart.utils import get_pylogger

log = get_pylogger(__name__)


class AVA_evaluator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.ava_valid_classes = joblib.load(f"{CACHE_DIR}/phalp/ava/ava_class_mapping.pkl")
        self.ava_valid_classes_inv = {v: k for k, v in self.ava_valid_classes.items()}  
    
    def compute_map(self, epoch):
        pbtext = open(f"{CACHE_DIR}/lart/ava_action_list_v2.2_for_activitynet_2019.pbtxt", "r")
        if(self.cfg.ava.map_on == "AVA"):
            gt_annotations = open(f"{CACHE_DIR}/lart/ava_val_v2.2.csv", "r")
        # elif(self.cfg.ava.map_on == "AVA-AK"):
        #     gt_annotations = open("data/kinetics/AVA-AK_val.csv", "r")
        # elif(self.cfg.ava.map_on == "AVA-AK2"):
        #     gt_annotations = open("data/kinetics/AVA-AK_val_2.csv", "r")
        else:
            raise ValueError("map_on should be either AVA or AVA-AK")
        
        pred_labels = open( self.cfg.storage_folder + "/ava_val.csv", "r")
        map_values = run_evaluation(pbtext, gt_annotations, pred_labels)
        log.info("mAP : " + str(map_values[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0))
        print("mAP : " + str(map_values[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0))
        joblib.dump(map_values, self.cfg.storage_folder + "/results/" + str(epoch) + ".pkl")

        return map_values 
    
    def write_ava_csv(self, ava_results_path, csv_path):
        AVA_VALID_FRAMES = range(902, 1799)
        log.info("Start reading predictions.")

        slowfast_files = [i for i in os.listdir(ava_results_path) if (i.endswith(".pkl") and not(i.startswith("eval_pkl_dict_")))]    
        slowfast_pkl_files = joblib.Parallel(n_jobs=8, timeout=9999)(joblib.delayed(joblib.load)(ava_results_path + path) for path in slowfast_files)
                
        label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map(f'{CACHE_DIR}/lart/ava_action_list.pbtxt')
        f = open(csv_path, 'w')
        writer = csv.writer(f)
        counter = 0

        list_of_name = slowfast_files
        list_of_data = slowfast_pkl_files
       
        for sl_, slowfast_file in enumerate(list_of_name):
            
            if("ava-val_" in slowfast_file):
                video_id  = slowfast_file.split("ava-val_")[1][:11]
                key_frame = slowfast_file.split("ava-val_")[1][12:18]
                frame_id  = "%04d"%(int(key_frame)//30 + 900,)
                if(int(key_frame)//30+900 not in AVA_VALID_FRAMES): 
                    continue
            elif("AK-val_" in slowfast_file):
                video_id  = slowfast_file.split("AK-val_")[1][:11]
                key_frame = slowfast_file.split("AK-val_")[1][12:18]
                frame_id  = str(key_frame)
            else:
                raise ValueError("Unknown file name.")
            
            data      = list_of_data[sl_]
            h, w = data[-2][0][0], data[-2][0][1]
            det_conf_ = data[-1]
            
            if(det_conf_ < 0.80 and "ava-val_" in slowfast_file): 
                continue
            if(det_conf_ < 0.90 and "AK-val_" in slowfast_file): 
                continue
            
            for i in range(len(data[2])):
                x1, y1, x2, y2 = data[-3][i][0], data[-3][i][1], data[-3][i][2], data[-3][i][3]
                pred  = data[1][i]
                pred_ = np.argsort(pred)[::-1]
                conf  = pred[pred_]
                loc_  = conf>-1
                pred_ = pred_[loc_]
                conf  = conf[loc_]

                for j in range(len(pred_)):
                    if(len(pred_)==self.cfg.ava.num_valid_action_classes+1):
                        pred_class = self.ava_valid_classes[pred_[j]]
                    else:
                        pred_class = pred_[j]
                    if(pred_class!=0 and pred_class in allowed_class_ids):
                        result = [video_id, frame_id, x1/w, y1/h, x2/w, y2/h, pred_class, conf[j]]
                        writer.writerow(result)
                counter += 1
        log.info("number of bbox detected : " + str(counter))
        f.close()
        
        del slowfast_pkl_files


    def process_meta_data(self, meta_data):

        frame_name_array = np.array(meta_data['frame_name']).transpose(1, 0)
        frame_size_array = []
        frame_conf_array = []
        frame_bbox_array = []
        for i in range(len(meta_data['frame_size'])):
            frame_size_array.append(meta_data['frame_size'][i])
            frame_conf_array.append(meta_data['frame_conf'][i])
            frame_bbox_array.append(meta_data['frame_bbox'][i])
        frame_size_array  = torch.stack(frame_size_array).permute(1, 0, 2)
        frame_conf_array  = torch.stack(frame_conf_array).permute(1, 0)
        frame_bbox_array  = torch.stack(frame_bbox_array).permute(1, 0, 2)

        return frame_name_array, frame_size_array, frame_conf_array, frame_bbox_array


    def store_results_batch(self, input_data, output_data, meta_data, smpl_output, video_name, save_path, output=None):
        frame_name_array, frame_size_array, frame_conf_array, frame_bbox_array = self.process_meta_data(meta_data)
        BS, T, P, _ = input_data['has_detection'].shape
        pred_action = smpl_output['pred_actions_ava'].view(BS, T, P, smpl_output['pred_actions_ava'].shape[-1])

        for bid in range(len(video_name)):
            if("ava-val_" in video_name[bid]):
                video_id     = video_name[bid].split("ava-val_")[1][:11]
                key_frame    = video_name[bid].split("ava-val_")[1].split(".jpg")[0][12:].split("_")[0]
                frame_id     = "%04d"%(int(key_frame)//30 + 900,)
            elif("AK-val_" in video_name[bid]):
                video_id     = video_name[bid].split("AK-val_")[1][:11]
                key_frame    = video_name[bid].split("AK-val_")[1].split(".jpg")[0][12:].split("_")[0]
                frame_id     = key_frame
            else:
                raise NotImplementedError
            
            kfid         = None
            max_length   = 0
            for fid, frame_name_ in enumerate(frame_name_array[bid]):
                if(frame_name_=="-1"):
                    break
                else:
                    max_length+=1
                    if(key_frame in frame_name_): 
                        kfid = fid

            if(kfid is not None):
                if(input_data['has_detection'][bid][kfid][0]):
                    size        = frame_size_array[bid][kfid]
                    conf        = frame_conf_array[bid][kfid]
                    

                    if("GT" in self.cfg.test_type.split("@")[1]):
                        if(self.cfg.ava.predict_valid):
                            preds     = output_data['action_label_ava'][bid, kfid, 0, 0, :self.cfg.ava.num_valid_action_classes].cpu().view(1, -1)
                        else:
                            preds     = output_data['action_label_ava'][bid, kfid, 0, 0, :].cpu().view(1, -1)
                        preds     = torch.cat([torch.zeros(preds.shape[0],1), preds], dim=1)  
                        
                    elif("avg" in self.cfg.test_type.split("@")[1]):
                        w_x       = int(self.cfg.test_type.split("avg.")[1].split(".")[0])
                        s_        = kfid - w_x if kfid-w_x>0 else 0
                        e_        = kfid + w_x if kfid+w_x<max_length else max_length
                        pred      = torch.sigmoid(pred_action[bid, s_:e_, 0, :].cpu())
                        pred      = pred.mean(0, keepdim=True)
                        preds     = torch.cat([torch.zeros(pred.shape[0],1), pred], dim=1)

                    else:
                        pred      = torch.sigmoid(pred_action[bid, kfid, 0, :].cpu().view(1, -1))
                        preds     = torch.cat([torch.zeros(pred.shape[0],1), pred], dim=1)

                    bbox      = frame_bbox_array[bid][kfid]
                    bbox_     = torch.cat((bbox[:2], bbox[:2] + bbox[2:]))
                    bbox_2    = np.array([i.item() for i in bbox_])
                    bbox_norm = [bbox_[0].item()/size[1].item(), bbox_[1].item()/size[0].item(), bbox_[2].item()/size[1].item(), bbox_[3].item()/size[0].item()]
                    bbox_norm = np.array(bbox_norm).reshape(1, 4)

                    result = [video_name[bid], np.array(preds), bbox_2.reshape(1, 4), np.array([size[0].item(), size[1].item()]).reshape(1, 2), conf.cpu()]
                    
                    joblib.dump(result, save_path[bid])
                    
                    del result