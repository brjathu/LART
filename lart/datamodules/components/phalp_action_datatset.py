import os

import joblib
import numpy as np
import torch
import torchvision.transforms as transforms
from phalp.configs.base import CACHE_DIR
from PIL import Image
from torch.utils.data import Dataset

from lart.utils import get_pylogger
from lart.utils.utils import task_divider

log = get_pylogger(__name__)

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
        
class PHALP_action_dataset(Dataset):
    def __init__(self, opt, train=True):
        
        self.opt              = opt
        self.data             = []
        self.track2video      = []
        self.frame_length     = self.opt.frame_length
        self.max_tokens       = self.opt.max_people
        self.train            = train
        self.mean_, self.std_ = np.load(f"{CACHE_DIR}/phalp/3D/mean_std.npy")
        self.mean_pose_shape  = np.concatenate((self.mean_, np.zeros((1, 229-self.mean_.shape[1]))), axis=1)
        self.std_pose_shape   = np.concatenate((self.std_, np.ones((1, 229-self.std_.shape[1]))), axis=1)
        
        # temp arguments
        self.opt.img_size     = 256
        self.pixel_mean_      = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape(3,1,1)
        self.pixel_std_       = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape(3,1,1)

        self.dataset_roots        = {
            # heira
            "kinetics_train"  : ["",    "data/kinetics_train/", self.opt.kinetics.sampling_factor],
            "ava_train"       : ["ava", "data/ava_train/",      self.opt.ava.sampling_factor],
            "ava_val"         : ["ava", "data/ava_val/",        1], 
        }
        
        if(self.train):
            
            self.list_of_datasets = opt.train_dataset.split(",")
            for dataset in self.list_of_datasets:
                log.info(self.dataset_roots[dataset][0])
                self.get_dataset(root_dir=self.dataset_roots[dataset][1], 
                                filter_seq=self.dataset_roots[dataset][0],
                                num_sample=self.dataset_roots[dataset][2], 
                                min_track_length=1, 
                                total_num_tracks=None)   
                
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])     

        else:
            self.list_of_datasets = opt.test_dataset.split(",")
            for dataset in self.list_of_datasets:
                self.get_dataset(root_dir=self.dataset_roots[dataset][1], 
                                filter_seq=self.opt.test_class, 
                                min_track_length=1, 
                                total_num_tracks=None)
            
            self.data = np.array(self.data)
            self.track2video = np.array(self.track2video)
            
            self.data = task_divider(self.data, self.opt.test_batch_id, self.opt.number_of_processes)
            self.track2video = task_divider(self.track2video, self.opt.test_batch_id, self.opt.number_of_processes)

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        self.ava_valid_classes = np.load(f"{CACHE_DIR}/lart/ava_valid_classes.npy")
        self.kinetics_annotations = joblib.load(f"{CACHE_DIR}/lart/kinetics_annot_train.pkl")
        log.info("Number of tracks: {}".format(len(self.data)))
            
        self.pose_key = "pose_shape"
        self.ego_id = 0


    def __len__(self):
        return len(self.data)

    def get_dataset(self, root_dir="", filter_seq="posetrack-train", num_sample=1, min_track_length=20, total_num_tracks=None):
        count    = 0
        count_f  = 0
        path_npy = "data/_TMP/"+"fast_".join(root_dir.split("/"))+".npy"
        
        # to store all the files in a list
        if(os.path.exists(path_npy)):
            list_of_files = np.load(path_npy)
        else:
            list_of_files = os.listdir(root_dir)
            np.save(path_npy, list_of_files)
            
        # to store all the tracks from same video in a dictionary
        list_of_t2v = {}
        list_of_v2t = {}
        for file_i in list_of_files:
            video_name = "_".join(file_i.split("_")[:-2])
            list_of_v2t.setdefault(video_name, []).append(file_i)
        for video_name in list_of_v2t.keys():
            all_tracks = list_of_v2t[video_name]
            for track_i in all_tracks:
                list_of_t2v[track_i] = [os.path.join(root_dir, i) for i in all_tracks if i!=track_i]
            
        for i_, video_ in enumerate(list_of_files):
            if(video_.endswith(".pkl") and filter_seq in video_):
                if(int(video_.split("_")[-1][:-4])>min_track_length):
                    for _ in range(num_sample): 
                        self.data.append(os.path.join(root_dir, video_))
                        self.track2video.append(list_of_t2v[video_])
                    count += 1
                    count_f += int(video_.split("_")[-1][:-4])
                    if(total_num_tracks is not None):
                        if(count>=total_num_tracks):
                            break
        log.info("Total number of tracks: {}".format(count))
        log.info("Total number of frames: {}".format(count_f))
    
    def get_start_end_frame(self, list_of_frames, f_):

        if(self.train):
            start_frame  = np.random.choice(len(list_of_frames)-(self.frame_length+f_), 1)[0] if(len(list_of_frames)>self.frame_length+f_) else 0
            end_frame    = start_frame + self.frame_length if(len(list_of_frames)>self.frame_length+f_) else  len(list_of_frames)-f_
            key_frame    = (start_frame+end_frame)//2
        else:
            start_frame  = 0
            end_frame    = len(list_of_frames) if(self.opt.full_seq_render) else min(len(list_of_frames), self.frame_length)
            key_frame    = (start_frame+end_frame)//2

        return start_frame, end_frame, key_frame

    def read_from_phalp_fast(self, idx):
        try:
            detection_data   = joblib.load(self.data[idx])
        except:
            np.save("data/bad_files/" + self.data[idx].split("/")[-1].split(".")[0] + ".npy", [self.data[idx]])
            detection_data   = joblib.load(self.data[0])
            
        list_of_frames   = list(range(len(detection_data["frame_name"])))
        if(self.opt.frame_rate_range>1 and self.train):
            frame_rate     = np.random.randint(1, self.opt.frame_rate_range)
            list_of_frames = list_of_frames[::frame_rate]
        return detection_data, list_of_frames, self.data[idx].split("/")[-1][:-4]
    
    def read_from_phalp_other(self, other_track):
        try:
            detection_data   = joblib.load(other_track)
        except:
            detection_data   = joblib.load(self.data[0])
        return detection_data

    def initiate_dict(self, frame_length, f_):
        
        input_data = { 
            'pose_shape'            : np.zeros((frame_length, self.max_tokens, 229))*0.0,
            'relative_pose'         : np.zeros((frame_length, self.max_tokens, 16))*0.0,
            'has_detection'         : np.zeros((frame_length, self.max_tokens, 1))*0.0,
            'mask_detection'        : np.zeros((frame_length, self.max_tokens, 1))*0.0,
            'fid'                   : np.zeros((frame_length, self.max_tokens, 1))*0.0,
        }

        output_data = {
            'pose_shape'            : np.zeros((frame_length, self.max_tokens, f_, 229))*0.0,
            'action_label_ava'      : np.zeros((frame_length, self.max_tokens, f_, 80))*0.0,
            'action_label_kinetics' : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0,
            'has_detection'         : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0,
            'has_gt'                : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0,
            'has_gt_kinetics'       : np.zeros((frame_length, self.max_tokens, f_, 1))*0.0
        }

        meta_data = {
            'frame_name'            : [],
            'frame_bbox'            : [],
            'frame_size'            : [],
            'frame_conf'            : [],
        }
        
        if("apperance" in self.opt.extra_feat.enable):
            input_data['apperance_emb'] = np.zeros((frame_length, self.max_tokens, self.opt.extra_feat.apperance.dim))*0.0
            
        if("joints_3D" in self.opt.extra_feat.enable):
            input_data['joints_3D'] = np.zeros((frame_length, self.max_tokens, 135))*0.0
        
        return input_data, output_data, meta_data
    
    def __getitem__(self, idx):

        f_ = self.opt.num_smpl_heads
        detection_data, list_of_frames, video_name = self.read_from_phalp_fast(idx)
        start_frame, end_frame, _   = self.get_start_end_frame(list_of_frames, f_)
        
        if(self.train): frame_length_ = self.opt.frame_length
        else:           frame_length_ = max(end_frame - start_frame, self.opt.frame_length)
       
        input_data, output_data, meta_data = self.initiate_dict(frame_length_, f_)
        
        # for n>1 setting, read all other tracks.
        other_tracks = []
        if(self.max_tokens>1):
            tracks_ = self.track2video[idx]
            tracks_tmp = tracks_.copy()
            np.random.shuffle(tracks_tmp)
            for i in range(min(self.max_tokens-1, len(tracks_tmp))):
                other_tracks.append(self.read_from_phalp_other(tracks_tmp[i]))
                
        delta = 0
        if(end_frame>frame_length_):
            end_frame = end_frame - start_frame
            start_frame = 0
    
    
        input_data['pose_shape'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]            = (detection_data[self.pose_key][start_frame:end_frame].copy() - self.mean_pose_shape[None, :, :])/(self.std_pose_shape[None, :, :] + 1e-10)
        input_data['has_detection'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]         = detection_data["has_detection"][start_frame:end_frame]
        input_data['fid'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]                   = detection_data["fid"][start_frame:end_frame]
        
        output_data['pose_shape'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :]        = detection_data[self.pose_key][start_frame:end_frame]
        output_data['has_detection'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :]     = detection_data["has_detection"][start_frame:end_frame]
        output_data['has_gt'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :]            = detection_data["has_gt"][start_frame:end_frame]
        
        # add kinetics labels
        if("kinetics" in self.opt.action_space and not("ava" in video_name)):
            class_label = self.kinetics_annotations[video_name.split("kinetics-train_")[1][:11]]
            output_data['has_gt_kinetics'][:, :, 0, :] = 1.0
            output_data['action_label_kinetics'][:, :, 0, :] = class_label[1]
        
        appe_idx = detection_data['apperance_index'][start_frame:end_frame][:,0,0]
        appe_feat = detection_data['apperance_dict']
        ava_pseudo_labels = detection_data['action_label_psudo']
        ava_gt_labels = detection_data['action_label_gt'][start_frame:end_frame]

        ava_pseudo_labels_ = np.zeros((end_frame-start_frame, 1, 80))
        for i in range(ava_pseudo_labels_.shape[0]):
            if(appe_idx[i]!=-1):
                ava_pseudo_labels_[i, 0, :] = ava_pseudo_labels[appe_idx[i]][0]
        
        ava_pseudo_vectors_ = np.zeros((end_frame-start_frame, 1, self.opt.extra_feat.apperance.dim))
        for i in range(ava_pseudo_vectors_.shape[0]):
            if(appe_idx[i]!=-1):
                ava_pseudo_vectors_[i, 0, :] = appe_feat[appe_idx[i]][0]

        has_gt_array = detection_data['has_gt'][start_frame:end_frame, 0, 0].copy()
        ava_pseudo_labels_[has_gt_array==2] = ava_gt_labels[has_gt_array==2]

        TMP_ = ava_pseudo_labels_.copy()

        if(self.opt.ava.predict_valid):
            action_label_ava_ = np.zeros((end_frame-start_frame, 1, self.opt.ava.num_action_classes))
            action_label_ava_[:, :, :self.opt.ava.num_valid_action_classes] = TMP_[:, :, self.ava_valid_classes-1]
            output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :] = action_label_ava_.copy()
        else:
            action_label_ava_ = np.zeros((end_frame-start_frame, 1, self.opt.ava.num_action_classes))
            action_label_ava_[:, :, :] = TMP_[:, :, :]
            output_data['action_label_ava'][start_frame:end_frame, self.ego_id:self.ego_id+1, 0, :]  = action_label_ava_.copy()
        
        # extra features.
        if("apperance_emb" in input_data.keys()):
            input_data['apperance_emb'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]              = ava_pseudo_vectors_

        if("joints_3D" in input_data.keys()):
            joints_ = detection_data['3d_joints'][:, :, :, :][start_frame:end_frame]
            camera_ = detection_data['camera'][:, None, :, :][start_frame:end_frame]
            joints_ = joints_ + camera_
            input_data['joints_3D'][start_frame:end_frame, self.ego_id:self.ego_id+1, :]              = joints_.reshape(end_frame-start_frame, 1, 135)
        
            
        if(self.max_tokens>1):
            # for n>1 setting, read all other tracks.
            base_idx = detection_data['fid'][start_frame:end_frame]
            for ot in range(len(other_tracks)):
                other_detection_data = other_tracks[ot]
                other_base_idx = other_detection_data['fid']

                if(other_base_idx[0]>base_idx[-1]): continue
                elif(other_base_idx[-1]<base_idx[0]): continue
                elif(other_base_idx[0]>=base_idx[0] and other_base_idx[-1, 0, 0]<=base_idx[-1, 0, 0]):
                    other_start_frame = 0
                    other_end_frame = len(other_base_idx)
                    delta = other_base_idx[0, 0, 0] - base_idx[0, 0, 0]
                elif(other_base_idx[0, 0, 0]>=base_idx[0, 0, 0] and other_base_idx[-1, 0, 0]>base_idx[-1, 0, 0]):
                    other_start_frame = 0
                    other_end_frame = len(other_base_idx) - (other_base_idx[-1, 0, 0] - base_idx[-1, 0, 0])
                    delta = other_base_idx[0, 0, 0] - base_idx[0, 0, 0]
                elif(other_base_idx[0, 0, 0]<base_idx[0, 0, 0] and other_base_idx[-1, 0, 0]<=base_idx[-1, 0, 0]):
                    other_start_frame = base_idx[0, 0, 0] - other_base_idx[0, 0, 0]
                    other_end_frame = len(other_base_idx)
                    delta = -other_start_frame
                else:
                    continue
                
                other_start_frame = int(other_start_frame)
                other_end_frame = int(other_end_frame)
                delta = int(delta)
                
                input_data['pose_shape'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]            = (other_detection_data[self.pose_key][other_start_frame:other_end_frame].copy() - self.mean_pose_shape[None, :, :])/(self.std_pose_shape[None, :, :] + 1e-10)
                input_data['has_detection'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]         = other_detection_data["has_detection"][other_start_frame:other_end_frame]
                input_data['fid'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]                   = other_detection_data["fid"][other_start_frame:other_end_frame]
                
                output_data['pose_shape'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :]        = other_detection_data[self.pose_key][other_start_frame:other_end_frame]
                output_data['has_detection'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :]     = other_detection_data["has_detection"][other_start_frame:other_end_frame]
                if(self.opt.loss_on_others_action):
                    output_data['has_gt'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :]            = other_detection_data["has_gt"][other_start_frame:other_end_frame]
                

                other_appe_idx = other_detection_data['apperance_index'][other_start_frame:other_end_frame][:,0,0]
                other_appe_feat = other_detection_data['apperance_dict']
                other_ava_pseudo_labels = other_detection_data['action_label_psudo']
                other_ava_gt_labels = other_detection_data['action_label_gt'][other_start_frame:other_end_frame]

                other_ava_pseudo_labels_ = np.zeros((other_end_frame-other_start_frame, 1, 80))
                for i in range(other_ava_pseudo_labels_.shape[0]):
                    if(other_appe_idx[i]!=-1):
                        other_ava_pseudo_labels_[i, 0, :] = other_ava_pseudo_labels[other_appe_idx[i]][0]
                
                other_ava_pseudo_vectors_ = np.zeros((other_end_frame-other_start_frame, 1, self.opt.extra_feat.apperance.dim))
                for i in range(other_ava_pseudo_vectors_.shape[0]):
                    if(other_appe_idx[i]!=-1):
                        other_ava_pseudo_vectors_[i, 0, :] = other_appe_feat[other_appe_idx[i]][0]

                other_has_gt_array = other_detection_data['has_gt'][other_start_frame:other_end_frame, 0, 0].copy()
                other_ava_pseudo_labels_[other_has_gt_array==2] = other_ava_gt_labels[other_has_gt_array==2]


                if(self.opt.ava.predict_valid):
                    action_label_ava_ = np.zeros((other_end_frame-other_start_frame, 1, self.opt.ava.num_action_classes))
                    TMP_ = other_ava_pseudo_labels_.copy()
                    action_label_ava_[:, :, :self.opt.ava.num_valid_action_classes] = TMP_[:, :, self.ava_valid_classes-1]
                    output_data['action_label_ava'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :] = action_label_ava_.copy()
                else:
                    output_data['action_label_ava'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, 0, :] = other_detection_data["action_label_ava"][other_start_frame:other_end_frame]
                
                # extra features.
                if("apperance_emb" in input_data.keys()):
                    input_data['apperance_emb'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]             = other_ava_pseudo_vectors_
                
                if("joints_3D" in input_data.keys()):
                    joints_ = other_detection_data['3d_joints'][:, :, :, :][other_start_frame:other_end_frame]
                    camera_ = other_detection_data['camera'][:, None, :, :][other_start_frame:other_end_frame]
                    joints_ = joints_ + camera_
                    if(not(self.opt.loss_on_others_action)):
                        input_data['joints_3D'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]              = joints_.reshape(other_end_frame-other_start_frame, 1, 135)
                        input_data['joints_3D'][:, ot+1:ot+2, :] -= input_data['joints_3D'][:, self.ego_id:self.ego_id+1, :]
                    else:
                        input_data['joints_3D'][delta+other_start_frame:delta+other_end_frame, ot+1:ot+2, :]              = joints_.reshape(other_end_frame-other_start_frame, 1, 135)
                del other_detection_data, other_base_idx, action_label_ava_
        
        if(not(self.train)):
            # add meta data for rendering
            meta_data['frame_name'] = detection_data["frame_name"][start_frame:end_frame].copy()
            meta_data['frame_size'] = detection_data["frame_size"][start_frame:end_frame].copy()
            meta_data['frame_bbox'] = detection_data["frame_bbox"][start_frame:end_frame].copy()
            meta_data['frame_conf'] = detection_data["frame_conf"][start_frame:end_frame].copy()
            
            if(end_frame-start_frame<frame_length_):
                for i in range((frame_length_)-(end_frame-start_frame)):
                    meta_data['frame_name'].append("-1")
                    meta_data['frame_size'].append(np.array([0, 0]))
                    meta_data['frame_bbox'].append(np.array([0.0, 0.0, 0.0, 0.0]))
                    meta_data['frame_conf'].append(0)
                    
        del detection_data
        
        return input_data, output_data, meta_data, video_name

