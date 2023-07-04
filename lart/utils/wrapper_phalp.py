
import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from phalp.configs.base import CACHE_DIR
from torch import nn

from lart.models.components.lart_transformer.transformer import \
    lart_transformer


class Pose_transformer(nn.Module):
    
    def __init__(self, cfg, phalp_tracker):
        super(Pose_transformer, self).__init__()
        
        self.phalp_cfg = cfg

        # load a config file
        self.cfg = OmegaConf.load(self.phalp_cfg.pose_predictor.config_path).configs
        self.cfg.max_people = 1
        self.encoder      = lart_transformer(   
                                opt         = self.cfg, 
                                dim         = self.cfg.in_feat,
                                depth       = self.cfg.transformer.depth,
                                heads       = self.cfg.transformer.heads,
                                mlp_dim     = self.cfg.transformer.mlp_dim,
                                dim_head    = self.cfg.transformer.dim_head,
                                dropout     = self.cfg.transformer.dropout,
                                emb_dropout = self.cfg.transformer.emb_dropout,
                                droppath    = self.cfg.transformer.droppath,
                                )
        
        self.mean_, self.std_ = np.load(self.phalp_cfg.pose_predictor.mean_std, allow_pickle=True)
        self.mean_            = np.concatenate((self.mean_, np.zeros((1, 229-self.mean_.shape[1]))), axis=1)
        self.std_             = np.concatenate((self.std_, np.ones((1, 229-self.std_.shape[1]))), axis=1)
        self.mean_, self.std_ = torch.tensor(self.mean_), torch.tensor(self.std_)
        self.mean_, self.std_ = self.mean_.float(), self.std_.float()
        self.mean_, self.std_ = self.mean_.unsqueeze(0), self.std_.unsqueeze(0)   
        self.register_buffer('mean', self.mean_)
        self.register_buffer('std', self.std_)
        
        self.smpl = phalp_tracker.HMAR.smpl
            
    def load_weights(self, path):
        checkpoint_file = torch.load(path)
        checkpoint_file_filtered = {k[8:]: v for k, v in checkpoint_file['state_dict'].items()} # remove "encoder." from keys
        self.encoder.load_state_dict(checkpoint_file_filtered, strict=False)
    
    def readout_pose(self, output):
        
        # return predicted gt pose, betas and location
        BS = output.shape[0]
        FL = output.shape[1]
        pose_tokens      = output.contiguous()
        pose_tokens_     = rearrange(pose_tokens, 'b tp dim -> (b tp) dim')
        
        pred_smpl_params = [self.encoder.smpl_head[i](pose_tokens_)[0] for i in range(self.cfg.num_smpl_heads)]
        pred_cam         = [self.encoder.loca_head[i](pose_tokens) for i in range(self.cfg.num_smpl_heads)]
        pred_ava         = [self.encoder.action_head_ava[i](pose_tokens) for i in range(self.cfg.num_smpl_heads)]
        
        pred_cam         = torch.stack(pred_cam, dim=0)[0]
        pred_cam         = rearrange(pred_cam, 'b (t p) dim -> b t p dim', b=BS, t=FL ,p=self.cfg.max_people) # (BS, T, P, 3)        
        

        global_orient    = rearrange(pred_smpl_params[0]['global_orient'], '(b t p) x y z -> b t p (x y z)', b=BS, t=FL ,p=self.cfg.max_people, x=1, y=3, z=3) # (BS, T, P, 9)
        body_pose        = rearrange(pred_smpl_params[0]['body_pose'], '(b t p) x y z -> b t p (x y z)', b=BS, t=FL ,p=self.cfg.max_people, x=23, y=3, z=3) # (BS, T, P, 207)
        betas            = rearrange(pred_smpl_params[0]['betas'], '(b t p) z -> b t p z', b=BS, t=FL ,p=self.cfg.max_people, z=10) # (BS, T, P, 10)
        pose_vector      = torch.cat((global_orient, body_pose, betas, pred_cam), dim=-1) # (BS, T, P, 229)
        
        pred_ava         = torch.stack(pred_ava, dim=0)[0]
        pred_ava         = rearrange(pred_ava, 'b (t p) dim -> b t p dim', b=BS, t=FL ,p=self.cfg.max_people) # (BS, T, P, 60)        

        # TODO: apply moving average for pridictions

        smpl_outputs = {
            'pose_camera'      : pose_vector,
            'camera'           : pred_cam,
            'ava_action'       : pred_ava,
        }
            
        return smpl_outputs
            
    def predict_next(self, en_pose, en_data, en_time, time_to_predict):
        
        """encoder takes keys : 
                    pose_shape (bs, self.cfg.frame_length, 229)
                    has_detection (bs, self.cfg.frame_length, 1), 1 if there is a detection, 0 otherwise
                    mask_detection (bs, self.cfg.frame_length, 1)*0       
        """
        
        # set number of people to one 
        n_p = 1
        pose_shape_ = torch.zeros(en_pose.shape[0], self.cfg.frame_length, n_p, 229)
        has_detection_ = torch.zeros(en_pose.shape[0], self.cfg.frame_length, n_p, 1)
        mask_detection_ = torch.zeros(en_pose.shape[0], self.cfg.frame_length, n_p, 1)
        
        # loop thorugh each person and construct the input data
        t_end = []
        for p_ in range(en_time.shape[0]):
            t_min = en_time[p_, 0].min()
            # loop through time 
            for t_ in range(en_time.shape[1]):
                # get the time from start.
                t = min(en_time[p_, t_] - t_min, self.cfg.frame_length - 1)
                
                # get the pose
                pose_shape_[p_, t, 0, :] = en_pose[p_, t_, :]
                
                # get the mask
                has_detection_[p_, t, 0, :] = 1
            t_end.append(t.item())
            
        input_data = {
            "pose_shape" : (pose_shape_ - self.mean_[:, :, None, :]) / (self.std_[:, :, None, :] + 1e-10),
            "has_detection" : has_detection_,
            "mask_detection" : mask_detection_
        }
        
        # place all the data in cuda
        input_data = {k: v.cuda() for k, v in input_data.items()}

        # single forward pass
        output, _ = self.encoder(input_data, self.cfg.mask_type_test)
        decoded_output = self.readout_pose(output[:, self.cfg.max_people:, :])
        
        assert len(t_end) == len(time_to_predict)
        t_end += time_to_predict + 1
        
        predicted_pose_camera_at_t = []
        for i in range(en_time.shape[0]): 
            t_x = min(t_end[i], self.cfg.frame_length-1)
            predicted_pose_camera_at_t.append(decoded_output['pose_camera'][:, t_x, 0, :])
        predicted_pose_camera_at_t = torch.stack(predicted_pose_camera_at_t, dim=0)[0]
        
        return predicted_pose_camera_at_t
    
    def add_slowfast_features(self, fast_track):
        # add slowfast features to the fast track
        from slowfast.config.defaults import assert_and_infer_cfg
        from slowfast.utils.parser import load_config, parse_args
        from slowfast.visualization.predictor import ActionPredictor, Predictor

        from lart.utils.wrapper_pyslowfast import SlowFastWrapper

        device = 'cuda'
        path_to_config = f"{CACHE_DIR}/phalp/ava/mvit.yaml"
        center_crop = True

        self.cfg.opts = None
        cfg = load_config(self.cfg, path_to_config)
        cfg = assert_and_infer_cfg(cfg)
        cfg.TEST.CHECKPOINT_FILE_PATH=f"{CACHE_DIR}/phalp/ava/mvit.pyth"

        video_model    = Predictor(cfg=cfg, gpu_id=None)
        seq_length     = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

        list_of_frames = fast_track['frame_name']
        list_of_bbox   = fast_track['frame_bbox']
        list_of_fids   = fast_track['fid']
        fast_track['apperance_emb'] = []
        fast_track['action_emb'] = []
        fast_track['apperance_index'] = []
        fast_track['has_gt'] = []

        NUM_STEPS        = 6 # 5Hz
        NUM_FRAMES       = seq_length
        list_iter        = list(range(len(list_of_frames)//NUM_STEPS + 1))

        for t_, time_stamp in enumerate(list_iter):    

            start_      = time_stamp * NUM_STEPS
            end_        = (time_stamp + 1) * NUM_STEPS if (time_stamp + 1) * NUM_STEPS < len(list_of_frames) else len(list_of_frames)
            time_stamp_ = list_of_frames[start_:end_]
            if(len(time_stamp_)==0): continue

            mid_        = (start_ + end_)//2
            mid_frame   = list_of_frames[mid_]
            mid_bbox    = list_of_bbox[mid_]
            mid_fid     = list_of_fids[mid_]

            list_of_all_frames = []
            for i in range(-NUM_FRAMES//2,NUM_FRAMES//2 + 1):
                if(mid_ + i < 0):
                    frame_id = 0
                elif(mid_ + i >= len(list_of_frames)):
                    frame_id = len(list_of_frames) - 1
                else:
                    frame_id = mid_ + i
                list_of_all_frames.append(list_of_frames[frame_id])


            mid_bbox_   = mid_bbox.reshape(1, 4).astype(np.int32)
            mid_bbox_   = np.concatenate([mid_bbox_[:, :2], mid_bbox_[:, :2] + mid_bbox_[:, 2:4]], 1)
            # img1 = cv2.imread(mid_frame)
            # img1 = cv2.rectangle(img1, (mid_bbox_[0, 0], mid_bbox_[0, 1]), (mid_bbox_[0, 2], mid_bbox_[0, 3]), (0, 255, 0), 2)
            # cv2.imwrite("test.png", img1)
            with torch.no_grad():
                task_      = SlowFastWrapper(t_, cfg, list_of_all_frames, mid_bbox_, video_model, center_crop=center_crop)
                preds      = task_.action_preds[0]
                feats      = task_.action_preds[1]
                preds      = preds.cpu().numpy()
                feats      = feats.cpu().numpy()

            for frame_ in time_stamp_:
                fast_track['apperance_emb'].append(feats)
                fast_track['action_emb'].append(preds)
                fast_track['apperance_index'].append(t_*1000)     
                fast_track['has_gt'].append(1)     
        
        assert len(fast_track['apperance_emb']) == len(fast_track['frame_name'])
        fast_track['apperance_emb'] = np.array(fast_track['apperance_emb'])
        fast_track['action_emb'] = np.array(fast_track['action_emb'])
        fast_track['apperance_index'] = np.array(fast_track['apperance_index']).reshape(-1, 1, 1)
        fast_track['has_gt'] = np.array(fast_track['has_gt']).reshape(-1, 1, 1)
        
        return fast_track

    def smooth_tracks(self, fast_track, moving_window=False, step=1, window=20):
        
        if("apperance" in self.cfg.extra_feat.enable):
            fast_track = self.add_slowfast_features(fast_track)

        # set number of people to one 
        n_p = 1
        fl  = fast_track['pose_shape'].shape[0]

        pose_shape_all = torch.zeros(1, fl, n_p, 229)
        has_detection_all = torch.zeros(1, fl, n_p, 1)
        mask_detection_all = torch.zeros(1, fl, n_p, 1)

        if("apperance" in self.cfg.extra_feat.enable):
            apperance_feat_all = fast_track['apperance_emb'][None, :, :,]
        
        if("joints_3D" in self.cfg.extra_feat.enable):
            joints_ = fast_track['3d_joints'][:, :, :, :]
            camera_ = fast_track['camera'][:, None, :, :]
            joints_3d_all = joints_ + camera_
            joints_3d_all = joints_3d_all.reshape(1, fl, n_p, 135)

        for t_ in range(fast_track['pose_shape'].shape[0]):
            pose_shape_all[0, t_, 0, :] = torch.tensor(fast_track['pose_shape'][t_])
            has_detection_all[0, t_, 0, :] = 1
            mask_detection_all[0, t_, 0, :] = 1.0 - torch.tensor(fast_track['has_detection'][t_, 0])

        S_ = 0
        STEP_ = step
        WINDOW_ = window
        w_steps = range(S_, S_+fl, STEP_)
        assert 2*WINDOW_ + STEP_ < self.cfg.frame_length
        STORE_OUTPUT_ = torch.zeros(1, fl, self.cfg.in_feat)

        for w_ in w_steps:

            pose_shape_ = torch.zeros(1, self.cfg.frame_length, n_p, 229)
            has_detection_ = torch.zeros(1, self.cfg.frame_length, n_p, 1)
            mask_detection_ = torch.zeros(1, self.cfg.frame_length, n_p, 1)

            start_ = w_ - WINDOW_ if (w_ - WINDOW_>0) else 0
            end_ = w_ + STEP_ + WINDOW_ if (w_ + STEP_ + WINDOW_<=fl) else fl

            pose_shape_[:, :end_-start_, :, :] = pose_shape_all[:, start_:end_, :, :]
            has_detection_[:, :end_-start_, :, :] = has_detection_all[:, start_:end_, :, :]
            mask_detection_[:, :end_-start_, :, :] = mask_detection_all[:, start_:end_, :, :]

            input_data = {
                "pose_shape" : (pose_shape_ - self.mean_[0, :, None, :]) / (self.std_[0, :, None, :] + 1e-10),
                "has_detection" : has_detection_,
                "mask_detection" : mask_detection_
            }
            
            # add other features if enables:
            if("joints_3D" in self.cfg.extra_feat.enable):
                joints_ = torch.zeros(1, self.cfg.frame_length, n_p, 135)
                joints_[:, :end_-start_, :, :] = torch.tensor(joints_3d_all[:, start_:end_, :, :])
                input_data["joints_3D"] = joints_

            if("apperance" in self.cfg.extra_feat.enable):
                apperance_ = torch.zeros(1, self.cfg.frame_length, n_p, self.cfg.extra_feat.apperance.dim)
                apperance_[:, :end_-start_, :, :] = torch.tensor(apperance_feat_all[:, start_:end_, :, :])
                input_data["apperance_emb"] = apperance_

            input_data = {k: v.cuda() for k, v in input_data.items()}

            output, _ = self.encoder(input_data, self.cfg.mask_type_test)
            output = output[:, self.cfg.max_people:, :]

            
            if(w_+STEP_<fl):
                if(w_<=WINDOW_):
                    STORE_OUTPUT_[:,  w_:w_+STEP_, :] = output[:,  w_:w_+STEP_, :]
                else:
                    STORE_OUTPUT_[:,  w_:w_+STEP_, :] = output[:,  WINDOW_:WINDOW_+STEP_, :]
            else:
                if(w_<=WINDOW_):
                    STORE_OUTPUT_[:,  w_:fl, :] = output[:,  w_:fl, :]
                else:
                    STORE_OUTPUT_[:,  w_:fl, :] = output[:,  WINDOW_:WINDOW_+(fl-w_), :]

        decoded_output = self.readout_pose(STORE_OUTPUT_.cuda())

        fast_track['pose_shape'] = decoded_output['pose_camera'][0, :fast_track['pose_shape'].shape[0], :, :]
        fast_track['cam_smoothed'] = decoded_output['camera'][0, :fast_track['pose_shape'].shape[0], :, :]
        fast_track['ava_action'] = decoded_output['ava_action'][0, :fast_track['pose_shape'].shape[0], :, :]
        
        return fast_track