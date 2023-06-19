import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def compute_loss(opt, output, smpl_output, output_data, input_data, train=True):
    loss_dict      = {
        "pose"   : 0, 
        "loca"   : 0, 
        "action" : 0, 
    }

    for fi in range(opt.num_smpl_heads):
        # groundtruth data
        gt_pose_shape        = output_data['pose_shape'][:, :, :, fi, :226]
        gt_location          = output_data['pose_shape'][:, :, :, fi, 226:229]
        gt_action_kinetics   = output_data['action_label_kinetics'][:, :, :, fi, :].float()
        gt_has_detection     = output_data['has_detection'][:, :, :, fi, :]
        if(opt.ava.predict_valid):
            gt_action_ava    = output_data['action_label_ava'][:, :, :, fi, :opt.ava.num_valid_action_classes].float()
        else:
            gt_action_ava    = output_data['action_label_ava'][:, :, :, fi, :].float()
  
        # predicted data  
        BS, T, P, _          = gt_pose_shape.shape
        masked_detection     = input_data['mask_detection']

        pred_global_orient   = smpl_output['pred_smpl_params'][fi]['global_orient'].view(BS, T, P, 9)
        pred_body_pose       = smpl_output['pred_smpl_params'][fi]['body_pose'].view(BS, T, P, 207)
        pred_betas           = smpl_output['pred_smpl_params'][fi]['betas'].view(BS, T, P, 10)
        pred_pose_shape      = torch.cat((pred_global_orient, pred_body_pose, pred_betas), dim=3)
        pred_location        = smpl_output['cam_t'][:, fi, :].view(BS, T, P, 3)
        pred_action_ava      = smpl_output['pred_actions_ava'][:, fi, :].view(BS, T, P, -1)
        pred_action_kinetics = smpl_output['pred_actions_kinetics']
            
        if(train): 
            if(opt.masked):
                loca_loss = torch.logical_and(gt_has_detection==1, masked_detection==1)
            else:
                loca_loss = gt_has_detection==1
        else:      
            loca_loss = gt_has_detection==1
        loca_loss = loca_loss[:, :, :, 0]

        loss_pose   = torch.tensor(0.0).cuda()
        loss_loca   = torch.tensor(0.0).cuda()
        loss_action = torch.tensor(0.0).cuda()

        if("pose_l1" in opt.loss_type):
            loss_pose   = ( pred_pose_shape[loca_loss] - gt_pose_shape[loca_loss] ).abs().sum()/(torch.sum(loca_loss)+1)

        if("pose_l2" in opt.loss_type):
            loss_pose   = ( pred_pose_shape[loca_loss] - gt_pose_shape[loca_loss] ).pow(2).sum()/(torch.sum(loca_loss)+1)
            loss_pose = torch.nan_to_num(loss_pose, nan=0.0, posinf=0.0, neginf=0.0)
            
        if("loca_l1" in opt.loss_type):
            loss_loca = ( pred_location[loca_loss] - gt_location[loca_loss] ).abs().sum()/(torch.sum(loca_loss)+1)
            loss_loca = torch.nan_to_num(loss_loca, nan=0.0, posinf=0.0, neginf=0.0) * opt.loca_l1_weight
            
        if("loca_l2" in opt.loss_type):
            loss_loca = ( pred_location[loca_loss] - gt_location[loca_loss] ).pow(2).sum()/(torch.sum(loca_loss)+1)

        if("action" in opt.loss_type):
            if("ava" in opt.action_space):
                gt_has_annotation  = output_data['has_gt'][:, :, :, fi, :]
                
                if(opt.ava.gt_type=="gt"):
                    loca_loss_annot = torch.logical_and(loca_loss==1, gt_has_annotation[:, :, :, 0]==2)
                elif(opt.ava.gt_type=="pseduo_gt"):
                    loca_loss_annot = torch.logical_and(loca_loss==1, gt_has_annotation[:, :, :, 0]==1)
                elif(opt.ava.gt_type=="all"):
                    loca_loss_annot = torch.logical_and(loca_loss==1, gt_has_annotation[:, :, :, 0]>=1)
                else:
                    raise ValueError("Unknown ava gt type")

                if("BCE" in opt.loss_type.split("action")[1]):
                    loss_action = 10 * F.binary_cross_entropy_with_logits(pred_action_ava[loca_loss_annot], gt_action_ava[loca_loss_annot])
                    
            if("kinetics" in opt.action_space):
                
                gt_has_annotation_k = output_data['has_gt_kinetics'][:, 0, :, 0, 0]==1
                # cross entropy loss with one-hot encoding
                loss_action_k = F.cross_entropy(pred_action_kinetics[gt_has_annotation_k], gt_action_kinetics[:, 0][gt_has_annotation_k][:, 0].long()) # for 0th person
                loss_action += torch.nan_to_num(loss_action_k)
        
        # log the loss values
        loss_dict["pose"]    += loss_pose/opt.num_smpl_heads
        loss_dict["loca"]    += loss_loca/opt.num_smpl_heads
        loss_dict["action"]  += loss_action/opt.num_smpl_heads
    
    return loss_dict