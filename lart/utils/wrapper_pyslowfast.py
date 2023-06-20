
import cv2
import numpy as np
import torch
from phalp.utils.io import IO_Manager

def SlowFastWrapper(t, cfg, list_of_imgs, mid_frame_bboxes, video_model, center_crop=False, half=False):            
    from slowfast.visualization.utils import TaskInfo

    buffer_size    = cfg.DEMO.BUFFER_SIZE
    test_crop_size = cfg.DATA.TEST_CROP_SIZE
    clip_vis_size  = cfg.DEMO.CLIP_VIS_SIZE
    
    images = []
    c = 0
    
    for frame in list_of_imgs:
        img_ = IO_Manager.read_frame(frame)

        # for better predictions
        # th_1 = 1.5
        # th_2 = 0.67

        th_1 = 1.2
        th_2 = 0.8
        if(half):
            th_1 = 1.0
            th_2 = 1.0
        
        if(center_crop):
            h, w = img_.shape[:2]
            r_   = w/h
            if(r_>th_1 and w>h):
                w_   = h*th_1
                l_   = int((w-w_)//2)
                img_ = img_[:, l_:-l_, ::]
                if(c==0):
                    mid_frame_bboxes = [[mid_frame_bboxes[0][0]-l_, 
                                         mid_frame_bboxes[0][1], 
                                         mid_frame_bboxes[0][2]-l_, 
                                         mid_frame_bboxes[0][3]]]
                    mid_frame_bboxes = np.array(mid_frame_bboxes)
            elif(r_<th_2 and w<h):
                h_   = w/th_2
                t_   = int((h-h_)//2)
                img_ = img_[t_:-t_, :, ::]
                if(c==0):
                    mid_frame_bboxes = [[mid_frame_bboxes[0][0], 
                                         mid_frame_bboxes[0][1]-t_, 
                                         mid_frame_bboxes[0][2], 
                                         mid_frame_bboxes[0][3]-t_]]
                    mid_frame_bboxes = np.array(mid_frame_bboxes)
        c += 1
        images.append(img_)
    
    task = TaskInfo()
    task.img_height    = images[0].shape[0]
    task.img_width     = images[0].shape[1]
    task.crop_size     = test_crop_size
    task.clip_vis_size = clip_vis_size
    task.add_frames(t, images)
    task.add_bboxes(torch.from_numpy(mid_frame_bboxes).float().cuda())
    with torch.no_grad(): 
        task = video_model(task, half=half)
    
    return task