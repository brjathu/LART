# PHALP Dataset

We ran PHALP on the `kinetics 400` and `AVA` to collect about 1.5 million human trajectories. Eack `.pkl` is a single tracklet. Naming convention is `<DATASET>_<VIDEO_ID>_<KEY_FRAME>_<TRACK_ID>_<TRACK_LENGTH>` . For example `ava-train_053oq2xB3oU_000150_3_128.pkl` means, this tracklet is from the `ava-train` dataset, from the video `053oq2xB3oU`, key frame is `000150` (will have annotations), with the track id `3` and the track length is `128` frames. Each tracklet contains the following information:
```python
import joblib

track = joblib.load("data/ava_train/ava-train_053oq2xB3oU_000150_3_128.pkl")

# Tracklet information
>>> track.keys()
dict_keys(['fid', 'pose_shape', '3d_joints', 'camera', 'camera_bbox', 'action_label_gt', 'action_label_psudo', 'has_gt', 'has_detection', 'apperance_index', 'apperance_dict', 'frame_name', 'frame_size', 'frame_bbox', 'frame_conf'])
```

- `fid`: counter for the frames in the tracklet # (128, 1, 1)
- `pose_shape`: SMPL pose, shape, global orient, and camera parameters # (128, 1, 229)
- `3d_joints`: 3D joints in camera coordinates # (128, 1, 45, 3)
- `camera`: camera parameters # (128, 1, 3)
- `camera_bbox`: camera with wrt bounding box # (128, 1, 3)
- `action_label_gt`: ground truth action label # (128, 1, 80)
- `action_label_psudo`: pseudo ground truth action label # <class 'dict'>, contains labels according to `apperance_index`
- `has_gt`: whether the tracklet has ground truth annotations # (128, 1, 1), 2 means gt is available, 1 means pseudo gt is available, 0 means no gt is available
- `has_detection`: whether the tracklet has detections # (128, 1, 1)
- `apperance_index`: index of the apperance features and pesudo labels of the dictionary # (128, 1, 1)
- `apperance_dict`: dictionary of apperance features # <class 'dict'>, contains apperance features from Hiera indexed by `apperance_index`
- `frame_name`: frame names # list of strings 
- `frame_size`: frame size # (128, 1, 2)
- `frame_bbox`: frame bounding box # (128, 1, 4)
- `frame_conf`: frame confidence # (128, 1, 1)

Please see the [phalp_action_dataset.py](lart/datamodules/components/phalp_action_datatset.py) dataset file for more details.