
import csv
import os

import joblib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from phalp.configs.base import CACHE_DIR

def read_labelmap(labelmap_file):
  """Reads a labelmap without the dependency on protocol buffers.

  Args:
    labelmap_file: A file object containing a label map protocol buffer.

  Returns:
    labelmap: The label map in the form used by the object_detection_evaluation
      module - a list of {"id": integer, "name": classname } dicts.
    class_ids: A set containing all of the valid class id integers.
  """
  labelmap = {}
  name = ""
  class_id = ""
  class_type = ""
  for line in labelmap_file:
    if line.startswith("  name:"):
      name = line.split('"')[1]
    elif line.startswith("  id:") or line.startswith("  label_id:"):
      class_id = int(line.strip().split(" ")[-1])
    elif line.startswith("  label_type:"):
      class_type = line.strip().split(" ")[-1]
    labelmap[name] = {"id": class_id, "name": name, "type": class_type}
  return labelmap

def read_ava_pkl(pkl_file, refence_file=None, best=False, verbose=True, subtask="all"):
    labelmap = read_labelmap(open(os.path.join(CACHE_DIR, 'lart/ava_action_list_v2.2.pbtxt'), 'r'))
    class_sum = joblib.load(os.path.join(CACHE_DIR, 'lart/class_sum.pkl'))
    def get_actions(pkl_file):
            
        data          = joblib.load(pkl_file)
        mAP_values    = data[0]
        catagories    = data[1]
        catagories_   = {}
        map_per_class = {}
        map_per_class_OM = {}
        map_per_class_PI = {}
        map_per_class_PM = {}
        for m in catagories: catagories_[m['name']] = m['id']
        for key in mAP_values.keys():
            if("PascalBoxes_PerformanceByCategory/AP@0.5IOU" in key):
                key_                         = key.split("PascalBoxes_PerformanceByCategory/AP@0.5IOU/")[1]
                if(labelmap[key_]['type']!=""): #OBJECT_MANIPULATION, PERSON_INTERACTION,PERSON_MOVEMENT
                  map_per_class[key_]          = mAP_values[key]*100
                if(labelmap[key_]['type']=="OBJECT_MANIPULATION"): #OBJECT_MANIPULATION, PERSON_INTERACTION,PERSON_MOVEMENT
                  map_per_class_OM[key_]          = mAP_values[key]*100
                if(labelmap[key_]['type']=="PERSON_INTERACTION"): #OBJECT_MANIPULATION, PERSON_INTERACTION,PERSON_MOVEMENT
                  map_per_class_PI[key_]          = mAP_values[key]*100
                if(labelmap[key_]['type']=="PERSON_MOVEMENT"): #OBJECT_MANIPULATION, PERSON_INTERACTION,PERSON_MOVEMENT
                  map_per_class_PM[key_]          = mAP_values[key]*100

        actions    = list(map_per_class.keys())
        action_map = list(map_per_class.values())
        counts     = [class_sum[catagories_[i]] for i in actions]
        idx_       = np.argsort(counts)[::-1]
        actions    = np.array(actions)[idx_]
        action_map = np.array(action_map)[idx_]
        
        
        actions_OM    = list(map_per_class_OM.keys())
        action_map_OM = list(map_per_class_OM.values())
        counts_OM     = [class_sum[catagories_[i]] for i in actions_OM]
        idx_          = np.argsort(counts_OM)[::-1]
        actions_OM    = np.array(actions_OM)[idx_]
        action_map_OM = np.array(action_map_OM)[idx_]
        
        
        actions_PI    = list(map_per_class_PI.keys())
        action_map_PI = list(map_per_class_PI.values())
        counts_PI     = [class_sum[catagories_[i]] for i in actions_PI]
        idx_          = np.argsort(counts_PI)[::-1]
        actions_PI    = np.array(actions_PI)[idx_]
        action_map_PI = np.array(action_map_PI)[idx_]
        
        
        actions_PM    = list(map_per_class_PM.keys())
        action_map_PM = list(map_per_class_PM.values())
        counts_PM     = [class_sum[catagories_[i]] for i in actions_PM]
        idx_          = np.argsort(counts_PM)[::-1]
        actions_PM    = np.array(actions_PM)[idx_]
        action_map_PM = np.array(action_map_PM)[idx_]

        results = {
            "all" : [actions, action_map],
            "OM"  : [actions_OM, action_map_OM],
            "PI"  : [actions_PI, action_map_PI],
            "PM"  : [actions_PM, action_map_PM],
        }
        return results
        
    if(best):
        if(verbose): print("reading best file from ", pkl_file)
        files = os.listdir(pkl_file)
        score = []
        results = []
        for file_ in files:
            action_result = get_actions(os.path.join(pkl_file, file_))
            results.append(action_result)
            if(refence_file is None):
                score.append(np.mean(action_result[subtask][1]))
            else:
                actions_, action_map_ = refence_file[0], refence_file[1]
                sum_ = 0
                for i in range(len(actions)):
                    if(action_result[subtask][1][i]>action_map_[i]):
                        sum_ += action_result[subtask][1][i]-action_map_[i]
                score.append(sum_)
                
        idx_ = np.argsort(score)[::-1]
        if(verbose): print(score)
        return results[idx_[0]]
    else:  
        if(".pkl" in pkl_file or ".pickle" in pkl_file):
            # read the given pkl file
            if(verbose): print("reading ", pkl_file)
            pkl_file = pkl_file
        else:
            # read the last pkl file
            if(verbose): print("reading last file.")
            files = os.listdir(pkl_file)
            filesId = [int(file.split(".")[0]) for file in files]
            ids     = np.argsort(filesId)
            pkl_file = os.path.join(pkl_file, files[-1])
            if(verbose): print(filesId)
            
        return get_actions(pkl_file)
    
    
    
def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}