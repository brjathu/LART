import gc
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from phalp.configs.base import CACHE_DIR, FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

# create the tracker with hmr2 backend
class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2Predictor(self.cfg)

# create the tracker with action predictor
class LART(HMR2_4dhuman):
    def __init__(self, cfg):

        download_files = {
            "lart_mvit.config" : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/lart_mvit.config", os.path.join(CACHE_DIR, "phalp/ava")],
            "lart_mvit.ckpt"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/lart_mvit.ckpt", os.path.join(CACHE_DIR, "phalp/ava")],
            "mvit.yaml"        : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/mvit.yaml", os.path.join(CACHE_DIR, "phalp/ava")],
            "mvit.pyth"        : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/mvit.pyth", os.path.join(CACHE_DIR, "phalp/ava")],
        }
        self.cached_download_from_drive(download_files)
        super().__init__(cfg)

    def setup_detectron2(self): pass
    def setup_detectron2_with_RPN(self): pass

    def setup_predictor(self):
        # setup predictor model witch predicts actions from poses
        log.info("Loading Predictor model...")
        from lart.utils.wrapper_phalp import Pose_transformer
        self.pose_predictor = Pose_transformer(self.cfg, self)
        self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""

    # # Setup the tracker and track the video
    cfg.phalp.low_th_c = 0.5
    cfg.phalp.small_w = 20
    cfg.phalp.small_h = 20
    cfg.render.enable = False
    phalp_tracker = HMR2_4dhuman(cfg)
    _, pkl_path = phalp_tracker.track()
    del phalp_tracker
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    
    # Setup the LART model and run it on the tracked video to get the action predictions
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    cfg.render.enable = True
    cfg.render.colors = 'slahmr'
    cfg.pose_predictor.config_path = f"{CACHE_DIR}/phalp/ava/lart_mvit.config"
    cfg.pose_predictor.weights_path = f"{CACHE_DIR}/phalp/ava/lart_mvit.ckpt"
    try:    cfg.pose_predictor.half = cfg.half
    except: cfg.pose_predictor.half = False
    log.info(f"Half precision: {cfg.pose_predictor.half}")
    lart_model = LART(cfg)
    lart_model.setup_postprocessor()
    lart_model.postprocessor.run_lart(pkl_path)

if __name__ == "__main__":
    main()
