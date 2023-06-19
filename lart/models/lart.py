import copy
import math
import os
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from phalp.configs.base import CACHE_DIR, FullConfig
from phalp.utils.smpl_utils import SMPL
from phalp.utils.utils_download import cache_url
from torchmetrics import MeanMetric

from lart.evaluators.ava import AVA_evaluator
from lart.models.components.lart_transformer.transformer import \
    lart_transformer
from lart.utils import get_pylogger
from lart.utils.losses import compute_loss
from lart.utils.utils_plot import read_ava_pkl

log = get_pylogger(__name__)


class LART_LitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):

        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False,)
        self.cfg = self.hparams.cfg
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        
        # download necessary files
        self.cached_download_from_drive()

        # create the model
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
                                device      = self.device,
                                )
        

        # CREATE VARIOUS EVALUATORS
        self.evaluator_ava = AVA_evaluator(self.cfg)

        ############ SMPL stuff ##############
        self.phalp_cfg = FullConfig()
        smpl_params  = {k.lower(): v for k,v in asdict(self.phalp_cfg.SMPL).items()}
        self.smpl     = SMPL(**smpl_params)

        self.best_val_acc = 0
        try:
            results_map = read_ava_pkl(self.cfg.storage_folder + "/results/", best=True)
            self.best_val_acc = np.mean(results_map["all"][1])
        except Exception as e:
            log.warning(e)
            
        os.makedirs(self.cfg.storage_folder + "/slowfast/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/videos/", exist_ok=True)
        log.info("Storage folder : " + self.cfg.storage_folder)
                
    def forward(self, tokens, mask_type):

        output, vq_loss       = self.encoder(tokens, mask_type)
        smpl_outputs          = self.decode(output, input=False, tokens=tokens)
    
        return output, smpl_outputs, vq_loss

    def decode(self, output, input=False, tokens=None, render=False):
            
        # return predicted gt pose, betas and location
        class_token      = output[:, 0:self.cfg.max_people, :].contiguous()
        pose_tokens      = output[:, self.cfg.max_people:, :].contiguous()
        
        smooth_embed     = pose_tokens.view(pose_tokens.shape[0]*pose_tokens.shape[1], -1)
        smooth_embed_    = pose_tokens.view(pose_tokens.shape[0]*pose_tokens.shape[1], -1)
        
        pred_smpl_params = [self.encoder.smpl_head[i](smooth_embed_[:, :])[0] for i in range(self.cfg.num_smpl_heads)]
        pred_cam         = [self.encoder.loca_head[i](smooth_embed_[:, :]) for i in range(self.cfg.num_smpl_heads)]
        
        action_preds_ava = [self.encoder.action_head_ava[i](smooth_embed[:, :]) for i in range(self.cfg.num_smpl_heads)]
        action_preds_ava = torch.stack(action_preds_ava, dim=0)
        action_preds_ava = action_preds_ava.permute(1, 0, 2)
        
        smpl_output      = self.smpl(**{k: v.float() for k,v in pred_smpl_params[0].items()}, pose2rot=False)
        BS               = smooth_embed.size(0)
        dtype            = smooth_embed.dtype
        focal_length     = self.phalp_cfg.EXTRA.FOCAL_LENGTH * torch.ones(BS, 2, dtype=dtype).to(output.device)

        pred_cam_t       = torch.stack(pred_cam, dim=0)
        pred_cam_t       = pred_cam_t.permute(1, 0, 2)

        action_preds_kinetics = self.encoder.action_head_kinetics(class_token)
        
        ##### reporject the keypoints #####
            
        smpl_outputs = {
            'pred_smpl_params'      : pred_smpl_params,
            'smpl_output'           : smpl_output,
            'cam'                   : pred_cam,
            'cam_t'                 : pred_cam_t,
            'focal_length'          : focal_length,
            'pred_actions_ava'      : action_preds_ava,
            'pred_actions_kinetics' : action_preds_kinetics,
        }
        
        return smpl_outputs

    def on_train_start(self):
        torch.cuda.empty_cache()
        if(self.cfg.debug): self.trainer.datamodule.data_train.__getitem__(0)
        pass
    
    def step(self, batch: Any):
        input_data, output_data, _, _ = batch
        output, smpl_output, vq_loss  = self.forward(input_data, mask_type=self.cfg.mask_type)
        loss_dict                     = compute_loss(self.cfg, output, smpl_output, output_data, input_data, train=True)
        loss_dict['vq_loss']          = vq_loss
        
        return loss_dict, output, smpl_output

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict, output, smpl_output = self.step(batch)
        loss = sum([v for k,v in loss_dict.items()])

        self.train_loss(loss.item())
        
        for key in loss_dict.keys():
            self.log("train/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True)
        
        self.log_iter_stats(batch_idx)
            
        del loss_dict, output, smpl_output, batch
        
        return {"loss": loss}

    def on_train_epoch_end(self):
        log.info("\n " + self.cfg.storage_folder +  " : Training epoch " + str(self.current_epoch) + " ended.")
        
    def on_validation_start(self):
        torch.cuda.empty_cache()
        if(self.cfg.debug): 
            self.trainer.datamodule.data_val.__getitem__(0)

    def validation_step(self, batch: Any, batch_idx: int):

        input_data, output_data, meta_data, video_name = batch

        slowfast_paths = [self.cfg.storage_folder + "/slowfast/" + "".join(video_name[i].split(".jpg"))+".pkl" for i in range(len(video_name))]

        output, smpl_output, _ = self.forward(input_data, self.cfg.mask_type_test)
        
        loss_dict = compute_loss(self.cfg, output, smpl_output, output_data, input_data, train=False)
        loss = sum([v for k,v in loss_dict.items()])
        
        if(self.cfg.compute_map and "ava" in self.cfg.action_space): 
            self.evaluator_ava.store_results_batch(input_data, output_data, meta_data, smpl_output, video_name, slowfast_paths, output=copy.deepcopy(output[:, self.cfg.max_people:, :]))

        # update and log metrics
        self.val_loss(loss.item())
        for key in loss_dict.keys():
            self.log("val/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log_iter_stats(batch_idx)
            
        del output, smpl_output, loss_dict, input_data, output_data, meta_data, video_name, batch
            
        return {"loss": loss}

    @rank_zero_only
    def on_validation_epoch_end(self):
                    
        # logic to read the results and compute the metrics
        if(self.cfg.compute_map and "ava" in self.cfg.action_space):
            self.evaluator_ava.write_ava_csv(self.cfg.storage_folder + "/slowfast/", self.cfg.storage_folder + "/ava_val.csv")
            
            current_map = self.evaluator_ava.compute_map(self.current_epoch)

            self.log("mAP : ", current_map[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0)
            self.log("val/mAP", current_map[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0, prog_bar=True)
            

            if(self.best_val_acc < current_map[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0):
                os.system("cp " + self.cfg.storage_folder + "/ava_val.csv" + " " + self.cfg.storage_folder + "/ava_val_best.csv")
                self.best_val_acc = current_map[0]['PascalBoxes_Precision/mAP@0.5IOU']*100.0

        else:
            log.info("mAP : " + str(0))
            self.log("val/mAP", 0, prog_bar=True)
    
    def log_iter_stats(self, cur_iter):
        
        def gpu_mem_usage():
            """Computes the GPU memory usage for the current device (MB)."""
            mem_usage_bytes = torch.cuda.max_memory_allocated()
            return mem_usage_bytes / 1024 / 1024
        
        if(cur_iter%self.cfg.log_frequency != 0):
            return 0
        
        mem_usage = gpu_mem_usage()
        try:
            stats = {
                "epoch": "{}/{}".format(self.current_epoch, self.trainer.max_epochs),
                "iter": "{}/{}".format(cur_iter + 1, self.trainer.num_training_batches),
                "train_loss": "%.4f"%(self.train_loss.compute().item()),
                "val_loss": "%.4f"%(self.val_loss.compute().item()),
                "time": "%.4f"%(self.timer.time_elapsed()-self.timer_last_iter),
                "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
                "mem": int(np.ceil(mem_usage)),
            }
            self.timer_last_iter = self.timer.time_elapsed()
        except:
            for cb_ in self.trainer.callbacks:
                if(cb_.__class__.__name__ == "Timer"):
                    self.timer = cb_
            self.timer_last_iter = self.timer.time_elapsed()
            stats = {}
            
        self.train_loss.reset()
        self.val_loss.reset()
        
        log.info(stats)
    
    def get_param_groups(self):
        def _get_layer_decay(name):
            layer_id = None
            if name in ("encoder.class_token", "encoder.pose_token", "encoder.mask_token"):
                layer_id = 0
            elif ("_encoder" in name):
                layer_id = 0
            elif ("_head" in name):
                layer_id = self.cfg.transformer.depth + 1
            elif name.startswith("encoder.pos_embedding"):
                layer_id = 0
            elif name.startswith("encoder.transformer1.layers"):
                layer_id = int(name.split("encoder.transformer1.layers.")[1].split(".")[0]) + 1
            else:
                layer_id = self.cfg.transformer.depth + 1
            layer_decay = self.cfg.solver.layer_decay ** (self.cfg.transformer.depth + 1 - layer_id)
            return layer_id, layer_decay

        non_bn_parameters_count = 0
        zero_parameters_count = 0
        no_grad_parameters_count = 0
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, p in self.named_parameters():
            if not p.requires_grad:
                group_name = "no_grad"
                no_grad_parameters_count += 1
                continue
            name = name[len("module."):] if name.startswith("module.") else name
            if ((len(p.shape) == 1 or name.endswith(".bias")) and self.cfg.solver.ZERO_WD_1D_PARAM):
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "zero")
                weight_decay = 0.0
                zero_parameters_count += 1
            else:
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "non_bn")
                weight_decay = self.cfg.solver.weight_decay
                non_bn_parameters_count += 1

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.cfg.solver.lr * layer_decay,
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.cfg.solver.lr * layer_decay,
                }
            parameter_group_names[group_name]["params"].append(name)
            parameter_group_vars[group_name]["params"].append(p)

        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        # linear learning rate scaling for multi-gpu
        if(self.trainer.num_devices * self.trainer.num_nodes>1 and self.cfg.solver.apply_linear_scaling):
            self.lr_scaler = self.trainer.num_devices * self.trainer.num_nodes * self.trainer.accumulate_grad_batches * self.cfg.train_batch_size / 256
        else:
            self.lr_scaler = 1
        log.info("num_devices: {}, num_nodes: {}, accumulate_grad_batches: {}, train_batch: {}".format(self.trainer.num_devices, self.trainer.num_nodes, self.trainer.accumulate_grad_batches, self.cfg.train_batch_size))
        log.info("Linear LR scaling factor: {}".format(self.lr_scaler))
        
        if(self.cfg.solver.layer_decay is not None):
            optim_params = self.get_param_groups()
        else:
            optim_params = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': self.cfg.solver.lr * self.lr_scaler}]
        
        if(self.cfg.solver.name=="AdamW"):
            optimizer = optim.AdamW(params=optim_params, weight_decay=self.cfg.solver.weight_decay, betas=(0.9, 0.95))
        elif(self.cfg.solver.name=="lion"):
            from lion_pytorch import Lion
            optimizer = Lion(params=optim_params, weight_decay=self.cfg.solver.weight_decay, betas=(0.9, 0.99))
        elif(self.cfg.solver.name=="SGD"):
            optimizer = optim.SGD(params=optim_params, momentum=self.cfg.solver.momentum, weight_decay=self.cfg.solver.weight_decay)
        else:
            raise NotImplementedError("Unknown solver : " + self.cfg.solver.name)

        def warm_start_and_cosine_annealing(epoch):
            if epoch < self.cfg.solver.warmup_epochs:
                lr = (epoch+1) / self.cfg.solver.warmup_epochs
            else:
                lr = 0.5 * (1. + math.cos(math.pi * ((epoch+1) - self.cfg.solver.warmup_epochs) / (self.trainer.max_epochs - self.cfg.solver.warmup_epochs )))
            return lr

        if(self.cfg.solver.scheduler == "cosine"):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_start_and_cosine_annealing for _ in range(len(optim_params))], verbose=False)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.cfg.solver.decay_steps, gamma=self.cfg.solver.decay_gamma, verbose=False)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval" : "epoch",
                'frequency': 1,
            }
        }

    def cached_download_from_drive(self):
        """Download a file from Google Drive if it doesn't exist yet.
        :param url: the URL of the file to download
        :param path: the path to save the file to
        """
        
        os.makedirs(os.path.join(CACHE_DIR, "lart"), exist_ok=True)

        download_files = {
            "class_sum.pkl" : ["https://people.eecs.berkeley.edu/~jathushan/projects/lart/class_sum.pkl", os.path.join(CACHE_DIR, "lart")],
            "ava_valid_classes.npy"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/lart/ava_valid_classes.npy", os.path.join(CACHE_DIR, "lart")],
            "ava_action_list.pbtxt"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/lart/ava_action_list.pbtxt", os.path.join(CACHE_DIR, "lart")],
            "ava_action_list_v2.2.pbtxt"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/lart/ava_action_list_v2.2.pbtxt", os.path.join(CACHE_DIR, "lart")],
            "kinetics_annot_train.pkl"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/lart/kinetics_annot_train.pkl", os.path.join(CACHE_DIR, "lart")],
            "ava_val_v2.2.csv" : ["https://people.eecs.berkeley.edu/~jathushan/projects/lart/ava_val_v2.2.csv", os.path.join(CACHE_DIR, "lart")],
            "ava_action_list_v2.2_for_activitynet_2019.pbtxt" : ["https://people.eecs.berkeley.edu/~jathushan/projects/lart/ava_action_list_v2.2_for_activitynet_2019.pbtxt", os.path.join(CACHE_DIR, "lart")],
        }
        
        for file_name, url in download_files.items():
            if not os.path.exists(os.path.join(url[1], file_name)):
                print("Downloading file: " + file_name)
                # output = gdown.cached_download(url[0], os.path.join(url[1], file_name), fuzzy=True)
                output = cache_url(url[0], os.path.join(url[1], file_name))
                assert os.path.exists(os.path.join(url[1], file_name)), f"{output} does not exist"



if __name__ == "__main__":
    pass