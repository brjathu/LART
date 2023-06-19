#!/bin/bash

# source ~/.zshrc
# conda activate lart

# # LART full model. 
python lart/train.py -m \
--config-name lart.yaml \
hydra/launcher=submitit_slurm \
launcher=slurm \
trainer=ddp_unused \
task_name=LART \
trainer.devices=8 \
trainer.num_nodes=8 \


# # LART Pose model 
# python lart/train.py -m \
# --config-name lart.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp_unused \
# task_name=LART_Pose \
# configs.extra_feat.enable=\'\' \
# configs.in_feat=128 \
# trainer.devices=8 \
# trainer.num_nodes=8 \




