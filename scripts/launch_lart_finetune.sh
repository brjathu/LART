#!/bin/bash

# source ~/.zshrc
# conda activate lart

# python lart/train.py -m \
# --config-name lart.yaml \
# hydra/launcher=submitit_slurm \
# launcher=slurm \
# trainer=ddp_unused \
# task_name=LART_FT \
# configs.train_dataset="ava_train" \
# trainer.devices=8 \
# trainer.num_nodes=1 \
# configs.mask_ratio=0.0 \
# configs.solver.layer_decay=0.95 \
# configs.solver.lr=0.0007 \
# configs.ava.gt_type='gt' \
# configs.test_type=track.fullframe@avg.6 \
# configs.weights_path="logs/LART/0/checkpoints/epoch_024-EMA.ckpt" \