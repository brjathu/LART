# On the Benefits of 3D Pose and Tracking for Human Action Recognition (CVPR 2023)
Code repository for the paper "On the Benefits of 3D Pose and Tracking for Human Action Recognition". \
[Jathushan Rajasegaran](http://people.eecs.berkeley.edu/~jathushan/), [Georgios Pavlakos](https://geopavlakos.github.io/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/). \
[![arXiv](https://img.shields.io/badge/arXiv-2304.01199-00ff00.svg)](https://arxiv.org/abs/2304.01199)       [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://people.eecs.berkeley.edu/~jathushan/LART/)     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QRLqEAePmgS41v0KQwf87G_Ss_BLhPYs?usp=sharing)
 
This code repository provides a code implementation for our paper LART (Lagrangian Action Recognition with
Tracking), with installation, training, and evaluating on datasets, and a demo code to run on any youtube videos. 

<!-- <p align="center"><img src="./assets/imgs/teaser.png" width="800"></p> -->
<p align="center"><img src="./assets/jump.gif" width="800"></p>

## Installation

After installing the [PyTorch 2.0](https://pytorch.org/get-started/locally/) dependency, you can install our `lart` package directly as:
```
pip install git+https://github.com/brjathu/LART
```

<details>
  <summary>Step-by-step instructions</summary>

```bash
conda create -n lart python=3.10
conda activate lart
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install -e .[demo]
```

If you only wants to train the model and not interested in running demo on any videos, you dont need to install packages rquired for demo code (`pip install -e .`).
</details>
<br>



## Demo on videos
Our Action recogition model uses PHALP to track people in videos, and then uses the tracklets to classify actions. `pip install -e .[demo]` will install nessory dependencies for running the demo code. Now, run `demo.py` to reconstruct, track and recognize humans in any video. Input video source may be a video file, a folder of frames, or a youtube link:
```bash
# Run on video file
python scripts/demo.py video.source="assets/jump.mp4"

# Run on extracted frames
python scripts/demo.py video.source="/path/to/frames_folder/"

# Run on a youtube link (depends on pytube working properly)
python scripts/demo.py video.source=\'"https://www.youtube.com/watch?v=xEH_5T9jMVU"\'
```
The output directory (`./outputs` by default) will contain a video rendering of the tracklets and a `.pkl` file containing the tracklets with 3D pose and shape and action labels. Please see the [PHALP](https://github.com/brjathu/PHALP) repository for details. The model for demo, uses `MViT` as a backend and a single person model. The demo code requires about 32 GB memory to run the slowfast code. [TODO: Demo with `Hiera` backend.] 
<br>

## Training and Evaluation

### Download the datasets
We are releasing about 1.5 million tracks of people from Kinetics 400 and AVA datasets. Run the following command to download the data from dropbox and extract them to the `data` folder. This will download preprocessed data form `ava_val`, `ava_train` and `kinetics_train` datasets (preporcessed data for AVA-kinetics and multispots will be released soon). These tracjectories contains backbone features as well as ground-truth annotations and pseudo ground-truth annotations. For more details see [DATASET.md](DATASET.md)
```bash
./scripts/download_data.sh
```

### Train LART model

For single node training, run the following command. This will evaulate the model at every epochs and compute the mean average precision on the validation set. 

```bash
# # LART full model. 
python lart/train.py -m \
--config-name lart.yaml \
trainer=ddp_unused \
task_name=LART \
trainer.devices=8 \
trainer.num_nodes=1 \
```

### Evaluate LART model
First download the pretrained model by running the following command. This will download the pretrained model to `./logs/LART_Hiera/` folder. 
```bash
./scripts/download_checkpoints.sh
```

Then run the following command to evaluate the model on the validation set. This will compute the mean average precision on the validation set (AVA-K evaluation will be released soon).

```bash
# # LART full model. 
python lart/train.py -m \
--config-name lart.yaml \
trainer=ddp_unused \
task_name=LART_eval \
train=False \
trainer.devices=8 \
trainer.num_nodes=1 \
configs.test_type=track.fullframe@avg.6 \
configs.weights_path=logs/LART_Hiera/0/checkpoints/epoch_002-EMA.ckpt \
```
Please specify a different `configs.weights_path` to evaluate your own trained model. However every checkpoint will be evaluated during training, and results will be saved to `./logs/<MODEL_NAME>/0/results/` folder.

## Citation
If you find this code useful for your research or the use data generated by our method, please consider citing the following paper:
```bibtex
@inproceedings{rajasegaran2023benefits,
  title={On the Benefits of 3D Pose and Tracking for Human Action Recognition},
  author={Rajasegaran, Jathushan and Pavlakos, Georgios and Kanazawa, Angjoo and Feichtenhofer, Christoph and Malik, Jitendra},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={640--649},
  year={2023}
}
```

