# Preface
The codes for viewport prediction (VP) are implemented based on the repository of [MANSY](https://github.com/duowuyms/MANSY_ImmersiveVideoStreaming). 

What is VP?
> Viewport prediction (VP) serves as a fundamental building block of the emerging streaming systems of immersive videos (e.g., 360-degree videos and volumetric videos), where only the video content within the viewer’s viewport (the region visible to viewer) is streamed in high quality to reduce the bandwidth
consumption of video transmission. To accomplish this, the VP model predicts viewer’s future viewport positions based on historical viewports, and potentially incorporates video content information (e.g., video frame) to enhance prediction performance. The goal of VP is to minimize the error between the predicted and viewer’s actual viewports.

# Code Structure

- `data`: This directory stores datasets and pre-trained model checkpoints.
   - `viewports`: This directory stores the viewport datasets.
   - `images`: This directory stores the video images of the correponding viewport datasets.
   - `image_features`: This directory stores the ViT-extracted video image features of the correponding viewport datasets.
   - `ft_plms`: This directory stores the fine-tuned (adapted) LLMs.
   - `models`: This directory stores the model checkpoints of TRACK.

- `dataset`: This directory stores the codes for processing and loading datasets.
   - `preprocess.py`: Preprocess the viewport datasets.
   - `load_dataset.py`: Load datasets.
   - `extract_saliency.py`: Extract saliency map from video images.
   - `extract_features.py`: Extract features from the saliency map.

- `models`: This directory stores the codes for different VP models and the NetLLM implementation on the VP task.
   - `gpt2.py, llama.py, opt.py, mistral.py`: Customized LLMs.
   - `low_rank.py`: Implements the low rank matrices.
   - `pipeline.py`: Implements the NetLLM pipeline for VP task.
   - `regression.py, velocity.py, track.py`: Implements the baselines.
   - `old`: This directory stores the old implementation of our NetLLM. We have provide a model checkpoint of Llama2-7B that can reproduce the results in our paper. But to use it, we need to use the old implementation. So the codes in this directory are for interested researchers that want to try the trained Llama2-7B model. 
   
      **Warning**: The codes in this directory are a bit messy...

- `utils`: This directory stores some utilization codes.
   - `plm_utils.py`: Some codes for loading LLMs.
   - `metrics.py`: Some codes for calculating performance metrics.
   - `normalize.py`: Some codes for normalizing data.
   - `model_utils.py`: Some codes for creating baseline models.
   - `result_notebook.py`: Some codes for recording the performance results.
   - `console_logger.py`: Some codes for logging the console outputs.
- `run_baseline.py`: The main file for running baselines. 
- `run_plm.py`: The main file for running NetLLM.

- `run_plm.py`: The main file for running the old implementation of NetLLM (use it only if you want to try our trained Llama2-7B model).

# Dataset download
We have provided the cooked viewport datasets Jin2022 and Wu2017 in this repo. 

The images or image features of the viewport datasets need to be manually downloaded. The download link is here: TODO.

After downloading the datasets, please put the image datasets into `data/images`, and image feature datasets into `data/image_features`.

*Note: The image datasets are too large. We will find a way to upload them soon. At present, you can just try our codes without using the image datasets.*

# Environment Setup
## Environment for NetLLM
1. Create a conda environment for NetLLM:

   `conda create -n vp_netllm python>=3.8.10`

2. Then install the following depdendencies:

   ```
   python==3.8.10
   torch==2.1.0
   numpy==1.24.4
   munch==4.0.0
   transformers==4.34.1
   peft==0.6.2
   ```

# Usage
## Usage of NetLLM
To run NetLLM, first we need to download some LLMs. For example, if you want to use Llama2-7b as the foundation model, please download Llama2-7b in the directory: `../downloaded_plms/llama2/base`. In the following, we will use the Llama2-7b as the example to illustrate the usage of NetLLM.

**Finetune LLM**

If you want to finetune LLM, please run the following command:
```sh
python run_plm.py --adapt --train-dataset Jin2022 --his-window 10 --fut-window 20 --plm-type llama --plm-size base --epochs 40 --bs 1 --lr 0.0002 --grad-accum-steps 32 --device cuda:0 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling
```
This command will finetune Llama2 on the Jin2022 dataset.
Note that we have made using image features an option. If you want to add image features to the LLM inputs, just append the argument `--multimodal` to the above command.

**Test LLM**

If you want to test the performance of the finetuned LLM, please run the following command:
```sh
python run_plm.py --test --test-dataset Jin2022 --his-window 10 --fut-window 20 --plm-type llama --plm-size base --epochs 40 --bs 1 --lr 0.0002 --grad-accum-steps 32 --device cuda:0 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling
```
You can also specify the path to the finetuned LLM with argument `--model-path`:
```sh
python run_plm.py --test --test-dataset Jin2022 --his-window 10 --fut-window 20 --plm-type llama --plm-size base --epochs 40 --bs 1 --lr 0.0002 --grad-accum-steps 32 --device cuda:0 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling --model-path you_finetune_llm_dir
```

We offer the model checkpoint of the finetuned Llama2-7b here: https://drive.google.com/file/d/1SLguoavlsO6CV_I6cfAtOjp_IPj3XnE7/view?usp=sharing. If you want to try our model, please download the model checkpoint and store it in `data/ft_plms/try_llama2_7b`. To run this model, we need to use the old implementation of the NetLLM. Please run the following command:
```sh
python run_old.py --test --test-dataset Jin2022 --his-window 10 --fut-window 20 --plm-type llama --plm-size base --epochs 40 --bs 1 --lr 0.0002 --grad-accum-steps 32 --device cuda:0 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling --model-path data/ft_plms/try_llama2_7b
```

## Run baselines

To run baselines, please run:
```sh
python run_baseline.py --model regression --test --device cpu --test-dataset Jin2022  --bs 64 --seed 1  --his-window 10 --fut-window 20

python run_baseline.py --model velocity --test --device cpu --test-dataset Jin2022  --bs 64 --seed 1  --his-window 10 --fut-window 20

python run_baseline.py --model track --train --test --device cuda:2 --train-dataset Jin2022 --test-dataset Jin2022 --lr 0.0005 --bs 64 --epochs 80 --seed 1 --compile --device cuda:2 --his-window 10 --fut-window 20 --dataset-frequency 5 --sample-step 15
```
We also offer the model checkpoint of the TRACK model here: https://drive.google.com/file/d/1pQrxqDoAr5mBXR0GPXewdew_diHQELPt/view?usp=sharing. If you want to try this model, please download the model checkpoint and store it in `data/models/track/pretrain.pth`, then run the following command:
```sh
python run_baseline.py --model track --test --device cuda:2 --test-dataset Jin2022 --lr 0.0005 --bs 64 --epochs 80 --seed 1 --compile --device cuda:2 --his-window 10 --fut-window 20 --dataset-frequency 5 --sample-step 15 --model-path data/models/track/pretrain.pth
```


# Citation
If you find this repository useful, please kindly cite the following paper:
```
@inproceedings{wu2024netllm,
      author = {Wu, Duo and Wang, Xianda and Qiao, Yaqi and Wang, Zhi and Jiang, Junchen and Cui, Shuguang and Wang, Fangxin},
      title = {NetLLM: Adapting Large Language Models for Networking},
      year = {2024},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      doi = {10.1145/3651890.3672268},
      booktitle = {Proceedings of the ACM SIGCOMM 2024 Conference},
      pages = {661–678},
      numpages = {18},
      location = {Sydney, NSW, Australia},
      series = {ACM SIGCOMM '24}
}

@misc{wu2023mansy,
      title={MANSY: Generalizing Neural Adaptive Immersive Video Streaming With Ensemble and Representation Learning}, 
      author={Duo Wu and Panlong Wu and Miao Zhang and Fangxin Wang},
      year={2023},
      eprint={2311.06812},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2311.06812}, 
}
```
