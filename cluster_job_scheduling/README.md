# Preface

The codes for cluster job scheduling (CJS) are implemented based on the repository of  [spark-sched-sim](https://github.com/ArchieGertsman/spark-sched-sim). Thanks for spark-sched-sim's authors for their open source codes! 

What is CJS? 

> Cluster job scheduling (CJS) plays a critical role in optimizing the allocation of computational resources in distributed computing environments, where multiple jobs need to be processed simultaneously. It usually need a policy to schedule incoming jobs within the cluster. Each job is represented as a directed acyclic graph (DAG), which describes the dependencies between each execution stage and the resource requirements of each stage. The primary task of the policy is to select the next stage of a job to execute and allocate a set of executors (computing resources) to that stage. The objective is to minimize the average job completion time, thereby optimizing the system-level utilization of computing resources within the cluster.

# Code structure

- `artifacts`: Stores some artifacts, e.g., result files.
    - `exp_pool`: Stores the experience pool files, which will be used for LLM adaptation.
    - `results`: Stores the result files.
- `stdout`: Output information generated when training baselines. (not necessary)
- `checkpoints`: Output information generated when training baselines. (not necessary)
- `config`: Stores configuration files for baselines, which are used during the running/training of baselines.
- `data`: Stores datasets. Note that this dataset is not the same as the experience pool. It is used to generate the simulation environment.
- `models`: Stores pre-trained model weights for baselines. 

- `spark_sched_sim`: Code directory for implementing the simulation environment, which is from [spark-sched-sim](https://github.com/ArchieGertsman/spark-sched-sim).
- `trainers`: Files related to training baselines.

- `plm_special`: Stores core codes for running NetLLM on cluster job scheduling.
    - `data`: Stores codes related to the training dataset.
      - `exp_pool.py`: Implements the experience pool for collecting trajectories.
      - `dataset.py`: Implements a dataset class that wraps the experience pool.
    - `models`: Stores codes related to models.
      - `gpt2.py, llama.py, t5.py`: Customized LLMs.
      - `low_rank.py`: Implements the low rank matrices.
      - `rl_policy.py`: Implements the Transformer-based offline RL policy.
      - `state_encoder.py`: Implements the feature encoder for encoding DAGs.
    - `utils`: Stores some utilization codes.
      - `plm_utils.py`: Some codes for loading LLMs.
      - `utils.py`: Some codes for data processing.
    - `trainer.py`: Implements a wrapper class for the training process.
    - `evaluate.py`: Some codes for validation process.
    - `test.py`: Some codes for testing process.

- `cfg_loader.py`: Some codes for loading configuration files for baselines from the `config` directory.
- `train_baseline.py`: A file provided by [spark-sched-sim](https://github.com/ArchieGertsman/spark-sched-sim) for training baselines.

- `generate_exp_pool.py`: Implements the generation of experience pool (i.e., training dataset for LLM).
- `run_baseline.py`: The main file for running baselines. 
- `run_plm.py`: The main file for running NetLLM.

# Environment Setup

1. Create a conda environment with `python=3.11.9` and activate it. Other versions of python might be okay as well.

   ```sh
   conda create -n cjs_netllm python==3.11.9 -y
   conda activate cjs_netllm
   ```

2. Install the following dependencies (the package versions are provided just for reference): 
   ```shell
   # For Pytorch
   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
   
   # For PyG
   pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
   pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
   pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
   pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.4.0+cu118.html
   pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric
   
   # For Gymnasium
   conda install swig -y
   pip install "Gymnasium[all]"
   
   # For other packages
   pip install numpy==1.26.4
   pip install transformers==4.37.1
   pip install munch==4.0.0
   pip install openprompt==1.0.1
   pip install peft==0.13.2
   ```

# Usage

## Usage of NetLLM
To run NetLLM, first we need to download the pretrained weights of some LLMs. For example, if you want to use Llama2-7b as the foundation model, please download Llama2-7b in the directory: `../downloaded_plms/llama2/base`. In the following, we will use the Llama2-7b as the example to illustrate the usage of NetLLM.

**Finetune LLM**

- If you want to finetune LLM, you can run the following command:

    ```sh
    python run_plm.py \
        --train \
        --test \
        --seed 666 \
        --plm-type llama \
        --plm-size base \
        --peft-rank 128 \
        --device cuda:0 \
        --device-out cuda:0 \
        --state-feature-dim 256 \
        --K 20 \
        --gamma 1.0 \
        --lr 0.0001 \
        --num-iters 40 \
        --freeze-encoder
    ```
    
    This command will finetune Llama2 on the default experience pool we provided at `artifacts/exp_pools/exp_pool.pkl`.
    
- If you want to use your own experience pool, first use the `generate_exp_pool.py` to generate a new experience pool.

    ```sh
    python generate_exp_pool.py \
        --scheds decima \
        --pool-size 1000 \
        --complete-episode \
        --num-executors 50 \
        --job-arrival-cap 200 \
        --job-arrival-rate 4e-5 \
        --moving-delay 2000 \
        --warmup-delay 1000 \
        --render-mode None \
        --dataset tpch \
        --seed 1 \
        --device cuda:0
    ```

    Next, specify the path to your own experience pool with argument `--exp-pool-path` and run the following command:

    ```sh
    python run_plm.py \
        --train \
        --test \
        --seed 666 \
        --plm-type llama \
        --plm-size base \
        --peft-rank 128 \
        --device cuda:0 \
        --device-out cuda:0 \
        --state-feature-dim 256 \
        --K 20 \
        --gamma 1.0 \
        --lr 0.0001 \
        --num-iters 40 \
        --freeze-encoder \
        --exp-pool-path "your_exp_pool_path"
    ```

**Test LLM**

If you want to test the performance of the finetuned LLM, please run the following command:
```sh
python run_plm.py \
    --test \
    --plm-type llama \
    --plm-size base \
    --peft-rank 128 \
    --state-feature-dim 256 \
    --device cuda:0
```
You can also specify the path to the finetuned LLM with argument `--model-dir`:
```sh
python run_plm.py \
    --test \
    --plm-type llama \
    --plm-size base \
    --peft-rank 128 \
    --state-feature-dim 256 \
    --device cuda:0 \
    --model-dir you_finetune_llm_dir
```

## Run baselines

To run baselines, you can run the following command:
```sh
python run_baseline.py \
    --sched decima \
    --num-executors 50 \
    --job-arrival-cap 200 \
    --seed 666 \
    --device cuda:0

python run_baseline.py \
    --sched fair \
    --num-executors 50 \
    --job-arrival-cap 200 \
    --seed 666

python run_baseline.py \
    --sched fifo \
    --num-executors 50 \
    --job-arrival-cap 200 \
    --seed 666
```


# Citation
If you find this repository useful, please kindly cite the following papers:
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

@misc{spark_sim,
    title={A Gymnasium environment for simulating job scheduling in Apache Spark},
    year={2024},
    author={Arkadiy Gertsman},
    note= {Accessed: 2024-12-15},
    url={https://github.com/ArchieGertsman/spark-sched-sim}
}
```