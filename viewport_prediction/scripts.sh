# command examples for running models
# NOTE: please remember to check the arguments and add/remove/modify some arguments according to the actual case.

# DVMS
# 360 video
python run_models.py --model=dvms --train --test --device=cuda:2 --train-dataset=Wu2017 --test-dataset=Wu2017 --dataset-type=360 --dataset-frequency=10 --sample-step=10 --lr=0.0005 --bs=64 --epochs=80 --seed=1 --his-window=5 --fut-window=1 --compile 
# Volumetric video
python run_models.py --model=dvms --train --test --device=cuda:2 --train-dataset=Serhan2020 --test-dataset=Serhan2020 --dataset-type=vv --dataset-frequency=1 --sample-step=10 --lr=0.0005 --bs=64 --epochs=80 --seed=1 --his-window=5 --fut-window=1 --compile

# Linear regression
# 360 video
python run_models.py --model=regression --test --device=cpu --train-dataset=Wu2017 --test-dataset=Wu2017 --dataset-type=360 --dataset-frequency=10 --sample-step=10 --bs=64 --seed=1 --his-window=5 --fut-window=1
# Volumetric video
python run_models.py --model=regression --test --device=cpu --train-dataset=Serhan2020 --test-dataset=Serhan2020 --dataset-type=vv --dataset-frequency=10 --sample-step=1 --bs=64 --seed=1 --his-window=5 --fut-window=1

# Prompt pipeline
# 360 video
python run_prompt.py --plm-type=t5-lm --plm-size=base --template-id=7 --soft-token-num=0 --train --test --device=cuda:1 --train-dataset=Wu2017 --test-dataset=Wu2017 --dataset-type=360 --dataset-frequency=10 --sample-step=10 --load-dataset-cache --lr=0.05 --bs=4 --epochs=100 --epochs-per-valid=10 --seed=1 --his-window=5 --fut-window=1 --compile --write-sentences 
# Volumetric video
python run_prompt.py --plm-type=t5-lm --plm-size=base --template-id=7 --soft-token-num=0 --train --test --device=cuda:1 --train-dataset=Serhan2020 --test-dataset=Serhan2020 --dataset-type=vv --dataset-frequency=10 --sample-step=1 --load-dataset-cache --lr=0.05 --bs=4 --epochs=100 --epochs-per-valid=10 --seed=1 --his-window=5 --fut-window=1 --compile --write-sentences 

# PLM adaption
python adapt_plm.py --plm-type=t5-lm --plm-size=base --template-id=0 --train --test --device=cuda:3 --dataset-frequency=10 --sample-step=5 --load-dataset-cache --lr=0.0001 --bs=4 --steps=10000 --steps-per-valid=100 --seed=1 --cand-num-axes 3 6 --cand-his-window 5 8 10 15 20 --cand-fut-window 2 5 8 10 15 --precision=4 --max-seq-length=768 --decoder-max-length=256 --compile --write-sentences --warmup

# Some information about the number of parameters of different PLMs
# T5: small=80M, base=220M, large=780M, xl=3B, xxl=11B
# GPT2: small=117M, base=345M, large=762M, xl=1.5B
# BERT: base=110M, large=340M
# RoBERTa: base=110M, large=340M