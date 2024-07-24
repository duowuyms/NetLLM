# llama [base]
python run_plm.py --train --train-dataset Jin2022 --test --test-dataset Jin2022 --dataset-type 360 --dataset-frequency 5 --sample-step 15 --his-window 10 --fut-window 20 --plm-type llama --plm-size base --epochs 43 --bs 1 --lr 0.0002 --grad-accum-steps 32 --head-type linear --device cuda:1 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --lora-rank 32 --multimodal --scheduled-sampling
# mistral [base]
python run_plm.py --train --train-dataset Jin2022 --test --test-dataset Jin2022 --dataset-type 360 --dataset-frequency 5 --sample-step 15 --his-window 10 --fut-window 20 --plm-type mistral --plm-size base --epochs 43 --bs 1 --lr 0.00008 --grad-accum-steps 32 --head-type linear --device cuda:1 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --lora-rank 32 --multimodal --scheduled-sampling
# llava
python run_plm.py --train --train-dataset Jin2022 --test --test-dataset Jin2022 --dataset-type 360 --dataset-frequency 5 --sample-step 15 --his-window 10 --fut-window 20 --plm-type llava --plm-size base --epochs 43 --bs 1 --lr 0.0002 --grad-accum-steps 32 --head-type linear --device cuda:1 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --lora-rank 32 --multimodal --scheduled-sampling
# opt [xxs, xs, small, base large]
python run_plm.py --train --train-dataset Jin2022 --test --test-dataset Jin2022 --dataset-type 360 --dataset-frequency 5 --sample-step 15 --his-window 10 --fut-window 20 --plm-type opt --plm-size base --epochs 43 --bs 1 --lr 0.0002 --grad-accum-steps 32 --head-type linear --device cuda:1 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --lora-rank 32 --multimodal --scheduled-sampling

# baseline
# linear regression
python run_baseline.py --model regression --train --test --device cuda:2 --train-dataset Jin2022 --test-dataset Jin2022 --dataset-type 360 --lr 0.0005 --bs 64 --epochs 80 --seed 1 --compile --device cuda:2 --his-window 10 --fut-window 5 --dataset-frequency 5 --sample-step 15
# velocity
python run_baseline.py --model velocity --test --device cuda:2 --train-dataset Jin2022 --test-dataset Jin2022 --dataset-type 360 --lr 0.0005 --bs 64 --epochs 80 --seed 1 --compile --device cuda:2 --his-window 10 --fut-window 5 --dataset-frequency 5 --sample-step 15

# prompt
python run_prompt_with_task_head.py --train --train-dataset Jin2022 --test --test-dataset Jin2022 --dataset-type 360 --dataset-frequency 5 --sample-step 15 --his-window 10 --fut-window 20 --plm-type llama --plm-size base --template-id 3 --soft-token-num 0 --epochs 100 --bs 1 --lr 0.0002 --grad-accum-steps 16 --lr 0.0002 --head-type linear --max-seq-length 1024 --device cuda:3 --lora-rank 32