#!/bin/bash
# Please run this code first.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Then we can start!
python /U2B_AAAI/main.py --dataset='PROTEINS'  --lr=0.001 --weight_decay=0.0005  --K_g=24 --K_v=48 --topk=12 --split_mode=high 
python /root/U2B_AAAI/main.py --dataset='PROTEINS'  --lr=0.001 --weight_decay=0.0005  --K_g=24 --K_v=96 --topk=12 --split_mode=low 

python /U2B_AAAI/main.py --dataset='NCI1'  --lr=0.001 --weight_decay=0.0005  --K_g=16 --K_v=64 --topk=8 --split_mode=high 
python /root/U2B_AAAI/main.py --dataset='NCI1'  --lr=0.001 --weight_decay=0.0005  --K_g=8 --K_v=48 --topk=4 --split_mode=low 

python /U2B_AAAI/main.py --dataset='DD'  --lr=0.001 --weight_decay=0.0005  --K_g=16 --K_v=32 --topk=8 --split_mode=high 
python /U2B_AAAI/main.py --dataset='DD'  --lr=0.001 --weight_decay=0.0005  --K_g=16 --K_v=48 --topk=8 --split_mode=low 

python /U2B_AAAI/main.py --dataset='COLLAB'  --lr=0.001 --weight_decay=0.0005  --K_g=32 --K_v=48 --topk=16 --split_mode=high 
python /U2B_AAAI/main.py --dataset='COLLAB'  --lr=0.001 --weight_decay=0.0005  --K_g=16 --K_v=32 --topk=8 --split_mode=high  

python /U2B_AAAI/main.py --dataset='IMDB-MULTI'  --lr=0.001 --weight_decay=0.0005  --K_g=24 --K_v=12 --topk=12 --split_mode=high --seed=8
python /U2B_AAAI/main.py --dataset='IMDB-MULTI'  --lr=0.001 --weight_decay=0.0005  --K_g=32 --K_v=48 --topk=24 --split_mode=low 

python /U2B_AAAI/main.py --dataset='PTC_MR'  --lr=0.001 --weight_decay=0.0005  --K_g=16 --K_v=32 --topk=8 --split_mode=high 
python /U2B_AAAI/main.py --dataset='PTC_MR'  --lr=0.001 --weight_decay=0.0005  --K_g=16 --K_v=48 --topk=8 --split_mode=low 

python /U2B_AAAI/main.py --dataset='IMDB-BINARY'  --lr=0.001 --weight_decay=0.0005  --K_g=24 --K_v=32 --topk=12 --split_mode=low  --seed=10
python /U2B_AAAI/main.py --dataset='IMDB-BINARY'  --lr=0.001 --weight_decay=0.0005  --K_g=32 --K_v=24 --topk=16 --split_mode=high  --seed=2

python /U2B_AAAI/main.py --dataset='REDDIT-BINARY'  --lr=0.001 --weight_decay=0.0005  --K_g=32 --K_v=32 --topk=16 --split_mode=high
python /U2B_AAAI/main.py --dataset='REDDIT-BINARY'  --lr=0.001 --weight_decay=0.0005  --K_g=32 --K_v=32 --topk=16 --split_mode=low  

