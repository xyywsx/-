#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 opengait/main.py --cfgs ./configs/smplgait/smplgaitgl.yaml --phase train
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 opengait/main.py --cfgs ./configs/gaitmbgl/gaitmbgl.yaml --phase train
