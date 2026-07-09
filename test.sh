# SMPLGaitGL
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=1 opengait/main.py  --cfgs ./configs/smplgait/smplgaitgl.yaml --phase test

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=1 opengait/main.py  --cfgs ./configs/gaitmbgl/gaitmbgl.yaml --phase test
