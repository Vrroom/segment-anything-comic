python3 main.py \
	--lr 0.0001 \
	--batch_size 4 \
	--accumulate_grad_batches 2 \
	--epochs 100 \
	--gpus 1 \
	--seed 42 \
	--sam_ckpt_path ./checkpoints/sam_vit_h_4b8939.pth \
	--num_workers 8 \

