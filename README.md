# G2P-DDM
[Project Link](https://slpdiffusier.github.io/g2p-ddm/) | [Demo Link](https://slpdiffusier.github.io/g2p-ddm/) | [Supplementary Materials](https://github.com/Anynoumsiccv9970/G2P-DDM/blob/main/ICCV9970_supplementary.pdf)

This repository contains the implementation of our paper "G2P-DDM: Generating Sign Pose Sequence from Gloss Sequence with Discrete Diffusion Model".

# requirements

```bash
pip install -r requirements.txt
```

# Traning

## Stage 0: Prepare Data
Prepare the data with instructions on [ProgressiveTransformerSLP](https://github.com/BenSaunders27/ProgressiveTransformersSLP/tree/master)

## Stage 1: Pose-VQVAE for Reconstruction
![image](imgs/vae_2.png)

```bash
python3 -m train_pose_vqvae \
    --gpus 8 --gpu_ids "0,1,2,3,4,5,6,7" \
    --init_lr 2e-4 \
    --embedding_dim 128 \
    --batchSize 12 \
    --n_codes 1024 \
    --data_path "Data/ProgressiveTransformersSLP" \
    --vocab_file "Data/ProgressiveTransformersSLP/src_vocab.txt" \
    --resume_ckpt "" \
    --default_root_dir "experiments/pose_vqvae/separate" \
    --max_steps 300000 \
    --max_frames_num 300 \
```

## Stage 1.1: Generating '.leng' files
Execute the following code for train, dev and test data
``` bash
python3 -m modules.sequential_kmeans \
    --mode "train" \
    --embedding_dim 128 \
    --n_codes 1024 \
    --pose_vqvae "experiments/pose_vqvae/separate/lightning_logs/version_2/checkpoints/epoch=502-step=297776-val_wer=0.0000-val_rec_loss=0.0647-val_ce_loss=0.0000.ckpt" \
    --data_path "datasets/phoenix-14t/processed_dataset/" \
    --vocab_file "datasets/phoenix-14t/processed_dataset/src_vocab.txt" \
    --num_workers 24
```

## Stage 1.2: Training backtranslate model pose2text 
This is required for evaluating text2pose prediction quality
``` bash
python3 -m train_backtranslate \
    --gpus 1 \
    --gpu_ids "0" \
    --batchSize 2 \
    --data_path "datasets/phoenix-14t/processed_dataset/" \
    --vocab_file "datasets/phoenix-14t/processed_dataset/src_vocab.txt" \
    --resume_ckpt "" \
    --default_root_dir "experiments/backmodel" \
    --max_steps 300000 \
    --max_frames_num 300 \
    --num_workers 24
```

## Stage 2: Discrete Diffusion Model for Latent Prior Learning
![image](imgs/diffusion.png)
```bash
python3 -m train_text2pose --gpus 4 --gpu_ids "0,1,2,3" \
    --stage2_model "configs/stage2_model/vq_diffusion_codeunet.yaml"  \
    --default_root_dir "experiments/text2pose/vq_diffusion_codeunet"
```

# Inference

```bash
python3 -m train_text2pose --gpus 8 --gpu_ids "0,1,2,3,4,5,6,7" \
    --stage2_model "configs/stage2_model/vq_diffusion_codeunet.yaml"  \
    --default_root_dir "experiments/text2pose/test"
```
