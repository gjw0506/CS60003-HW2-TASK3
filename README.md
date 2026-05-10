# CS60003-HW2-TASK3
本项目为图像分割作业「从零搭建与损失函数工程：图像分割模型的像素级训练」的代码实现。
项目使用 PyTorch 从零手写实现经典 U-Net 语义分割网络，不使用任何预训练权重，并在 Stanford Background Dataset 上比较三种损失函数配置：
1. Cross-Entropy Loss
2. Dice Loss
3. Cross-Entropy Loss + Dice Loss

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── model_unet.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize_compare_3ckpts.py
├── data/
│   └── stanford_background/
│       ├── images/
│       └── labels/
├── checkpoints/
│   ├── best_ce.pth
│   ├── best_dice.pth
│   └── best_ce_dice.pth
├── vis_compare_topdiff/

````

---

## Environment Setup
```bash
conda create -n unet_seg python=3.10 -y
conda activate unet_seg
pip install torch torchvision torchaudio numpy pillow matplotlib tqdm wandb opencv-python
```

## Training

Before training with W&B logging, login to wandb:

```bash
wandb login

python train.py \
  --data_root data/stanford_background \
  --loss_type ce/dice/ce_dice \
  --num_classes 8 \
  --epochs 400 \
  --batch_size 8 \
  --lr 1e-3 \
  --run_name my_ce/my_dice/my_ce_dice
```

The best checkpoints will be saved to:

```text
checkpoints/best_ce.pth
checkpoints/best_dice.pth
checkpoints/best_ce_dice.pth
```


里面需要你自己改的地方主要有两个：
`Results` 表格如果后面结果更新了就同步修改；`Checkpoints` 部分如果权重太大，就换成你的网盘或 release 链接。
