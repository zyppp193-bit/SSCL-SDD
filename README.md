# SDD: Semi-supervised Continual Learning with Contrastive Representation Learning

Official PyTorch implementation of **SDD**, a semi-supervised continual learning method for image classification. SDD is built on top of the [PAWS](https://github.com/facebookresearch/suncet) self-supervised framework and the continual-learning pipeline of [NNCSL](https://github.com/kangzhiq/NNCSL), and extends them with:

- **Supervised contrastive learning** (`SupConLoss`) applied to unlabeled views, using the soft pseudo-label probabilities produced by the PAWS nearest-neighbor classifier as grouping signals;
- **Experience-replay memory**: a reservoir buffer that stores unlabeled samples together with their feature responses, replayed with a feature-consistency (smooth-L1) loss to mitigate catastrophic forgetting of unlabeled knowledge.

## Requirements

- Python 3.8+
- PyTorch >= 1.7 (developed with 1.9.1 / tested with 2.6)
- torchvision, numpy, PyYAML, tqdm

## Data preparation

1. Download CIFAR-10/CIFAR-100 and place them under `datasets/` (e.g. `datasets/cifar10/`).
2. Generate labeled subsets (per-class splits listing the labeled sample indices), e.g. `subsets/cifar10/spc.40_percent0.008_split0.txt`, and point `data.subset_path` / `data.subset_path_cls` in the config to them.

## Training

```bash
python main.py --sel sdd_train --fname configs/sdd/cifar10/cifar10_0.8%_buffer5120_sdd.yaml --devices cuda:0
```

Checkpoints are saved per task under `experiments/<folder>/Task<i>/`. To resume from task `t`, set `meta.load_checkpoint: true` and `meta.start_task_idx: t` in the config.

## Evaluation

After training, the following scripts compute continual-learning metrics from the per-task checkpoints:

```bash
# accuracy matrix R[i,j], per-task accuracy, forgetting, final accuracy
python eval_results.py <config> <experiments_folder>

# forward transfer (FWT) with a random-init baseline
python eval_fwt.py <config> <experiments_folder>

# bias-accumulation diagnostics: class-center bias, pseudo-label imbalance,
# class separability ratio, feature drift
python eval_bias.py <config> <experiments_folder>
```

## Code structure

```
├── main.py                 # entry point (--sel sdd_train)
├── src/
│   ├── sdd_train.py        # training script (SDD)
│   ├── losses.py           # PAWS loss + me-max regularizer + SNN classifier
│   ├── scloss.py           # SupCon / InfoNCE losses
│   ├── buffer.py           # reservoir replay buffer
│   ├── resnet.py           # ResNet backbone (with projection & pred heads)
│   ├── data_manager.py     # datasets, samplers, transforms, buffer list
│   ├── utils.py            # distributed helpers, schedules, buffer-list utils
│   ├── sgd.py / lars.py    # optimizers
│   ├── buffer_select1.py   # (reference) score-based buffer variant, unused
│   ├── derpp.py/derppp.py  # (reference) DER++ implementations, unused
│   └── conf.py             # device/seed helpers
├── configs/sdd/            # config files per dataset / label ratio / buffer size
└── eval_{results,fwt,bias}.py
```

## Acknowledgement & License

The code is developed based on the source code of [PAWS](https://github.com/facebookresearch/suncet) (MIT) and [NNCSL](https://github.com/kangzhiq/NNCSL). Released under the MIT License (see `LICENSE`).

## Citation

If you find this code useful, please cite:

```bibtex
@article{sdd,
  title   = {SDD: Semi-supervised Continual Learning with Contrastive Representation Learning},
  author  = {},
  year    = {2026}
}
```
