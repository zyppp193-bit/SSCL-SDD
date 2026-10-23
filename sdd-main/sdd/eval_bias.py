"""偏差累积分析：从逐任务 checkpoint 计算偏差指标，用于对比不同方法。
指标（每个任务结束时的模型上测量）：
  1. center_bias  类中心偏差：无标签特征中心与有标签特征中心的余弦距离（按类平均）
  2. imb_kl       伪标签失衡度：SNN(最近有标签类中心)分配结果的类别分布与均匀分布的KL散度
  3. sep_ratio    类间/类内可分性比（无标签特征）：越大越好
  4. drift        特征漂移：与上一任务模型在固定无标签样本上的平均余弦距离
用法: python eval_bias.py <config路径> <实验目录>
"""
import os
import sys
import glob
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from src.sdd_train import init_model

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_PER_CLASS = 500  # 每类取样的无标签样本数


def get_features(encoder, dataset, indices, bs=512):
    loader = DataLoader(Subset(dataset, indices), batch_size=bs,
                        shuffle=False, num_workers=4)
    feats, labels = [], []
    with torch.no_grad():
        for img, label in loader:
            img = img.to(DEVICE)
            h, _ = encoder._forward_backbone(img)  # 投影头输出空间(SNN工作空间)
            feats.append(h.float().cpu())
            labels.append(label)
    return torch.cat(feats), torch.cat(labels)


def cos_dist(a, b):
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def metrics_for_task(feat_l, lab_l, feat_u, lab_u, seen):
    centers_l, centers_u = {}, {}
    biases = []
    for c in seen:
        cl = feat_l[lab_l == c]
        cu = feat_u[lab_u == c]
        if len(cl) == 0 or len(cu) == 0:
            continue
        centers_l[c] = cl.mean(0)
        centers_u[c] = cu.mean(0)
        biases.append(cos_dist(centers_l[c].unsqueeze(0),
                               centers_u[c].unsqueeze(0)).item())
    center_bias = float(np.mean(biases))

    # 伪标签失衡：无标签样本分配到最近的有标签类中心
    keys = list(centers_l.keys())
    C = torch.stack([centers_l[c] for c in keys])
    sim = F.cosine_similarity(feat_u.unsqueeze(1), C.unsqueeze(0), dim=-1)
    assign = sim.argmax(1)
    counts = torch.stack([(assign == k).float().sum() for k in range(len(keys))])
    p = (counts / counts.sum()).clamp(min=1e-8)
    imb_kl = float((p * (p * len(p)).log()).sum())  # KL(p||uniform)

    # 类间/类内可分性比（无标签特征）
    intra, inter = [], []
    for c in keys:
        cu = feat_u[lab_u == c]
        intra.append(cos_dist(cu, centers_u[c].unsqueeze(0)).mean().item())
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            inter.append(cos_dist(centers_u[keys[a]].unsqueeze(0),
                                  centers_u[keys[b]].unsqueeze(0)).item())
    sep_ratio = float(np.mean(inter) / max(np.mean(intra), 1e-8))

    return center_bias, imb_kl, sep_ratio


def main():
    config_path, exp_folder = sys.argv[1], sys.argv[2]
    with open(config_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    num_classes = {'cifar10': 10, 'cifar100': 100}[params['data']['dataset']]
    num_tasks = params['continual']['num_tasks']
    cpt = num_classes // num_tasks
    tasks = [list(range(num_classes))[i:i + cpt] for i in range(0, num_classes, cpt)]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    root = os.path.join(params['data']['root_path'],
                        params['data']['image_folder'].rstrip('/'))
    train_set = datasets.CIFAR10(root=root, train=True, transform=transform,
                                 download=False)
    targets = np.array(train_set.targets)

    with open(params['data']['subset_path']) as f:
        labeled_idx = np.array([int(l.strip()) for l in f if l.strip()])
    labeled_set = set(labeled_idx.tolist())

    # 固定无标签样本池（所有类，跨任务固定，保证漂移指标可比）
    rng = np.random.RandomState(0)
    unlab_pool = {}
    for c in range(num_classes):
        cand = [x for x in np.where(targets == c)[0] if x not in labeled_set]
        unlab_pool[c] = rng.choice(cand, size=N_PER_CLASS, replace=False).tolist()

    def build_model(ckpt_path=None):
        m = init_model(
            device=DEVICE, model_name=params['meta']['model_name'],
            use_pred=params['meta']['use_pred_head'],
            output_dim=params['meta']['output_dim'],
            cifar=True, num_classes=num_classes,
            detach=params['continual']['detach'])
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            m.load_state_dict(ckpt['encoder'])
            del ckpt
        return m.eval()

    print(f'===== {exp_folder} =====')
    print(f'{"task":>4} {"center_bias":>12} {"imb_kl":>10} {"sep_ratio":>10} {"drift":>8}')
    prev_feats = None
    for i in range(num_tasks):
        seen = sum(tasks[:i + 1], [])
        ckpts = glob.glob(os.path.join(exp_folder, f'Task{i}', '*.pth.tar'))
        assert ckpts, f'no checkpoint for task {i}'
        encoder = build_model(sorted(ckpts)[-1])

        lidx = [x for x in labeled_idx.tolist() if targets[x] in seen]
        uidx = sum([unlab_pool[c] for c in seen], [])

        feat_l, lab_l = get_features(encoder, train_set, lidx)
        feat_u, lab_u = get_features(encoder, train_set, uidx)
        center_bias, imb_kl, sep_ratio = metrics_for_task(
            feat_l, lab_l, feat_u, lab_u, seen)

        allidx = sum([unlab_pool[c] for c in range(num_classes)], [])
        feat_all, _ = get_features(encoder, train_set, allidx)
        drift = float('nan') if prev_feats is None else cos_dist(feat_all, prev_feats).mean().item()
        prev_feats = feat_all

        print(f'{i:>4} {center_bias:>12.4f} {imb_kl:>10.4f} {sep_ratio:>10.4f} {drift:>8.4f}')
        del encoder
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
