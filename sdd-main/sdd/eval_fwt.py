"""仅计算前向迁移 FWT 所需的数据：
- R[j-1][j]: 训练到任务 j-1 的模型在任务 j 上的零样本 2-way 准确率
- b_j: 随机初始化模型在任务 j 上的 2-way 基线
用法: python eval_fwt.py <config路径> <实验目录>
"""
import os
import sys
import glob
import yaml
import torch

from src.sdd_train import init_model
from src.data_manager import init_data, make_transforms
from src.utils import init_distributed


@torch.no_grad()
def eval_on_task(encoder, loader, cls_pair, device):
    correct, total = 0, 0
    for img, label in loader:
        mask = (label == cls_pair[0]) | (label == cls_pair[1])
        if mask.sum() == 0:
            continue
        img = img.to(device)
        _, logits = encoder._forward_backbone(img)
        sub = logits[:, cls_pair][mask]
        pred = sub.argmax(dim=1).cpu()
        target = (label[mask] == cls_pair[1]).long()
        correct += pred.eq(target).sum().item()
        total += mask.sum().item()
    return 100.0 * correct / total


def main():
    config_path, exp_folder = sys.argv[1], sys.argv[2]
    with open(config_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    init_distributed()
    device = torch.device(params['meta']['device'])
    torch.cuda.set_device(device)

    num_classes = {'cifar10': 10, 'cifar100': 100}[params['data']['dataset']]
    num_tasks = params['continual']['num_tasks']
    cpt = num_classes // num_tasks
    tasks = [list(range(num_classes))[i:i + cpt] for i in range(0, num_classes, cpt)]

    val_transform, val_init_transform = make_transforms(
        dataset_name=params['data']['dataset'] + '_fine_tune',
        subset_path=params['data']['subset_path'],
        unlabeled_frac=-1, training=False, basic_augmentations=False,
        force_center_crop=False, normalize=params['data']['normalize'])

    task_loaders = {}
    for j in range(num_tasks):
        loader, _ = init_data(
            dataset_name=params['data']['dataset'] + '_fine_tune',
            transform=val_transform, init_transform=val_init_transform,
            u_batch_size=None, s_batch_size=100, classes_per_batch=None,
            world_size=1, rank=0,
            root_path=params['data']['root_path'],
            image_folder=params['data']['image_folder'],
            training=False, copy_data=params['meta']['copy_data'],
            tasks=tasks, task_idx=j, visible_class_ul=tasks[j])
        task_loaders[j] = loader

    def fresh_model():
        return init_model(
            device=device, model_name=params['meta']['model_name'],
            use_pred=params['meta']['use_pred_head'],
            output_dim=params['meta']['output_dim'],
            cifar=True, num_classes=num_classes,
            detach=params['continual']['detach']).eval()

    base = {}
    rand_model = fresh_model()
    for j in range(num_tasks):
        base[j] = eval_on_task(rand_model, task_loaders[j], tasks[j], device)
    del rand_model
    torch.cuda.empty_cache()

    zero_shot = {}
    for j in range(1, num_tasks):
        i = j - 1
        ckpts = glob.glob(os.path.join(exp_folder, f'Task{i}', '*.pth.tar'))
        assert ckpts, f'no checkpoint for task {i}'
        encoder = fresh_model()
        checkpoint = torch.load(sorted(ckpts)[-1], map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder'])
        encoder.eval()
        del checkpoint
        zero_shot[j] = eval_on_task(encoder, task_loaders[j], tasks[j], device)
        del encoder
        torch.cuda.empty_cache()

    fwt_vals = [zero_shot[j] - base[j] for j in range(1, num_tasks)]
    fwt = sum(fwt_vals) / len(fwt_vals)

    print('===== 前向迁移 FWT（2-way 零样本定义） =====')
    for j in range(1, num_tasks):
        print(f'Task {j}: 零样本={zero_shot[j]:.2f} 随机基线={base[j]:.2f} 迁移={fwt_vals[j-1]:+.2f}')
    print(f'FWT 平均: {fwt:+.2f}')


if __name__ == '__main__':
    main()
