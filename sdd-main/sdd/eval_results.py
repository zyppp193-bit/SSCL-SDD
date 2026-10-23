"""训练结束后评估：加载每个任务的 checkpoint，构建准确率矩阵 R[i,j]，
计算逐任务准确率、平均遗忘率和最终平均准确率。"""
import os
import glob
import yaml
import torch

from src.sdd_train import init_model, validate
from src.data_manager import init_data, make_transforms
from src.utils import init_distributed

CONFIG = 'configs/sdd/cifar10/cifar10_0.8%_buffer5120_sdd.yaml'


def main():
    with open(CONFIG, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    init_distributed()
    device = torch.device(params['meta']['device'])
    torch.cuda.set_device(device)

    dataset_name = params['data']['dataset']
    num_classes = {'cifar10': 10, 'cifar100': 100}[dataset_name]
    num_tasks = params['continual']['num_tasks']
    classes_per_task = num_classes // num_tasks
    classes = list(range(num_classes))
    tasks = [classes[i:i + classes_per_task] for i in range(0, len(classes), classes_per_task)]

    folder = params['logging']['folder']

    val_dataset_name = dataset_name + '_fine_tune'
    val_transform, val_init_transform = make_transforms(
        dataset_name=val_dataset_name,
        subset_path=params['data']['subset_path'],
        unlabeled_frac=-1,
        training=False,
        basic_augmentations=False,
        force_center_crop=False,
        normalize=params['data']['normalize'])

    # R[i][j]: 任务 i 训练结束后，在任务 j 测试集上的 acc1
    R = {}
    final_overall = None
    for i in range(num_tasks):
        ckpts = glob.glob(os.path.join(folder, f'Task{i}', '*.pth.tar'))
        assert ckpts, f'no checkpoint for task {i}'
        ckpt_path = sorted(ckpts)[-1]

        encoder = init_model(
            device=device,
            model_name=params['meta']['model_name'],
            use_pred=params['meta']['use_pred_head'],
            output_dim=params['meta']['output_dim'],
            cifar=True,
            num_classes=num_classes,
            detach=params['continual']['detach'])
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder'])
        encoder.eval()
        del checkpoint

        seen_classes = sum(tasks[:i + 1], [])
        val_loader, _ = init_data(
            dataset_name=val_dataset_name,
            transform=val_transform,
            init_transform=val_init_transform,
            u_batch_size=None,
            s_batch_size=100,
            classes_per_batch=None,
            world_size=1,
            rank=0,
            root_path=params['data']['root_path'],
            image_folder=params['data']['image_folder'],
            training=False,
            copy_data=params['meta']['copy_data'],
            tasks=tasks,
            task_idx=i,
            visible_class_ul=seen_classes)

        acc1, acc5, per_task = validate(val_loader, encoder, device,
                                        len(seen_classes), task_id=i, tasks=tasks)
        R[i] = per_task
        final_overall = (acc1, acc5)
        print(f'ckpt: {ckpt_path}')
        print(f'after task {i}: overall acc1={acc1:.2f} acc5={acc5:.2f}, per-task={per_task}')
        del encoder
        torch.cuda.empty_cache()

    last = num_tasks - 1
    print('\n===== 准确率矩阵 R[i][j] (行: 训练到任务 i, 列: 任务 j 的 acc1) =====')
    header = '        ' + ''.join(f'Task{j:>4}' for j in range(num_tasks))
    print(header)
    for i in range(num_tasks):
        row = f'Task{i}   '
        for j in range(num_tasks):
            row += f'{R[i][j]:7.2f}' if j in R[i] else '      -'
        print(row)

    final_per_task = {j: R[last][j] for j in range(num_tasks)}
    print('\n===== 逐任务最终准确率 =====')
    for j, acc in final_per_task.items():
        print(f'Task {j} (classes {tasks[j]}): {acc:.2f}')

    # 平均遗忘率: 对每个旧任务 j, max_{i>=j} R[i][j] - R[last][j] 的平均
    forgettings = []
    for j in range(last):
        best = max(R[i][j] for i in range(j, num_tasks))
        forgettings.append(best - R[last][j])
    avg_forgetting = sum(forgettings) / len(forgettings)
    print('\n===== 遗忘率 =====')
    for j, fgt in enumerate(forgettings):
        best = max(R[i][j] for i in range(j, num_tasks))
        print(f'Task {j}: best={best:.2f} final={R[last][j]:.2f} forgetting={fgt:.2f}')
    print(f'平均遗忘率: {avg_forgetting:.2f}')

    final_avg = sum(final_per_task.values()) / num_tasks
    print('\n===== 最终准确率 =====')
    print(f'最终逐任务平均 acc1: {final_avg:.2f}')
    print(f'最终整体 acc1/acc5: {final_overall[0]:.2f} / {final_overall[1]:.2f}')


if __name__ == '__main__':
    main()
