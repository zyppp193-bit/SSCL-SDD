# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Derppp(ContinualModel):
    NAME = 'derppp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derppp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        real_batch_size = inputs.shape[0]

        outputs1 = self.net(inputs)
        outputs2 = self.net(not_aug_inputs)

        loss1 = self.loss(outputs1, labels, reduction='none')
        loss2 = self.loss(outputs2, labels, reduction='none')

        loss_scores = 0.5 * loss1 + 0.5 * loss2
        # print(loss_scores)
        loss = loss_scores.mean()

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits, buf_indexes = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_indexes=True)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.smooth_l1_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _, buf_indexes = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_indexes=True)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

            _, buf_labels, buf_logits, buf_indexes = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_indexes=True)
            loss += self.args.gamma * self.loss(buf_logits, buf_labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs1.data)

        return loss.item()
