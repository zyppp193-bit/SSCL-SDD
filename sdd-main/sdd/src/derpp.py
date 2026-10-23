# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer_attention import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
#from utils.scloss import SupConLoss
#from utils.crd import CRDLoss
from copy import deepcopy

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


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.teacher_model = deepcopy(self.net).to(self.device)
    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs,f_map = self.net(inputs,return_features=True)
        #print(f_map.shape)
        #print(type(f_map))
        #exit()
        #outputs_1=self.teacher_model()

        loss = self.loss(outputs, labels)


        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits,o_f_map = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs,buf_map = self.net(buf_inputs,return_features=True)
            #print(buf_map.shape, o_f_map.shape)
            #exit()
            #buf_p = self.teacher_model(buf_inputs)
            loss += self.args.alpha * F.smooth_l1_loss(buf_outputs, buf_logits)
            #loss += self.args.gamma * F.mse_loss(o_f_map, buf_map)
            #loss += 0.3* F.mse_loss(buf_p, buf_logits)
            #loss += 0.3 * F.mse_loss(buf_outputs, buf_p)
            buf_inputs, buf_labels, _,_ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)



        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             logits=outputs.data,
                             labels=labels,
                             f_map=f_map,
                             )

        return loss.item()

