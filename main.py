import torch
import torch.nn as nn
import torch.random
from lstm_att_con import AttentionLstm as Model
# from lstm import Lstm as Model
from DataManager import DataManager
import numpy as np
import argparse
import sys
import time
import torch.optim as optim
import json
from tqdm import tqdm
# import pysnooper
import torch.nn.functional as F
import copy


# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cuda'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self):
        self.weight_list = self.get_weight(self.model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

def compute_loss(pred, label):
    loss = -torch.mul(torch.log(pred), label.float())
    # return torch.sum(loss)
    return loss

# @pysnooper.snoop()
def train(model, train_data, batch_size, batch_n):

    #for param in model.named_parameters():
    #    print(param[0])
    #print(type(model.Ws))
    #print(model.lstm.all_weights[0])
    #
    # raise ValueError
    # paramslist = []
    # paramslist.append({'params':model.aspect_embedding.weight, 'lr':0.1})
    # [{'params': filter(notaspect_embedding(), model.parameters())}]
    # optimizer = optim.Adagrad(model.parameters(),lr=3e-3, weight_decay=1e-3)
    # model = Model()
    #optimizer = optim.SGD([
    #            {'params': model.aspect_embedding.weight, 'lr':1e-3},
    #            {'params': [model.Ws, model.bs, model.Wh, model.Wv, model.w, model.Wp, model.Wx,
    #                        model.embedding.weight, model.lstm.weight_ih_l0, model.lstm.weight_hh_l0,
    #                        model.lstm.bias_ih_l0, model.lstm.bias_hh_l0]}
    #        ], lr=1e-3, weight_decay=1e-3, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9)
    # scheduler = optim.CyclicLR(optimizer, step_size_up=500)
    # optimizer = optim.Adam(model.parameters(),lr=1e-2, eps=1e-10, weight_decay=1e-3)
    epoch_loss = 0
    # l2_loss = Regularization(model, weight_decay=1e-3, p=2).forward()
    correct = 0

    for batch in tqdm(range(batch_n)):
        batch_loss = 0
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(train_data))
        data_bunch = train_data[start:end]
        for data in data_bunch:
            # print(data)
            seqs = data['seqs']
            solution = data['solution']
            # target 是 aspect level 在总单词表的索引
            aspect_level_in_vocab_index = data['target']
            # target_index 是 五个 aspect level {'price': 0, 'service': 1, 'miscellaneous': 2, 'ambience': 3, 'food': 4}
            aspect_level_five = data['target_index']
            # print(seqs, aspect_word_index, aspect_level_index, solution)
#            raise ValueError
            y = model(seqs, solution, aspect_level_in_vocab_index, aspect_level_five, train=True)
            one_step_loss = compute_loss(y, solution)
            epoch_loss += one_step_loss
            batch_loss += torch.sum(one_step_loss)
            total_loss = torch.sum(one_step_loss)
            total_loss.backward()
            grad_for_perturb = copy.deepcopy(model.embedding.weight.grad.data)

            optimizer.zero_grad()
            # print(grad_for_perturb)
            perturb = F.normalize(grad_for_perturb, p=2, dim=1) * 5.0  # 5 is the norm of perturbation. Hyperparam.
            model.embedding.weight.data += perturb
            y = model(seqs, solution, aspect_level_in_vocab_index, aspect_level_five, train=True)
            one_step_loss = compute_loss(y, solution)
            total_loss = torch.sum(one_step_loss)
            total_loss.backward()
            if torch.argmax(y) == torch.argmax(solution):
                correct += 1

            # print(epoch_loss)
            optimizer.step()
        #
        # if batch%20 == 0:
        #     print('pred', y, '\nlabel', solution)
        #     print('batch loss:', batch_loss)

    acc = correct / len(train_data)
    return epoch_loss, acc


def test(model, test_data):
    loss = 0
    correct = 0
    node = 0
    with torch.no_grad():
        for data in test_data:
            # print(data)
            seqs = data['seqs']
            solution = data['solution']
            # target 是 aspect level 在总单词表的索引
            aspect_level_in_vocab_index = data['target']
            # target_index 是 五个 aspect level {'price': 0, 'service': 1, 'miscellaneous': 2, 'ambience': 3, 'food': 4}
            aspect_level_five = data['target_index']
            # print(seqs, aspect_word_index, aspect_level_index, solution)
            #            raise ValueError
            y = model(seqs, solution, aspect_level_in_vocab_index, aspect_level_five, train=False)
            loss += compute_loss(y, solution)
            node += len(solution)
            # torch.save()
            if torch.argmax(y) == torch.argmax(solution):
                correct += 1

    acc = correct / len(test_data)
    return loss/node, acc

if __name__ == '__main__':
    import os
    #print(os.listdir('../result'))
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='lstm')
    parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--dim_gram', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--fast', type=int, choices=[0, 1], default=0)
    parser.add_argument('--screen', type=int, choices=[0, 1], default=0)
    parser.add_argument('--optimizer', type=str, default='ADAGRAD')
    parser.add_argument('--grained', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_word_vector', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--batch', type=int, default=25)
    args, _ = parser.parse_known_args(argv)

    torch.random.manual_seed(args.seed)
    datamanager = DataManager(args.dataset, train=True)
    wordlist = datamanager.gen_word()
    train_data, val_data, test_data = datamanager.gen_data()
    model = Model(wordlist, argv, len(datamanager.dict_target))
    batch_n = (len(train_data)-1) // args.batch + 1

    details = {'acc_train': [], 'acc_dev': [], 'acc_test': []}

    for epoch in range(args.epoch):
        np.random.shuffle(train_data)
        now = {}
        now['loss'], now['acc_train'] = train(model, train_data, batch_size=args.batch, batch_n=batch_n)
        now['sum_loss'] = torch.sum(now['loss'])
        _, now['acc_dev'] = test(model, val_data)
        _, now['acc_test'] = test(model, test_data)
        # now['sum_loss_dev'] = torch.sum(now['loss_dev'])
        print(now)

        for key, value in now.items():
            try:
                details[key].append(value)
            except:
                pass

        with open('../result/%s.txt' % 'pytorch_lstm_adv', 'w') as f:
           f.writelines(json.dumps(details))
