import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import time
from WordLoader import WordLoader
import torch.random
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
# import pysnooper

class AttentionLstm(nn.Module):
    def __init__(self, wordlist, argv, aspect_num=0):
        super(AttentionLstm, self).__init__()
        self.model_name = 'atae-lstm'
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='lstm')
        parser.add_argument('--rseed', type=int, default=int(1000 * time.time()) % 19491001)
        parser.add_argument('--dim_word', type=int, default=300)
        parser.add_argument('--dim_hidden', type=int, default=300)
        parser.add_argument('--dim_aspect', type=int, default=100)
        parser.add_argument('--grained', type=int, default=3, choices=[3])
        parser.add_argument('--regular', type=float, default=0.001)
        parser.add_argument('--word_vector', type=str, default='data/glove.840B.300d.txt')
        args, _ = parser.parse_known_args(argv)
        self.wordlist = wordlist
        self.name = args.name
        self.word_vector = args.word_vector
        torch.random.manual_seed(args.rseed)
        torch.cuda.manual_seed_all(args.rseed)
        torch.backends.cudnn.deterministic = True
        self.dim_word, self.dimh = args.dim_word, args.dim_hidden
        self.dim_aspect = args.dim_aspect
        # 输入是 aspect vector 和 word vector 堆起来的
        self.dim_lstm_para = self.dim_word + self.dim_aspect
        # 情感细粒度
        self.grained = args.grained
        # 正则化参数
        self.regular = args.regular
        self.num = len(wordlist) + 1
        # aspect的个数，等于dict_target的长度
        self.aspect_num = aspect_num

        # 给 lstm 传的 input 是 PackedSequence 实例,
        # 他可以通过把 batch 的 sequence 输给 pack_padded_sequence()函数，然后返回一个 PackedSequence 实例
        # pack_padded_sequence() 接受 input, length，
        # input can be of size T x B x * where T is the length of the longest sequence
        # 对长度未排序的 sequences , 设定enforce_sorted = False.
        # 如果 enforce_sorted = True, 这一个 batch 的 sequences 应该按序列长度降序
        # 然后把 PackedSequence 给 lstm，它就能返回所有时间步上的 hidden 了，
        # 所以就可以把它们堆起来，得到论文里的所有 hidden 组成的 H
        u = lambda x : 1 / np.sqrt(x)
        self.Ws = nn.Parameter(torch.rand((self.grained, self.dimh)).uniform_(-u(self.dimh), u(self.dimh)))
        self.bs = nn.Parameter(torch.zeros((self.grained, 1)))
        self.Wh = nn.Parameter(torch.rand((self.dimh, self.dimh)).uniform_(-u(self.dimh), u(self.dimh)))
        self.Wv = nn.Parameter(torch.rand((self.dim_aspect, self.dim_aspect)).uniform_(-u(self.dimh), u(self.dimh)))
        self.w = nn.Parameter(torch.zeros((self.dimh+self.dim_aspect, 1)))
        self.Wp = nn.Parameter(torch.rand((self.dimh, self.dimh)).uniform_(-u(self.dimh), u(self.dimh)))
        self.Wx = nn.Parameter(torch.rand((self.dimh, self.dimh)).uniform_(-u(self.dimh), u(self.dimh)))

        self.Va = torch.rand((self.aspect_num, self.dim_aspect)).uniform_(-0.01, 0.01)
        self.Vw = torch.rand((self.num, self.dim_word)).uniform_(-0.01, 0.01)
        self.load_word_vector(self.word_vector, self.wordlist)

        self.embedding = nn.Embedding.from_pretrained(self.Vw, freeze=False)
        # wordlist的单词索引从1开始，将0设为pad的索引，该 pad vector 全为0
        self.embedding.padding_idx = 0
        # aspect一共五个词，创建一个5个单词表的embedding，通过target_index得到aspect vector
        # {'price': 0, 'service': 1, 'miscellaneous': 2, 'ambience': 3, 'food': 4}
        self.aspect_embedding =nn.Embedding.from_pretrained(self.Va, freeze=False)
        self.lstm = nn.LSTM(self.dim_lstm_para, self.dimh)
        self.params = nn.ParameterList([self.Wv, self.Wh, self.Ws, self.bs, self.w, self.Wp, self.Wx])

        '''
        self.Ws = nn.Linear(self.dimh, self.grained)
        self.Wh = nn.Linear(self.dimh, self.dimh, bias=False)
        self.Wv = nn.Linear(self.dim_aspect, self.dim_aspect, bias=False)

        self.w = nn.Linear(self.dim_aspect+self.dimh, 1, bias=False)

        self.Wp = nn.Linear(self.dimh, self.dimh, bias=False)
        self.Wx = nn.Linear(self.dimh, self.dimh, bias=False)
        self.Va = torch.rand((self.aspect_num, self.dim_aspect)).uniform_(-0.01, 0.01)
        self.Vw = torch.zeros((self.num, self.dim_word)).uniform_(-0.01, 0.01)
        self.load_word_vector(self.word_vector, self.wordlist)

        self.embedding = nn.Embedding.from_pretrained(self.Vw, freeze=False)
        # wordlist的单词索引从1开始，将0设为pad的索引，该 pad vector 全为0
        self.embedding.padding_idx = 0
        # aspect一共五个词，创建一个5个单词表的embedding，通过target_index得到aspect vector
        # {'price': 0, 'service': 1, 'miscellaneous': 2, 'ambience': 3, 'food': 4}
        self.aspect_embedding = nn.Embedding.from_pretrained(self.Va, freeze=False)
        self.lstm = nn.LSTM(self.dim_lstm_para, self.dimh)
        # self.params = nn.ParameterList([self.Wv, self.Wh, self.Ws, self.bs, self.w, self.Wp, self.Wx])
        '''


    # @pysnooper.snoop()
    def forward(self, x, solution, aspect_word, aspect_level, train=True):
        x = x.view(-1, 1)
        solution = solution.view(1, 3)
        # x size = (N, 1, 300)  即 time_step * batch_size * dim_word
        x = self.embedding(x)
        aspect_vector = torch.LongTensor(np.tile(aspect_level, (x.size(0), 1)))

        try:
            # aspect_vector.size() = (N, 1, 100)
            aspect_vector = self.aspect_embedding(aspect_vector)
        except:
            print(aspect_level)
            raise ValueError

        lstm_input = torch.cat((x, aspect_vector), dim=2)
        packed_seqs = pack_padded_sequence(lstm_input, lengths=[lstm_input.size(0)])
        output, (h_n, c_n) = self.lstm(packed_seqs)
        # H.size = (300, N)
        H = torch.t(output.data)
        # h_n size = (300, 1)
        h_n = h_n.view(300, -1)
        # Wh_H.size = (300, N)
        Wh_H = torch.matmul(self.Wh, H)
        # Wv_a.size = (100, N)
        Wv_a = torch.matmul(self.Wv, aspect_vector.view(self.dim_aspect, -1))
        # M.size = (400, N)
        M = torch.tanh(torch.cat((Wh_H, Wv_a), dim=0))
        # alpha.size = (1, N)
        alpha = F.softmax(torch.matmul(torch.t(self.w), M), dim=1)
        # r.size = (300, 1)
        r = torch.matmul(H, torch.t(alpha))
        Wp_r = torch.matmul(self.Wp, r)
        Wx_h = torch.matmul(self.Wx, h_n)
        # h_star.size = (300, 1)
        h_star = torch.tanh(Wp_r + Wx_h)
        drop = nn.Dropout(p=0.5)
        h_star = drop(h_star)

        # y.size = (self.grained, 1)
        y = F.softmax(torch.matmul(self.Ws, h_star) + self.bs, dim=0)
        return torch.t(y)

#     @pysnooper.snoop()
    def forward_old(self, x, solution, aspect_word, aspect_level):
        x = x.view(-1, 1)
        solution = solution.view(1, 3)
        # x size = (N, 1, 300)  即 time_step * batch_size * dim_word
        x = self.embedding(x)
        # print('x', x.size())
        aspect_vector = torch.LongTensor(np.tile(aspect_level, (x.size(0), 1)))
        try:
            # aspect_vector.size() = (N, 1, 100)
            aspect_vector = self.aspect_embedding(aspect_vector)
        except:
            print(aspect_level)
            raise ValueError
        # print('aspect_vector', aspect_vector.size())
        lstm_input = torch.cat((x, aspect_vector), dim=2)
        # print('lstm_input', lstm_input.size())
        packed_seqs = pack_padded_sequence(lstm_input, lengths=[lstm_input.size(0)])
        output, (h_n, c_n) = self.lstm(packed_seqs)
        # print('output', output.data.size())
        # print('h_n', h_n.size())
        # h_n size = (1, 300)
        h_n = h_n.view(-1, 300)
        # output.data 即为 H size = (N, 300), Wh_H size = (N, 300)
        H = output.data
        Wh_H = self.Wh(H)
        # 把 aspect_vector 转成 (N, 100), 再和 Wv 相乘，和 Wh_H 堆起来，再 tanh 得出 M
        Wv_va = self.Wv(aspect_vector.view(-1, 100))
        # M size = (N, 400)
        M = torch.tanh(torch.cat((Wh_H, Wv_va), dim=1))
        # print('M', M.size())
        # alpha size = (N, 1)
        alpha = F.softmax(self.w(M),dim=0)
        # print('alpha', alpha.size())
        # r size = (1, 300)
        r = torch.matmul(torch.t(alpha), H)
        # print('r', r.size())
        # h_star size = (1, 300)
        h_star = torch.tanh(self.Wp(r) + self.Wx(h_n))
        # print('h_star', h_star.size())
        y = F.softmax(self.Ws(h_star), dim=1)
        # CEloss = nn.CrossEntropyLoss()
        # loss = CEloss(y, torch.argmax(solution, dim=1))
        return y




    # @pysnooper.snoop()
    def load_word_vector(self, fname, wordlist):
        loader = WordLoader()
        # dic为 'word' : [weights] 的字典, 这里传入的wordlist并没有用
        dic = loader.load_word_vector(fname, wordlist, self.dim_word)
        # print(dic.keys())
        not_found = 0
        # wordlist = {word1:1, word2:2, ...} index越小的单词出现次数越多
        for word, index in wordlist.items():
            try:
                # 按照wordlist的顺序，设置embedding层的weights，这样就可以根据索引把seqs变成对应词的weights了
                # index从1开始，Vw的len比vocab len多1，应该是留出了第一行做unkwon
                self.Vw[index] = torch.FloatTensor(dic[word])
            except:
                # raise ValueError
                not_found += 1