import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.random
import argparse
from WordLoader import WordLoader
import time


class Lstm(nn.Module):
    def __init__(self, wordlist, argv, aspect_num=0):
        super(Lstm, self).__init__()
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
        self.dim_word, self.dimh = args.dim_word, args.dim_hidden
        self.grained = args.grained
        self.num = len(wordlist) + 1
        # aspect的个数，等于dict_target的长度
        self.aspect_num = aspect_num
        self.Vw = torch.rand((self.num, self.dim_word)).uniform_(-0.01, 0.01)
        self.load_word_vector(self.word_vector, self.wordlist)
        self.embedding = nn.Embedding.from_pretrained(self.Vw, freeze=False)
        # wordlist的单词索引从1开始，将0设为pad的索引，该 pad vector 全为0
        self.embedding.padding_idx = 0
        self.lstm = nn.LSTM(self.dim_word, self.dimh)
        self.Ws = nn.Linear(self.dimh, self.grained, bias=True)

    def forward(self, x, solution, aspect_word, aspect_level, train=True, test=False):
        x = x.view(-1, 1)
        solution = solution.view(1, 3)
        # x size = (N, 1, 300)  即 time_step * batch_size * dim_word
        x = self.embedding(x)
        # h_n size = (1, 300)
        output, (h_n, c_n) = self.lstm(x)
        h_n = h_n.view(-1, 300)

        y = self.Ws(h_n)
        y = F.softmax(y, dim=1)

        return y

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