import numpy as np
from numpy import float32
import torch
from torch.utils import data
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence


class Sentence(object):
    """docstring for sentence"""
    def __init__(self, content, target, rating, grained):
        # 小写化句子
        self.content, self.target = content.lower(), target
        # grained是细粒度情感，共3种 positive，negative，neutral 1，-1， 0
        self.solution = torch.zeros(grained, dtype=torch.int64)
        # 句子长度
        self.senlength = len(self.content.split(' '))
        try:
            # solution的index对应0,1,2的位置分别对应 negative -1, neutral 0, positive 1
            # 如 rating = 1 即 positive，则 solution = [0, 0, 1]
            # solution为真实的polarity vector
            self.solution[int(rating)+1] = 1
        except:
            exit()

    def stat(self, target_dict, wordlist, grained=3):
        data, data_target, i = [], [], 0
        # solution.shape = (senlength, 3), 输入一句话，预测出三种情感的数值
        # 其实此处的solution没有用，它并非self.solution，因为它根本没有被调用的地方
        solution = torch.zeros((self.senlength, grained), dtype=torch.float32)
        # 把一句话的每个单词在wordlist里的映射添加到data列表里
        for word in self.content.split(' '):
            data.append(wordlist[word])
            try:
                # Lexicons_dict应该包含真实的 aspect-level-word 的 polarity
                # 用 try 来 找出该句的 aspect word，pol 为该 aspect 在该句中的polarity
                # pol = -1 / 0 / 1
                # 不过有的句子并不包含真的 aspect level word 比如 It's cheap. 没有对应的aspect 'price' 这个词
                pol = Lexicons_dict[word]
                # 预测结果中句子的第i个单词的polarity = 1
                solution[i][pol+1] = 1
            except:
                pass
            i = i+1
        # target就是aspect level word
        for word in self.target.split(' '):
            # 与映射句子的单词类似，添加到data_target列表
            data_target.append(wordlist[word])
        return {'seqs': torch.LongTensor([data]), 'target': torch.LongTensor([data_target]),
                'solution': self.solution, 'target_index': self.get_target(target_dict)}

    def get_target(self, dict_target):
        return dict_target[self.target]

class DataManager(object):

    def __init__(self, dataset, grained=3, train=True, val=False, test=False):
        self.fileList = ['train', 'test', 'dev']
        self.origin = {}
        self.train = train
        self.val = val
        self.test = test

        for fname in self.fileList:
            data = []
            # 读取数据
            with open('%s/%s.cor' % (dataset, fname)) as f:
                sentences = f.readlines()
                # 因为每三行为一个数据，第一行为sentence，第二行为aspect level word，第三行为polarity
                for i in range(len(sentences)//3):
                    content, target, rating = sentences[i*3].strip(), sentences[i*3+1].strip(), sentences[i*3+2].strip()
                    # 每个 Sentecne 其实就是一个数据的实例，包含着评论，aspect level word 和对应的 polarity
                    sentence = Sentence(content, target, rating, grained)
                    data.append(sentence)
            # origin字典保存着 train, test, dev 为键的预处理过的数据，
            # 例如 train 数据保存着训练集的每一句话的单词在wordlist中对应的 index 组成的 vector，
            # 以及该句话的 aspect level word 即 target，以及该 aspect 的 polarity 保存在 solution 里
            self.origin[fname] = data
        # 这里生成了 dict_target, 保存在了self.dict_target里
        self.dict_target = self.gen_target()
        self.gen_word()
        # self.train_data, self.dev_data, self.test_data = self.gen_data()
        self.gen_data()
    def __getitem__(self, index):
        return self.data['train'][index]

    def gen_target(self, threshold=5):
        self.dict_target = {}
        # 遍历每个 train, dev, test 数据集
        for fname in self.fileList:
            # 遍历每一个 Sentence
            for sent in self.origin[fname]:
                # 如果dict_target 中包含了该 aspect level word，则该单词对应的值 +1
                if self.dict_target.get(sent.target)is not None:
                    self.dict_target[sent.target] = self.dict_target[sent.target] + 1
                else:
                    self.dict_target[sent.target] = 1
        i = 0
        # key = aspect level word, val = 出现次数
        for (key,val) in self.dict_target.items():
            # 如果出现次数少于5，就将对应的index设为0，不过如果第一个单词出现大于5次，不也被设为0了吗？
            if val < threshold:
                self.dict_target[key] = 0
            else:
                self.dict_target[key] = i
                i = i + 1
        return self.dict_target

    def gen_word(self):
        wordcount = {}
        def sta(sentence):
            for word in sentence.content.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1
            for word in sentence.target.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1

        for fname in self.fileList:
            for sent in self.origin[fname]:
                sta(sent)
        # words = [(word1, count1), (word2, count2), ...] (其实是生成器，这里为了方便理解写成list)
        words = wordcount.items()
        # 按出现次数降序排列
        words = sorted(words, key=lambda x:x[1], reverse=True)
        # wordlist = {word1:1, word2:2, ...} index越小的单词出现次数越多
        self.wordlist = {item[0]:index+1 for index, item in enumerate(words)}
        return self.wordlist


    def gen_data(self, grained=3):
        self.data = {}
        for fname in self.fileList:
            # data字典保存着以 train, dev, test为键的数据列表
            # 例如 train列表包含着一句sentence的数据，
            # 包含了未对单词embedding的句向量seq，target为用来aspect embedding的单词，solution为情感极性的one-hot编码，target_index为aspect所属类别
            self.data[fname] = []
            for sent in self.origin[fname]:
                self.data[fname].append(sent.stat(self.dict_target, self.wordlist))
        return self.data['train'], self.data['dev'], self.data['test']

    def word2vec_pre_select(self, mdict, word2vec_file_path, save_vec_file_path):
        list_seledted = ['']
        line = ''
        with open(word2vec_file_path) as f:
            for line in f:
                tmp = line.strip().split(' ', 1)
                if mdict.has_key(tmp[0]):
                    list_seledted.append(line.strip())
        list_seledted[0] = str(len(list_seledted)-1) + ' ' + str(len(line.strip().split())-1)
        open(save_vec_file_path, 'w').write('\n'.join(list_seledted))

class Dataset(data.Dataset):

    def __init__(self, root_path):
        '''划分数据集'''
        train = pd.read_csv(root_path)
        self.seqs_data = []
        self.solutions_data = []
        self.targets_data = []
        self.target_index_data = []

        for seqs in train['seqs']:
            self.seqs_data.append(eval(seqs))
        for solution in train['solution']:
            self.solutions_data.append(eval(solution))
        for target in train['target']:
            self.targets_data.append(eval(target))
        for target_index in train['target_index']:
            self.target_index_data.append(target_index)

    def __getitem__(self, index):
        return torch.LongTensor(self.seqs_data[index]), torch.LongTensor(self.solutions_data[index]), \
               torch.LongTensor(self.targets_data[index]), self.target_index_data[index]

    def __len__(self):
        return len(self.seqs_data)
if __name__ == '__main__':
    dataset = Dataset('train.csv')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    for ii, (seqs, solutions,targets, target_indexes) in enumerate(dataloader):
        seqs = pack_padded_sequence(seqs)

        # print('seqs', seqs, 'solutions\n', solutions, 'targets\n', targets, 'target_indexes\n', target_indexes)
        print(solutions)
        break

