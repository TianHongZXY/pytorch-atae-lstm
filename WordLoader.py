from numpy import dtype, fromstring, float32
import pysnooper
class WordLoader(object):
    # @pysnooper.snoop()
    def load_word_vector(self, fname, wordlist, dim, binary=None):
        # 判断是不是二进制文件, 这里我们用的都是txt格式
        if binary == None:
            if fname.endswith('.txt'):
                binary = False
            elif fname.endswith('.bin'):
                binary = True
            else:
                raise NotImplementedError('Cannot infer binary from %s' % (fname))

        vocab = {}
        with open(fname) as fin:
            header = fin.readline()
            print(header)
            vocab_size, vec_size = map(int, header.split())
            if binary:
                binary_len = dtype(float32).itemsize * vec_size
                for line_no in range(vocab_size):
                    try:
                        word = []
                        while True:
                            ch = fin.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        vocab[unicode(word)] = fromstring(fin.read(binary_len), dtype=float32)
                    except:
                        pass
            else:
                # 把单词以及其对应的300维词向量存入字典
                # vocab = {'and':[-0.18567,
                #  0.066008,
                #  -0.25209,....], 'the':[。。。。。]}
                for line_no, line in enumerate(fin):
                    try:
                        parts = line.strip().split(' ')
                        if len(parts) != vec_size + 1:
                            print("Wrong line: %s %s\n" % (line_no, line))
                        # 在python3 map返回迭代器，这里我觉得应该要转成list，再赋给weights就是一个300维的浮点数向量了
                        word, weights = parts[0], map(float32, parts[1:])
                        # print(list(weights))
                        vocab[word] = list(weights)
                        # print(vocab)
                    except:
                        pass
        # print(vocab)
        return vocab