import torch
import pickle
import argparse

# from pytorch_transformers import BertTokenizer, RobertaTokenizer
from transformers import AutoTokenizer
from torchtext.data import Field, TabularDataset
from torchtext.data import BucketIterator
import os

# 方法1：在代码中设置环境变量
# os.environ['HF_ENDPOINT'] = r'https://hf-mirror.com'  # 国内常用镜像

class Dataset(object):
    def __init__(self, args, params):
        self.batch_size = params.batch_size
        self.fix_length = params.fix_length
        self.root_path = args.data_dir
        self.use_distilbert = args.distilbert
        self.pad_token = 0
        if self.use_distilbert:
            self.tokenizer = AutoTokenizer.from_pretrained(r'D:\\distilbert_cache\\models--distilbert-base-uncased',
                                                           local_files_only=True)
            # 记录DistilBERT特殊符号ID（便于后续维护）
            self.cls_id = self.tokenizer.cls_token_id  # 0
            self.sep_id = self.tokenizer.sep_token_id  # 2
            self.pad_id = self.tokenizer.pad_token_id  # 1
        else:
            with open(args.embedding_pkl_path + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)

        def word_tokenize(sentence):
            if self.use_distilbert:
                # tokenized_text = self.tokenizer.tokenize(sentence)    #会切分单词 导致不对应
                # 这里不是按照传统的tokenizer分词，而是按照空格切分，我们的研究最终验证的是关系抽取耦合架构的有效性，而不是词嵌入的有效性
                tokenized_text = sentence.split(' ')
                sentence = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                # # sentence = tokenizer.add_special_tokens_single_sentence(sentence)
                # # 添加[CLS]和[SEP] 或 <s> </s>
                sentence = [self.cls_id] + sentence + [self.sep_id]
                # 使用tokenizer正确分词
                return sentence
            else:
                tokenized_text = sentence.split(' ')
                sentence = [word2idx.get(word, 0) for word in tokenized_text]
                sentence = [0] + sentence  # 与bert统一
            return sentence

        def pos_tokenize(posids):
            return [int(_) for _ in posids.split(' ')]

        TEXT = Field(sequential=True, tokenize=word_tokenize,
                     use_vocab=False, batch_first=True,
                     fix_length=self.fix_length + 2,  # 添加了 cls和sep 或 <s> </s>
                     pad_token=self.pad_token) # bert是0， roberta是1
        POSITION = Field(sequential=True, tokenize=pos_tokenize, use_vocab=False, fix_length=self.fix_length,
                         pad_token=0, batch_first=True, include_lengths=True)
        POSITION_NO_LEN = Field(sequential=True, tokenize=pos_tokenize, use_vocab=False, fix_length=self.fix_length,
                                pad_token=0, batch_first=True)
        LABEL = Field(sequential=False, use_vocab=False, batch_first=True)
        
        # semeval: e1,e2, prompt: entity1,entity2
        fields = {
            'sentence': ('words', TEXT),
            'label': ('label', LABEL),
            'e1': ('pos_e1', POSITION),
            'e2': ('pos_e2', POSITION_NO_LEN)
        }
        
        # 训练集：验证集：测试集=7：1：2
        self.train, self.valid, self.test = TabularDataset.splits(
            path=self.root_path,
            train='train.txt', 
            validation='val.txt',
            test="test.txt",
            format='json',
            skip_header=False,
            fields=fields
        )

    def get_data(self, name='training'):
        if name == 'training':
            return BucketIterator(self.train, batch_size=self.batch_size, shuffle=True)
        elif name == 'validation':
            return BucketIterator(self.valid, batch_size=self.batch_size, shuffle=False)
        elif name == 'test':
            return BucketIterator(self.test, batch_size=self.batch_size, shuffle=False)


class BatchWrapper(object):
    """对batch做个包装，方便调用，可选择性使用"""

    def __init__(self, dl, gpu, pad_token=0):
        self.dl = dl
        self.gpu = gpu  # 是否使用gpu
        self.pad_token = pad_token # 接收模型的pad_token

    def __iter__(self):
        for batch in self.dl:
            words = getattr(batch, 'words')
            labels = getattr(batch, 'label')

            pos1s = getattr(batch, 'pos_e1')[0]
            lens = getattr(batch, 'pos_e1')[1]
            pos2s = getattr(batch, 'pos_e2')

            # sdp = getattr(batch, 'sdp')
            # sdp_id = getattr(batch, 'sdp_id')
            # oov = getattr(batch, 'oov')

            func = lambda x: x.cuda()
            if not self.gpu:
                yield [words, pos1s, lens, pos2s, labels]
            else:
                yield list(map(func, [words, pos1s, lens, pos2s, labels]))

    def __len__(self):
        return len(self.dl)
