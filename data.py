import torch
import pickle
import argparse

from pytorch_transformers import BertTokenizer
from torchtext.data import Field, TabularDataset
from torchtext.data import BucketIterator


class Dataset(object):
    def __init__(self, args, params):
        self.batch_size = params.batch_size
        self.fix_length = params.fix_length
        self.root_path = args.data_dir
        self.use_bert = args.bert
        # self.sdp_length = params.sdp_length

        if not self.use_bert:
            with open(args.embedding_pkl_path + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
        else:
            # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # tokenizer = BertTokenizer.from_pretrained('./bert_uncased_128/')

        def word_tokenize(sentence):
            if self.use_bert:
                # tokenized_text = tokenizer.tokenize(sentence)    #会切分单词 导致不对应
                tokenized_text = sentence.split(' ')
                sentence = tokenizer.convert_tokens_to_ids(tokenized_text)
                # sentence = tokenizer.add_special_tokens_single_sentence(sentence)
                # 添加[CLS]和[SEP]
                sentence = [101] + sentence +[102]
            else:
                tokenized_text = sentence.split(' ')
                sentence = [word2idx.get(word, 0) for word in tokenized_text]
                sentence = [0] + sentence  # 与bert统一
            return sentence
        
        # def oov_tokenize(oov):
        #     return eval(oov)

        def pos_tokenize(posids):
            return [int(_) for _ in posids.split(' ')]
        
        # def sdp_tokenize(sdp):
        #     if self.use_bert:
        #         tokenized_text = sdp.split(' ')
        #         sdp = tokenizer.convert_tokens_to_ids(tokenized_text)
        #         sdp = [101] + sdp +[102]
        #     else:
        #         tokenized_text = sdp.split(' ')
        #         sdp = [word2idx.get(word, 0) for word in tokenized_text]
        #         sdp = [0] + sdp  # 与bert统一
        #     return sdp
        
        # def sdp_id_tokenize(sdp_id):
        #     return eval(sdp_id)

        # dtype = torch.cuda.LongTensor if args.gpu and torch.cuda.is_available() else torch.int64

        TEXT = Field(sequential=True, tokenize=word_tokenize,
                     use_vocab=False, batch_first=True,
                     fix_length=self.fix_length + 2,  # 添加了 cls和sep
                     pad_token=0)
        POSITION = Field(sequential=True, tokenize=pos_tokenize, use_vocab=False, fix_length=self.fix_length,
                         pad_token=0, batch_first=True, include_lengths=True)
        POSITION_NO_LEN = Field(sequential=True, tokenize=pos_tokenize, use_vocab=False, fix_length=self.fix_length,
                                pad_token=0, batch_first=True)
        LABEL = Field(sequential=False, use_vocab=False, batch_first=True)
        # SDP = Field(sequential=True, tokenize=sdp_tokenize,
        #             use_vocab=False, batch_first=True,
        #             fix_length=self.sdp_length + 2,
        #             pad_token=0, include_lengths=False
        # )
        # SDP_ID = Field(sequential=True, tokenize=sdp_id_tokenize, use_vocab=False, fix_length=self.sdp_length, pad_token=0, batch_first=True)
        # OOV = Field(sequential=True, tokenize=oov_tokenize,
        #             use_vocab=False, batch_first=True,
        #             fix_length=self.fix_length,
        #             pad_token=-1, include_lengths=False
        # )
        
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

    def __init__(self, dl, gpu):
        self.dl = dl
        self.gpu = gpu  # 是否使用gpu

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
