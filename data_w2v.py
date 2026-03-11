import torch
import pickle
import argparse

from pytorch_transformers import BertTokenizer
from torchtext.data import Field, TabularDataset
from torchtext.data import BucketIterator
from gensim.models import Word2Vec


class Dataset(object):
    def __init__(self, args, params):
        self.batch_size = params.batch_size
        self.fix_length = params.fix_length
        self.root_path = args.data_dir
        self.use_bert = args.bert
        self.use_word2vec = args.word2vec

        if self.use_bert:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.use_word2vec:
            # 从 Word2Vec 模型生成 word2idx
            word2vec_model = Word2Vec.load(args.word2vec_path)
            # 2. 获取模型词表（完整单词列表）
            vocab = word2vec_model.wv.index_to_key
            # 3. 生成word2idx：0号索引留给<UNK>，真实单词从1开始
            self.word2idx = {'<UNK>': 0}  # 先加<UNK>
            for idx, word in enumerate(vocab):
                self.word2idx[word] = idx + 1  # 真实单词索引从1开始
            tokenizer = None
        else:
            # 随机向量，从数据统计 vocab
            with open(args.word2idx_path, 'rb') as f:
                word2idx = pickle.load(f)
            tokenizer = None

        def word_tokenize(sentence):
            if self.use_bert:
                tokenized_text = sentence.split(' ')
                sentence = tokenizer.convert_tokens_to_ids(tokenized_text)
                # 添加[CLS]和[SEP]
                sentence = [101] + sentence +[102]
            elif self.use_word2vec:
                tokenized_text = sentence.split(' ')
                sentence = []
                for word in tokenized_text:
                    # 用类内的word2idx，找不到就返回<UNK>的索引0
                    word_idx = self.word2idx.get(word, 0)
                    sentence.append(word_idx)
            return sentence

        def pos_tokenize(posids):
            return [int(_) for _ in posids.split(' ')]

        TEXT = Field(sequential=True, tokenize=word_tokenize,
                     use_vocab=False, batch_first=True,
                     fix_length=self.fix_length + 2 if self.use_bert else self.fix_length,  # 添加了 cls和sep
                     pad_token=0)
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

            func = lambda x: x.cuda()
            if not self.gpu:
                yield [words, pos1s, lens, pos2s, labels]
            else:
                yield list(map(func, [words, pos1s, lens, pos2s, labels]))

    def __len__(self):
        return len(self.dl)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', default='./data/', help="Directory containing the dataset")
#     parser.add_argument('--embedding_pkl_path', default='./data/word_embedding', help="Path to word vecfile.")
#     parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
#     parser.add_argument('--bert', default=False, help=" use Bert or wordembedding")

#     params = type('classA', (object,), dict(batch_size=32, fix_length=96))()

#     args = parser.parse_args()
#     dataset = Dataset(args=args, params=params)

#     val_data = BatchWrapper(dataset.get_data('validation'), gpu=True)
#     batch = next(iter(val_data))
#     print(batch)
# print('batch:\n', batch)
# print('batch_text:\n', batch.text)
# print('batch_label:\n', batch.label)
