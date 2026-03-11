import pickle
import torch
from torch import nn
from pytorch_transformers import BertModel, BertTokenizer
# from transformers import BertModel, BertTokenizer
import os
import sys
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_folder)
import torch.nn.functional as F
from torch_multi_head_attention import MultiHeadAttention
import gensim

class Net(torch.nn.Module):

    # def __init__(self, args, params, mamba_args: ModelArgs):
    def __init__(self, args, params, class_num):
        super(Net, self).__init__()
        self.params = params
        ######################此处更换数据集时需要修改关系的类别总数##########################
        self.class_num = class_num
        self.bert=args.bert
        self.batch_size = params.batch_size
        if self.bert:
            word_emb_dim = 768
            self.word_emb = BertModel.from_pretrained('bert-base-uncased')
            self.word_emb.eval()  # 设置为评估模式，关闭dropout等
        elif args.word2vec:
            word2vec_model = gensim.models.Word2Vec.load(args.word2vec_path)
            # 1. 获取词向量矩阵
            original_weights = torch.FloatTensor(word2vec_model.wv.vectors)
            # 2. 生成<UNK>的向量（推荐用所有词向量的均值，也可以用全0/随机）
            unk_vector = torch.mean(original_weights, dim=0, keepdim=True)  # shape: [1, emb_dim]
            # 3. 将<UNK>向量添加到词向量矩阵的开头
            weights = torch.cat([unk_vector, original_weights], dim=0)
            # 创建embedding层
            self.word_emb = nn.Embedding.from_pretrained(weights, freeze=True)
            # 获取词向量维度
            word_emb_dim = word2vec_model.wv.vector_size
        elif args.fasttext:
            fasttext_model = gensim.models.FastText.load(args.fasttext_path)
            # 1. 获取词向量矩阵
            original_weights = torch.FloatTensor(fasttext_model.wv.vectors)
            # 2. 生成<UNK>的向量（推荐用所有词向量的均值，也可以用全0/随机）
            unk_vector = torch.mean(original_weights, dim=0, keepdim=True)  # shape: [1, emb_dim]
            # 3. 将<UNK>向量添加到词向量矩阵的开头
            weights = torch.cat([unk_vector, original_weights], dim=0)
            # 创建embedding层
            self.word_emb = nn.Embedding.from_pretrained(weights, freeze=True)
            # 获取词向量维度
            word_emb_dim = fasttext_model.wv.vector_size
        else:
            # 随机词向量
            word_emb_dim = 768
            self.word_emb = nn.Embedding(args.vocab_size, word_emb_dim)
        self.pos1_emb = nn.Embedding(params.pos_emb_size, params.pos_emb_dim, padding_idx=0)
        self.pos2_emb = nn.Embedding(params.pos_emb_size, params.pos_emb_dim, padding_idx=0)
        # feature_dim  868
        feature_dim = word_emb_dim + params.pos_emb_dim * 2
        #################################################################################
        # bigru-att
        self.seq_len = params.fix_length
        self.hidden_dim = 1024
        self.bigru = nn.GRU(feature_dim, self.hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.atten_layer = MultiHeadAttention(self.hidden_dim, 8)
        ######################################################
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 256), # 512 --> 256
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.class_num)
        )
        self.loss = nn.CrossEntropyLoss()
        if args.gpu:
            self.cuda()
    
    def begin_state(self):
        # ******注意这里的hidden_dim，需要动态调整******
        state = (
            torch.zeros(2, self.batch_size, self.hidden_dim // 2),
            torch.zeros(2, self.batch_size, self.hidden_dim // 2)
        )
        return tuple(data.to("cuda") for data in state)

    def forward(self, words, pos1, pos2):
        if self.bert:
            with torch.no_grad():
                words = self.word_emb(words)[0]
            words = words[:, 1:-1, :]  # 去除第一个[CLS]
        else:
            words = self.word_emb(words)
        pos1 = self.pos1_emb(pos1)
        pos2 = self.pos2_emb(pos2)
        # features.shape [5, 96, 868]
        features = torch.cat([words, pos1, pos2], dim=2)
        ########################BiMamba-MCNN##############################
        x, _ = self.bigru(features)
        x = torch.mean(x, dim=1)
        x = x.view(-1, 1, self.hidden_dim)
        x = self.atten_layer(x, x, x)
        ######################################################################
        # 输出分类结果
        x = self.fc(x.view(-1, self.hidden_dim))
        return x 
