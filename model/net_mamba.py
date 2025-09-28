import pickle
import torch
from torch import nn
from pytorch_transformers import BertModel, BertTokenizer
import os
import sys
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_folder)
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import json
import torch.nn.functional as F
from mamba import ModelArgs
from bimamba import BiMambaEncoder

class Net(torch.nn.Module):

    def __init__(self, args, params):
        super(Net, self).__init__()
        self.params = params
        ######################此处更换数据集时需要修改关系的类别总数##########################
        # semeval-task 9class
        # self.class_num = 9
        # re-traced
        self.class_num = 8
        self.bert=args.bert
        self.hidden_dim = params.hidden_dim
        self.batch_size = params.batch_size
        if self.bert:
            word_emb_dim = 768
            self.word_emb = BertModel.from_pretrained('bert-base-uncased')
        else:
            with open(args.embedding_pkl_path + '_numpy.pkl', 'rb') as f:
                pretrained_weight = pickle.load(f)
                word_emb_dim = pretrained_weight.shape[1]
                self.word_emb = nn.Embedding(pretrained_weight.shape[0], pretrained_weight.shape[1])
        self.pos1_emb = nn.Embedding(params.pos_emb_size, params.pos_emb_dim, padding_idx=0)
        self.pos2_emb = nn.Embedding(params.pos_emb_size, params.pos_emb_dim, padding_idx=0)
        feature_dim = word_emb_dim + params.pos_emb_dim * 2
        self.bilstm_gru_hidden_dim = 256
        ##############################BiLSTM##############################
        self.mamba_args = ModelArgs(d_model=feature_dim, n_layer=8, d_state=16)
        self.bimamba_layer = nn.Sequential(
            BiMambaEncoder(self.mamba_args)
        )
        self.mamba_norm = nn.LayerNorm(feature_dim)
        self.mamba_dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
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
        else:
            words = self.word_emb(words)
        words = words[:, 1:-1, :]  # 去除第一个[CLS]
        pos1 = self.pos1_emb(pos1)
        pos2 = self.pos2_emb(pos2)
        features = torch.cat([words, pos1, pos2], dim=2)
        # 多尺度卷积操作
        # 输入卷积神经网络的数据格式为（batch_size, embedding_dim, sequence_length）
        ##############################BiLSTM#################################
        x = self.bimamba_layer(features)
        x = self.mamba_norm(x)
        x = self.mamba_dropout(x)
        x = torch.sum(x, 1)
        # 输出分类结果
        x = self.fc(x)
        return x 
