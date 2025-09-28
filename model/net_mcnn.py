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

    # def __init__(self, args, params, mamba_args: ModelArgs):
    def __init__(self, args, params):
        super(Net, self).__init__()
        self.params = params
        ######################此处更换数据集时需要修改关系的类别总数##########################
        # semeval-task 9class
        # self.class_num = 9
        # re_traced
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
        # feature_dim  868
        feature_dim = word_emb_dim + params.pos_emb_dim * 2
        self.bilstm_gru_hidden_dim = 256
        self.convs = nn.ModuleList([nn.Conv1d(feature_dim, params.kernel_num, kernel_size=kernel_size) \
                                    for kernel_size in params.kernel_sizes])
        self.convs_dropout = nn.Dropout(0.3)
        self.convs_bn = nn.LayerNorm(params.kernel_num * len(params.kernel_sizes))
        self.convs_pool = nn.AdaptiveAvgPool1d(512)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, self.class_num)
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
        # features.shape [5, 96, 868]
        features = torch.cat([words, pos1, pos2], dim=2)
        x = [torch.max(torch.relu(oneconv(features.permute(0, 2, 1))), 2)[0] for oneconv in self.convs]
        x = [self.convs_dropout(item) for item in x]
        x = [item.reshape(item.shape[0], 1, -1) for item in x]
        x = torch.cat(x, dim=2)
        x = self.convs_bn(x)
        x = self.convs_pool(x)
        x = x.reshape(x.shape[0], -1)
        # 输出分类结果
        x = self.fc(x)
        return x 
