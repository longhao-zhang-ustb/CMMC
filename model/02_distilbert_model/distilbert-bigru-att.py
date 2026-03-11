import pickle
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
# from transformers import BertModel, BertTokenizer
import os
import sys
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_folder)
import torch.nn.functional as F
from torch_multi_head_attention import MultiHeadAttention

class Net(torch.nn.Module):

    # def __init__(self, args, params, mamba_args: ModelArgs):
    def __init__(self, args, params, class_num):
        super(Net, self).__init__()
        self.params = params
        ######################此处更换数据集时需要修改关系的类别总数##########################
        self.class_num = class_num
        self.bert=args.bert
        self.distilbert = args.distilbert
        self.batch_size = params.batch_size
        if self.bert:
            word_emb_dim = 768
            self.word_emb = BertModel.from_pretrained('bert-base-uncased')
            self.word_emb.eval()  # 设置为评估模式，关闭dropout等
        elif self.distilbert:
            word_emb_dim = 768
            self.word_emb = AutoModel.from_pretrained(
                r'D:\\distilbert_cache\\models--distilbert-base-uncased',
                local_files_only=True  # 缓存到D盘，避免C盘权限/空间问题
            )
            self.word_emb.eval()  # 设置为评估模式，关闭dropout等
        else:
            with open(args.embedding_pkl_path + '_numpy.pkl', 'rb') as f:
                pretrained_weight = pickle.load(f)
                word_emb_dim = pretrained_weight.shape[1]
                self.word_emb = nn.Embedding(pretrained_weight.shape[0], pretrained_weight.shape[1])
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
            nn.Linear(self.hidden_dim, self.class_num)
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
        elif self.distilbert:
            with torch.no_grad():
                words = self.word_emb(words)[0]
        else:
            words = self.word_emb(words)
        words = words[:, 1:-1, :]  # 去除第一个[CLS]
        pos1 = self.pos1_emb(pos1)
        pos2 = self.pos2_emb(pos2)
        # features.shape [5, 96, 868]
        features = torch.cat([words, pos1, pos2], dim=2)
        ########################BiMamba-MCNN##############################
        x, _ = self.bigru(features)
        x = x[:, -1, :]
        x = x.view(-1, 1, self.hidden_dim)
        x = self.atten_layer(x, x, x)
        ######################################################################
        # 输出分类结果
        x = self.fc(x.view(-1, self.hidden_dim))
        return x 
