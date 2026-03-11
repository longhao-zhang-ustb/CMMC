import os
import logging
import torch

import torch.optim as optim

from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from data import Dataset, BatchWrapper
from model.net import Net
from evaluate import evaluate
import utils.tool as tool
import argparse
from gensim.models import Word2Vec
from datetime import datetime
# from model.mamba import ModelArgs

parser = argparse.ArgumentParser()
# 目前使用的是re-tacred数据集
parser.add_argument('--data_dir', default=r'data\\', help="Directory containing the dataset")
# parser.add_argument('--embedding_pkl_path', default=r'data\\word_embedding', help="Path to word vecfile.")
# 修改数据集时记得修改配置文件！！！！！！！！！！！！！！
parser.add_argument('--model_dir', default=r'experiments\\base_model', help="Directory containing params.json")
###################################################################################
# 切换不同词向量时，这里需要进行修改！！！！！！！！！！！！！！！！！！
parser.add_argument('--bert', default=True, help=" use Bert or wordembedding")
parser.add_argument('--roberta', default=False, help=" use Roberta or wordembedding")
parser.add_argument('--word2vec', default=False, help="use word2vec")
parser.add_argument('--fasttext', default=False, help="use fasttext")
parser.add_argument('--xlnet', default=False, help=" use XLNet or wordembedding")
parser.add_argument('--distilbert', default=False, help=" use DistilBERT or wordembedding")
# re-tacred数据集的词向量
# parser.add_argument('--word2vec_path', default=r'wordvec\\data_word2vec\\re_tacred\\word2vec_768d.model', help="Path to word2vec model.")
# parser.add_argument('--word2idx_path', default=r'wordvec\\data_word2vec\\re_tacred\\word2idx.pkl', help="Path to word2idx file.")
# parser.add_argument('--fasttext_path', default=r'wordvec\\data_fasttext\\re_tacred\\gensim_fasttext_768d_line.model', help="Path to fasttext model.")
# parser.add_argument('--fasttext_word2idx_path', default=r'wordvec\\data_fasttext\\re_tacred\\word2idx_line.pkl', help="Path to fasttext word2idx file.")
# semeval数据集的词向量
parser.add_argument('--word2vec_path', default=r'wordvec\\data_word2vec\\semeval_300d_60epoch\\word2vec_300d.model', help="Path to word2vec model.")
parser.add_argument('--word2idx_path', default=r'wordvec\\data_word2vec\\semeval_300d_60epoch\\word2idx.pkl', help="Path to word2idx file.")
parser.add_argument('--fasttext_path', default=r'wordvec\\data_fasttext\\semeval_300d\\gensim_fasttext_300d_line.model', help="Path to fasttext model.")
parser.add_argument('--fasttext_word2idx_path', default=r'wordvec\\data_fasttext\\semeval_300d\\word2idx_line.pkl', help="Path to fasttext word2idx file.")

parser.add_argument('--random_vec', default=False, help="use random vector")
parser.add_argument('--vocab_size', default=45431, help="vocab size")
###################################################################################
parser.add_argument('--gpu', default=True, help="use GPU")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")

def train(model, train_data, optimizer, scheduler, args):
    model.train()
    loss_avg = tool.RunningAverage()
    t = trange(len(train_data))
    train_iter = iter(train_data)

    for i in t:
        # fetch the next training batch
        if args.roberta:
            words, pos1s, lens, pos2s, labels, attention_mask = next(train_iter)
            if len(lens) == 1:
                continue
            # compute model output and loss
            outputs = model(words, pos1s, pos2s, attention_mask)
        else:
            words, pos1s, lens, pos2s, labels = next(train_iter)
            if len(lens) == 1:
                continue
            # compute model output and loss
            outputs = model(words, pos1s, pos2s)
        # if outputs.shape[0] != labels.shape[0]:
        #     outputs = outputs[:labels.shape[0]]
        loss = model.loss(outputs, labels)
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()

        # gradient clipping
        # nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()
        # update the average loss
        loss_avg.update(loss.cpu().item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    scheduler.step(loss_avg())
    return loss_avg()


def train_and_evaluate(model, train_data, val_data, optimizer, params, scheduler, metric_labels, model_dir,
                       restore_file, tb_writer, file_name, datatype, args):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        tool.load_checkpoint(restore_path, model, optimizer=None)

    best_val_f1 = 0.0
    patience_counter = 0
    
    # 创建一个新的文件，用于保存模型的时间复杂度信息
    file_time = open(file_name, 'a+', encoding='utf-8')
    file_time.write('all epoch start time: ' + str(datetime.now()) + '\n')
    file_time.close()
    for epoch in range(1, params.max_epoch + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.max_epoch))

        # Train for one epoch on training set
        # 微观状态下，记录每个batch的运行时间，以考察模型的时间复杂
        # 记录1个epoch的时间
        epoch_start_time = datetime.now()
        train_loss = train(model, train_data, optimizer, scheduler, args)
        epoch_end_time = datetime.now()
        epoch_run_time = (epoch_end_time - epoch_start_time).total_seconds()
        # 对单次epoch的运行时间进行记录
        file_time = open(file_name, 'a+', encoding='utf-8')
        file_time.write('epoch ' + str(epoch) + ' run time: ' + str(epoch_run_time) + ' seconds.\n')
        file_time.close()
        
        # Evaluate for one epoch on training set and validation set
        # train_metrics = evaluate(model, train_data, metric_labels)
        train_metrics = evaluate(model, train_data, metric_labels, args=args, data_type=datatype)
        train_metrics['loss'] = train_loss
        train_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in train_metrics.items())
        logging.info("- Train metrics: " + train_metrics_str)
        # logging.info(train_all_metrics)
        val_metrics = evaluate(model, val_data, metric_labels, args=args, data_type=datatype)
        val_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics: " + val_metrics_str)
        # logging.info(val_all_metrics)

        tb_writer.add_scalars('loss',
                              {'train': train_metrics['loss'],
                               'val': val_metrics['loss'], },
                              epoch)
        tb_writer.close()
        # 将损失写入txt文件
        with open(r'loss_exp\loss_record_' + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S").replace(':','-')) + '.txt', 'a+', encoding='utf-8') as file:
            file.write('Train_loss:' + str(train_metrics['loss']) + ', Val_loss:' + str(val_metrics['loss']) + '.\n')
        file_time = open(file_name, 'a+', encoding='utf-8')
        # 对每个epoch的训练损失与验证损失进行记录
        file_time.write('Train_loss:' + str(train_metrics['loss']) + ', Val_loss:' + str(val_metrics['loss']) + '.\n')
        file_time.close()

        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights ot the network
        tool.save_checkpoint({'epoch': epoch + 1,
                              'state_dict': model.state_dict(),
                              'optim_dict': optimizer.state_dict()},
                              is_best=improve_f1 > 0,
                              checkpoint=model_dir)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.max_epoch:
            logging.info("best val f1: {:05.2f}".format(best_val_f1))
            break
    

# 更换数据集时需要进行的操作：
# 1.替换数据
# 2.关系类别替换(2处)
# 3.最后评估结果键值对替换

# 实验结果的保存
# 1.保存每一种类别的实验结果
# 2.保存每一个模型的损失变化
# 3.保存每一个数据集的最优模型
# 4.保存该数据集的整体评价性能

import random
import numpy as np
import torch
from torchinfo import summary

def set_global_seed(seed):
    """
    设置全局随机种子，确保实验完全可复现
    Args:
        seed: 任意整数（如42、10086、230、5888、9999等）
    """
    # 1. Python 原生随机数（如 random.shuffle）
    random.seed(seed)
    # 2. NumPy 随机数（如 np.random.randn）
    np.random.seed(seed)
    # 3. PyTorch CPU 随机数
    torch.manual_seed(seed)
    # 4. PyTorch GPU 随机数（单卡/多卡）
    torch.cuda.manual_seed(seed)
    # 5. CuDNN 确定性模式（避免GPU算法选择的随机性）
    torch.backends.cudnn.deterministic = True
    # 6. 禁用CuDNN自动调优（避免不同算法导致结果差异）
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    # 获取参数设置
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = tool.Params(json_path)

    # Set the random seed for reproducible experiments
    # torch.manual_seed(230)
    # 此处修改为设置全局随机种子: 42、10086、230、5888、9999
    seed = [42, 10086, 230, 5888, 9999]
    seed = seed[0]
    model_name = 'mcnn-cem' 
    dataset = '-semeval'
    wordvec= '-bert'
    set_global_seed(seed)
    # 目前需要修改2处！！！！！！！！！！！！！！！！！！！！！！！！！
    # 记得保存模型，并修改名字！！！！！！！！！！！！！！！！！！！！
    # 切换不同词向量时，上边有一处arg参数需要修改！！！！！！！！！！！！！！
    # data、model、params文件中需要修改为加载word2vec的配置
    # re_traced class
    # metric_labels = list(range(0, 8))
    # data_type = 're_traced'
    # res_file_name = model_name + dataset + '-seed' + str(seed) + wordvec
    # semeval-9 class
    metric_labels = list(range(0, 9))
    data_type = 'semeval_9class'
    res_file_name = model_name + dataset + '-seed' + str(seed) + wordvec

    # Set the logger
    tool.set_logger(os.path.join(args.model_dir, 'train.log'))
    # tensorboard 设置
    tb_writer = tool.TensorBoardWriter(args.model_dir)

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    data_loader = Dataset(args=args, params=params)
    # 此处划分训练集，验证集与测试集
    train_data = BatchWrapper(data_loader.get_data('training'), args.gpu)
    # 验证集与测试集不能为同一个，这里的代码进行修改
    val_data = BatchWrapper(data_loader.get_data('validation'), args.gpu)
    test_data = BatchWrapper(data_loader.get_data('test'), args.gpu)
    ######################此处更换数据集时需要修改关系的类别总数##########################
    model = Net(args, params, class_num=len(metric_labels))

    if params.optim_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.lr,
                              momentum=0.9,
                              weight_decay=params.weight_decay)
    elif params.optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.lr,
                               weight_decay=params.weight_decay)
    else:
        raise ValueError("Unknown optimizer, must be one of 'sgd'/'adam'.")

    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6, verbose=True)
    
    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.max_epoch))
    file_name = r'complexity\\' + res_file_name + '.txt'
    train_and_evaluate(model=model,
                       train_data=train_data,
                       val_data=val_data,
                       optimizer=optimizer,
                       params=params,
                       scheduler=scheduler,
                       metric_labels=metric_labels,
                       model_dir=args.model_dir,
                       restore_file=args.restore_file,
                       tb_writer=tb_writer,
                       file_name=file_name,
                       datatype=data_type,
                       args=args
                       )
    # 调换数据集时这里生成的模型名字修改一下
    restore_path = os.path.join(args.model_dir, 'best.pth.tar')
    tool.load_checkpoint(restore_path, model, optimizer=None)
    # 记录一下模型在测试集上做预测所用的时间 ==> 模型推理所用的时间
    test_start_time = datetime.now()
    test_metrics, test_metrics_all = evaluate(model, test_data, metric_labels, mode='Test', args=args, data_type = data_type)
    test_end_time = datetime.now()
    test_run_time = (test_end_time - test_start_time).total_seconds()
    with open(file_name, 'a+', encoding='utf-8') as file_time:
        file_time.write('test set run time: ' + str(test_run_time) + ' seconds.\n')
        file_time.close()
    # .2f改为.4f，保留小数点后4位
    test_metrics = "; ".join("{}: {:05.2f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- TEST metrics: " + test_metrics)
    with open(file_name, 'a+', encoding='utf-8') as file_time:
        file_time.write(str(test_metrics_all) + '\n')
        file_time.write(test_metrics + '\n')
        file_time.close()
