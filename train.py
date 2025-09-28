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
parser.add_argument('--data_dir', default='./data/', help="Directory containing the dataset")
parser.add_argument('--embedding_pkl_path', default='./data/word_embedding', help="Path to word vecfile.")
parser.add_argument('--model_dir', default=r'experiments\base_model', help="Directory containing params.json")
parser.add_argument('--bert', default=True, help=" use Bert or wordembedding")
parser.add_argument('--gpu', default=True, help="use GPU")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")

def train(model, train_data, optimizer, scheduler):
    model.train()
    loss_avg = tool.RunningAverage()
    t = trange(len(train_data))
    train_iter = iter(train_data)

    for i in t:
        # fetch the next training batch
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
                       restore_file, tb_writer):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        tool.load_checkpoint(restore_path, model, optimizer=None)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.max_epoch + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.max_epoch))

        # Train for one epoch on training set
        train_loss = train(model, train_data, optimizer, scheduler)

        # Evaluate for one epoch on training set and validation set
        # train_metrics = evaluate(model, train_data, metric_labels)
        train_metrics = evaluate(model, train_data, metric_labels)
        train_metrics['loss'] = train_loss
        train_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in train_metrics.items())
        logging.info("- Train metrics: " + train_metrics_str)
        # logging.info(train_all_metrics)
        val_metrics = evaluate(model, val_data, metric_labels)
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

if __name__ == '__main__':

    # 获取参数设置
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = tool.Params(json_path)

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)

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
    # metric_labels = list(range(0, 9))
    # 目前需要修改2处！！！！！！！！！！！！！！！！！！！！！！！！！
    # re_tacred class
    metric_labels = list(range(0, 8))
    # logging.info("- done.")
    
    model = Net(args, params)

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
    train_and_evaluate(model=model,
                       train_data=train_data,
                       val_data=val_data,
                       optimizer=optimizer,
                       params=params,
                       scheduler=scheduler,
                       metric_labels=metric_labels,
                       model_dir=args.model_dir,
                       restore_file=args.restore_file,
                       tb_writer=tb_writer
                       )
    # 调换数据集时这里生成的模型名字修改一下
    restore_path = os.path.join(args.model_dir, 'best.pth.tar')
    tool.load_checkpoint(restore_path, model, optimizer=None)
    test_metrics = evaluate(model, test_data, metric_labels, mode='Test')
    test_metrics = "; ".join("{}: {:05.2f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- TEST metrics: " + test_metrics)
