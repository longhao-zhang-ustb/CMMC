import argparse
import logging
import os
import sys
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_folder)
import torch
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from data import Dataset, BatchWrapper
from model.net import Net
import utils.tool as tool
from prepare_data.rel2json import get_single_evaluate
from sklearn.metrics import confusion_matrix

def evaluate(model, test_data, metric_labels, mode='Train', args=None, data_type='re_traced'):
    """Evaluate the model on `num_steps` batches."""
    # set model to evaluation mode
    model.eval()

    loss_avg = tool.RunningAverage()
    output_labels = list()
    target_labels = list()

    # compute metrics over the dataset
    for i, batch_data in enumerate(test_data):
        # fetch the next evaluation batch
        if args.roberta:
            words, pos1s, lens, pos2s, labels, attention_mask = batch_data
            # compute model output
            outputs = model(words, pos1s, pos2s, attention_mask)
        else:
            words, pos1s, lens, pos2s, labels = batch_data
            # compute model output
            outputs = model(words, pos1s, pos2s)
        # if outputs.shape[0] != labels.shape[0]:
        #     outputs = outputs[:labels.shape[0]]
        loss = model.loss(outputs, labels)
        loss_avg.update(loss.cpu().item())

        batch_output_labels = torch.max(outputs, dim=1)[1]
        output_labels.extend(batch_output_labels.data.cpu().numpy().tolist())
        target_labels.extend(labels.data.cpu().numpy().tolist())

    # Calculate precision, recall and F1 for all relation categories micro==>macro
    # 暂时修改为micro-averge记录结果
    p_r_f1_all = None
    if mode == 'Train':
        p_r_f1_s = precision_recall_fscore_support(target_labels, output_labels, labels=metric_labels, average='macro', zero_division=1)
        precision, recall, f1score, support = precision_recall_fscore_support(target_labels, output_labels, labels=metric_labels, average=None, zero_division=1)
        p_r_f1 = {'precison': p_r_f1_s[0] * 100,
                'recall': p_r_f1_s[1] * 100,
                'f1': p_r_f1_s[2] * 100,
                'loss': loss_avg()
        }
    # 暂时修改为micro-average
    elif mode == 'Test':
        # 绘制混淆矩阵
        # print(metric_labels)
        # cm = confusion_matrix(target_labels, output_labels, labels=metric_labels)
        # plt.figure(figsize=(10,7))
        # sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')  # fmt="d"表示整数格式，cmap选择颜色映射
        # plt.ylabel('Actual label')
        # plt.xlabel('Predicted label')
        # plt.savefig(r'20260218_experimental_data\fig_save\confusion_matrix.png', dpi=1000)
        # 打印错误的样本
        # for i in range(len(target_labels)):
        #     if target_labels[i] != output_labels[i]:
        #         print(f"Sample {i}: True label = {target_labels[i]}, Predicted label = {output_labels[i]}")
        p_r_f1_s = precision_recall_fscore_support(target_labels, output_labels, labels=metric_labels, average='macro', zero_division=1)
        precision, recall, f1score, support = precision_recall_fscore_support(target_labels, output_labels, labels=metric_labels, average=None, zero_division=1)
        # 更换数据集时，这里的指标需要修改 semeval_18class、semeval_9class、i2b2、re_traced
        p_r_f1_all = get_single_evaluate(precision=precision, recall=recall, f1score=f1score, data_type=data_type)
        logging.info(p_r_f1_all)
        p_r_f1 = {
            'precison': p_r_f1_s[0] * 100,
            'recall': p_r_f1_s[1] * 100,
            'f1': p_r_f1_s[2] * 100,
            'loss': loss_avg()
        }
    
    return p_r_f1 if mode == 'Train' else (p_r_f1, p_r_f1_all)
