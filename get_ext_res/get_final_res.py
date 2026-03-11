from statistics import stdev
import json
import os

if __name__ == '__main__':
    # 读取某个文件夹下的所有txt文件
    folder = r'complexity'
    output_file = r'final_res.json'
    file_names = [f for f in os.listdir(folder) if f.endswith('.txt')]
    for file in file_names:
        file_path = os.path.join(folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            res_dict = {}
            res_dict['title'] = file_path.split('\\')[-1].split('.')[0]
            lines = f.readlines()
            train_loss = []
            val_loss = []
            train_time = []
            test_time = 0
            # 统计每种关系的识别结果
            relation_single_res = {}
            # 统计整体的识别结果
            for line in lines:
                # 统计下训练和验证的损失
                if line.startswith("Train_loss:"):
                    line = line.strip()[:-1]
                    train_loss.append(round(eval(line.split(', ')[0].split(':')[1]), 6))
                    val_loss.append(round(eval(line.split(', ')[1].split(':')[1]), 6))
                    res_dict['train_loss'] = train_loss
                    res_dict['val_loss'] = val_loss
                # 统计下训练时间
                if line.startswith("epoch"):
                    line = line.strip()
                    train_time.append(eval(line.split(' ')[4]))
                    res_dict['train_time'] = train_time
                # 统计下推理时间
                if line.startswith('test'):
                    line = line.strip()
                    test_time = eval(line.split(' ')[-2])
                    res_dict['test_time'] = test_time
                # 统计下对每种结果的识别结果
                if line.startswith("{"):
                    relation_single_res.update(eval(line))
                    res_dict['relation_single_res'] = relation_single_res
                # 统计下宏平均分类结果
                if line.startswith("precision"):
                    res_dict["macro_precison"] = eval(line.split('; ')[0].split(': ')[1])
                    res_dict["macro_recall"] = eval(line.split('; ')[1].split(': ')[1])
                    res_dict["macro_f1"] = eval(line.split('; ')[2].split(': ')[1])
            # 统计下训练时间的平均值
            res_dict['train_time_avg'] = round(sum(res_dict['train_time'])/len(res_dict['train_time']), 6)
            # 统计下训练时间的标准差
            res_dict['train_time_std'] = round(stdev(res_dict['train_time']), 6)
            # 统计下训练损失的平均值
            res_dict['train_loss_avg'] = round(sum(res_dict['train_loss'])/len(res_dict['train_loss']), 6)
            # 统计下训练损失的标准差
            res_dict['train_loss_std'] = round(stdev(res_dict['train_loss']), 6)
            # 统计下验证损失的平均值
            res_dict['val_loss_avg'] = round(sum(res_dict['val_loss'])/len(res_dict['val_loss']), 6)
            # 统计下验证损失的标准差
            res_dict['val_loss_std'] = round(stdev(res_dict['val_loss']), 6)
            print(res_dict)
            res_json = json.dumps(res_dict, ensure_ascii=False)
        # 将结果追加到json文件中
        with open(output_file, 'a+', encoding='utf-8') as f:
            f.write(res_json + '\n')
