import nltk
from nltk.corpus import stopwords
import json
import contractions

def remove_stopwords(file_input, file_output):
    # 英文停用词列表
    stop_words = set(stopwords.words('english'))
    output_file = open(file_output, 'a+', encoding='utf-8')
    with open(file_input, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for index, row in enumerate(content):
            content = eval(row)
            text = content["sentence"]
            # 分词
            words = text.split()
            # 过滤停用词
            filtered_words = [word for word in words if word not in stop_words]
            deal_text = ' '.join(filtered_words)
            # 处理缩略词
            deal_text = contractions.fix(deal_text)
            content["sentence"] = deal_text
            content = json.dumps(content, ensure_ascii=False)
            output_file.write(str(content) + '\n')
            print('处理完第'+str(index)+'行!')

if __name__ == '__main__':
    file_input = r'experiment_data\semeval8\sem_data_9class\train.txt'
    file_output = r'experiment_data\semeval8\sem_data_9class\train_stop_words.txt'
    remove_stopwords(file_input, file_output)
