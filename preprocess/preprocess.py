# coding=utf-8

'''
Preprocess original Sem-eval task8 data
'''
import json
import os
import sys
import contractions
import nltk
from nltk.corpus import stopwords
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_suffix_regex, compile_prefix_regex
import itertools
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_folder)

# class2label = {'Other': 0,
#                'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
#                'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
#                'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
#                'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
#                'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
#                'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
#                'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
#                'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
#                'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
class2label = {
    'Message-Topic':0, 
    'Product-Producer':1,
    'Instrument-Agency':2, 
    'Entity-Destination':3,
    'Cause-Effect':4, 
    'Component-Whole':5,
    'Entity-Origin': 6, 
    'Member-Collection':7,
    'Content-Container':8
}
label2class = {
    0: 'Message-Topic',
    1:'Product-Producer',
    2:'Instrument-Agency', 
    3:'Entity-Destination',
    4:'Cause-Effect', 
    5:'Component-Whole',
    6:'Entity-Origin', 
    7:'Member-Collection',
    8:'Content-Container'
}
# label2class = {0: 'Other',
#                1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
#                3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
#                5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
#                7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
#                9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
#                11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
#                13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
#                15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
#                17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}
def custom_tokenizer(nlp):
    pref = list(nlp.Defaults.prefixes)
    pref.remove(r'\$')
    pref_re = compile_prefix_regex(pref)
    
    inf = list(nlp.Defaults.infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])   # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=pref_re.search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

def handleposition(entity, sentence_length):
    res = [_pos(i - entity) for i in range(sentence_length)]
    return res


def handlelabel(y):
    return class2label.get(y, 0)


def _pos(x):
    '''
    map the relative distance between [0, 123)
    '''
    if x < -49:
        return 0
    if 49 >= x >= -49:
        return x + 50
    if x > 49:
        return 0


def process_question(question):
    
    question = question.lower()
    question = question.replace("'", " '")
    question = question.replace(",", " ,")
    question = question.replace(".", " .")
    question = question.replace("  ", " ")
    question = question.split(' ')
    e1_begin = e1_end = e2_begin = e2_end = 0
    if '' in question:
        question.remove('')
    for i, item in enumerate(question):
        if item.startswith('<e1>'):
            e1_begin = i
        if item.endswith('</e1>'):
            e1_end = i
        if item.startswith('<e2>'):
            e2_begin = i
        if item.endswith('</e2>'):
            e2_end = i

    def remove_tag(x):
        x = x.replace('<e1>', '')
        x = x.replace('</e1>', '')
        x = x.replace('<e2>', '')
        x = x.replace('</e2>', '')
        return x

    question = list(map(remove_tag, question))
    return question, e1_begin, e1_end, e2_begin, e2_end


def process_file(in_filename, out_filename):
    max_len = 0
    max_distance = 0
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = custom_tokenizer(nlp)
    with open(in_filename, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for i in range(0, len(lines), 4):
        # 每隔4行处理一次
        relation = lines[i + 1].strip()
        question = lines[i].strip().split('\t')[1][1:-1]
        question, e1_begin, e1_end, e2_begin, e2_end = process_question(question)
        max_len = max(max_len, len(question))
        max_distance = max(max_distance, e1_end)
        max_distance = max(max_distance, len(question) - e1_end)
        max_distance = max(max_distance, e2_end)
        max_distance = max(max_distance, len(question) - e2_end)
        sentence = ' '.join(question[:96])
        # 添加最短依存路径信息
        # 拼接一下实体信息
        sdp = nlp(sentence)
        edges = []
        for token in sdp:
            for child in token.children:
                edges.append(('{0}'.format(token.lower_), '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)
        # nx.draw(graph,with_labels=True)
        # plt.show()
        # 获取实体1
        entity1 = question[e1_begin].lower()
        entity2 = question[e2_begin].lower()
        print(entity1, entity2)
        sdp = nx.shortest_path(graph, source=entity1, target=entity2)
        # 将最短依存路径转换为数字符号
        entity1 = question[e1_begin: e1_end + 1]
        entity2 = question[e2_begin: e2_end + 1]
        sdp = sdp[1:-1]
        sdp = list(itertools.chain(entity1, sdp, entity2))
        # print(sdp, question)
        sdp_id = [question.index(ele) for ele in sdp]
        sdp = ' '.join(sdp)
        
        print('处理完第'+str(i + 1)+'行句子')

        new_lines.append({'sentence': sentence,
                          'label': class2label.get(relation, 0),
                          "e1": ' '.join([str(_pos(i - e1_begin)) for i in range(len(question))]),
                          'e1_begin': e1_begin,
                          'e2': ' '.join([str(_pos(i - e2_begin)) for i in range(len(question))]),
                          'e2_begin': e2_begin,
                          'sdp': sdp,
                          'sdp_id': sdp_id})
        
    with open(out_filename, 'w') as f:
        for dic in new_lines:
            f.writelines(json.dumps(dic) + '\n')

    print("Max length: {}".format(max_len))
    print("Max distance: {}".format(max_distance))
    
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
    train_in_file = r"data_all\semeval_data_9class\TRAIN_FILE.TXT"
    train_out_file = r"experiment_data\semeval8\sem_data_9class\train_new.txt"
    test_in_file = r'data_all\semeval_data_9class\TEST_FILE_FULL.TXT'
    test_out_file = r'experiment_data\semeval8\sem_data_9class\test_new.txt'

    process_file(train_in_file, train_out_file)
    process_file(test_in_file, test_out_file)
