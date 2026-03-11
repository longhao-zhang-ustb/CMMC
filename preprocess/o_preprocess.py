# coding=utf-8
'''
Preprocess original Sem-eval task8 data
'''
import json
import os
import sys
import nltk
from nltk.corpus import stopwords
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_suffix_regex, compile_prefix_regex
import itertools
import spacy
import networkx as nx
import matplotlib.pyplot as plt
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_folder)
from wiki80_rel import getWiki80_class2label, getWiki80_label2class
from spacy.tokens import Doc
from pytorch_transformers import BertTokenizer
# from nyt10_rel import getnyt10_class2label, getnyt10_label2class
# from wiki20m_rel import getwiki20m_class2label, getwiki20m_label2class
# from nyt10m_rel import getnyt10m_class2label, getnyt10m_label2class
# class2label = getnyt10_class2label()
# label2class = getnyt10_label2class()

# class2label = getwiki20m_class2label()
# label2class = getwiki20m_label2class()

class2label = getWiki80_class2label()
label2class = getWiki80_label2class()

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

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def process_file(in_filename, out_filename, tokenizer, vocab_array):
    max_len = 0
    max_distance = 0
    nlp = spacy.load('en_core_web_trf')
    # nlp.tokenizer = custom_tokenizer(nlp)
    nlp.tokenizer= WhitespaceTokenizer(nlp.vocab)
    with open(in_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 将每一行转换为字典类型数据
    for row, line in enumerate(lines):
        lines[row] = eval(line)
    new_lines = []
    for i in range(0, len(lines)):
        # 执行新处理逻辑
        relation = lines[i]['relation']
        question = ' '.join(lines[i]['token'])
        e1_begin = lines[i]['h']['pos'][0]
        e1_end = lines[i]['h']['pos'][1]
        e2_begin = lines[i]['t']['pos'][0]
        e2_end = lines[i]['t']['pos'][1]
        ################################################
        question = question.lower()
        sdp = nlp(question)
        question = question.split(' ')
        edges = []
        for token in sdp:
            for child in token.children:
                edges.append(('{0}'.format(token.lower_), '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)
        # 获取实体1
        entity1 = question[e1_begin].lower()
        entity2 = question[e2_begin].lower()
        try:
            sdp = nx.shortest_path(graph, source=entity1, target=entity2)
            # 将最短依存路径转换为数字符号
            entity1 = question[e1_begin: e1_end]
            entity2 = question[e2_begin: e2_end]
            sdp = sdp[1:-1]
            sdp = list(itertools.chain(entity1, sdp, entity2))                                                        
            sdp_id = [question.index(ele) for ele in sdp]
            sdp = ' '.join(sdp)
        except:
            nx.draw(graph,with_labels=True)
            plt.show()
            exit()
        print('处理完第'+str(i + 1)+'行句子')
        # question = question.replace("'", " '")
        # question = question.replace(",", " ,")
        # question = question.replace(".", " .")
        ################################################
        max_len = max(max_len, len(question))
        max_distance = max(max_distance, e1_end)
        max_distance = max(max_distance, len(question) - e1_end)
        max_distance = max(max_distance, e2_end)
        max_distance = max(max_distance, len(question) - e2_end)
        # 在原始数据中增加oov选项，处理完成后，手动替换下\n
        tokenized_text = question[:96]
        sentence = tokenizer.convert_tokens_to_ids(tokenized_text)
        oov = [0 if item != 100 else tokenized_text[index] for index, item in enumerate(sentence)]
        oov = [-1 if item==0 else vocab_array.index(item) for item in oov]
        new_lines.append({
                          'sentence': ' '.join(question[:96]),
                          'label': class2label.get(relation, 0),
                          "e1": ' '.join([str(_pos(i - e1_begin)) for i in range(len(question))]),
                          'e1_begin': e1_begin,
                          'e2': ' '.join([str(_pos(i - e2_begin)) for i in range(len(question))]),
                          'e2_begin': e2_begin,
                          'sdp': sdp,
                          'sdp_id': sdp_id,
                          'oov': oov
                          })

    with open(out_filename, 'w') as f:
        for dic in new_lines:
            f.writelines(json.dumps(dic) + '\n')

    print("Max length: {}".format(max_len))
    print("Max distance: {}".format(max_distance))

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    file = open(r'corpus\vocab.txt', 'r', encoding='utf-8')
    vocab_content = file.readlines()
    vocab_array = []
    for index, item in enumerate(vocab_content):
        vocab_array.append(item.strip())
        print('添加第'+str(index+1)+'行!')
    # oov = [-1 if item==0 else vocab_content.index(item) for item in oov]
    train_in_file = "database/wiki80/wiki80_train.txt"
    train_out_file = "database/wiki80/train.txt"
    test_in_file = 'database/wiki80/wiki80_val.txt'
    test_out_file = 'database/wiki80/test.txt'
    # val_in_file = 'database/wiki80/wiki80_val.txt'
    # val_out_file = 'database/wiki80/val.txt'
    # nltk.download('stopwords')
    # process_file(train_in_file, train_out_file, tokenizer, vocab_array)
    process_file(test_in_file, test_out_file, tokenizer, vocab_array)
    # process_file(val_in_file, val_out_file)
