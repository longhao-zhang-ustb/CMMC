import os
import re
import nltk
import numpy as np
from gensim.models import FastText
from nltk.tokenize import word_tokenize
import pickle

# 下载nltk分词资源（首次运行需执行）
nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess_english_corpus_line_by_line(corpus_path):
    """
    适配“每行1句”的英文语料预处理（核心修改：按行读取，无需分割句子）
    :param corpus_path: 每行1句的英文语料txt路径
    :return: 分词后的语料列表 [[word1, word2], [word3, ...], ...]
    """
    # 检查文件是否存在
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"语料文件不存在：{corpus_path}")
    
    processed_corpus = []
    # 按行读取语料（每行1句）
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 逐行处理（每行对应一个句子）
    for line in lines:
        # 去除行首尾的空格/换行符
        line = line.strip()
        # 4. 英文分词（将句子拆分为单词列表）
        words = word_tokenize(line)
        # 跳过分词后为空的句子
        if words:
            processed_corpus.append(words)
    
    print(f"语料预处理完成：共读取 {len(lines)} 行，有效句子 {len(processed_corpus)} 句")
    return processed_corpus

def save_word2idx_to_txt(word2idx, txt_save_path):
    """
    将word2idx字典保存为txt文件（易读格式）
    :param word2idx: 单词-索引字典
    :param txt_save_path: txt文件保存路径
    """
    # 按索引升序排列（方便查看，0=UNK，1开始是正常单词）
    sorted_items = sorted(word2idx.items(), key=lambda x: x[1])
    
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        # 首行写说明（可选，提升可读性）
        f.write("# 单词 索引（索引0为未知词<UNK>，后续为FastText词表顺序）\n")
        # 逐行写入：单词 索引
        for word, idx in sorted_items:
            f.write(f"{word} {idx}\n")

def train_fasttext_768d_gensim(corpus_path, model_save_path, word2idx_pkl_path, word2idx_txt_path, unk_token="<UNK>"):
    """训练768维度FastText词向量（逻辑不变，适配新的预处理函数）"""
    # 调用修改后的预处理函数
    processed_corpus = preprocess_english_corpus_line_by_line(corpus_path)
    
    model = FastText(
        sentences=processed_corpus,
        vector_size=768,  # 768维度核心参数
        window=5,         # 上下文窗口大小
        min_count=2,      # 忽略出现次数<2的词（小语料可设为1）
        sg=1,             # skipgram模型（语义效果更好）
        epochs=30,        # 训练轮数
        workers=8,        # 多线程加速
        min_n=3,          # 英文子词最小长度
        max_n=6           # 英文子词最大长度
    )
    
    # 保存模型
    model.save(model_save_path)
    print(f"768维模型已保存至：{model_save_path}")
    
    # 生成word2idx字典
    vocab = model.wv.index_to_key
    word2idx = {unk_token: 0} # 0留给未知词
    for idx, word in enumerate(vocab):
        word2idx[word] = idx + 1
    
    # 保存word2idx.pkl
    with open(word2idx_pkl_path, 'wb') as f:
        pickle.dump(word2idx, f)
    print(f"word2idx.pkl已生成，保存至：{word2idx_pkl_path}")
    
    # 同步生成word2idx.txt
    save_word2idx_to_txt(word2idx, word2idx_txt_path)
    print(f"word2idx.txt已生成，保存至：{word2idx_txt_path}")
    
    return model, word2idx

def export_vectors_to_txt(model, txt_save_path, unk_token="<UNK>", unk_vector_type="mean"):
    """导出768维词向量为TXT文件（逻辑不变）"""
    vocab = model.wv.index_to_key
    vocab_size = len(vocab) + 1
    real_vectors = [model.wv[word] for word in vocab]
    vector_dim = model.vector_size  # 768维度
    
    # 2. 生成<UNK>的向量
    if unk_vector_type == "mean":
        # 推荐：所有词向量的均值（语义更合理）
        unk_vector = np.mean(real_vectors, axis=0)
    elif unk_vector_type == "zero":
        # 备选：全零向量
        unk_vector = np.zeros(vector_dim)
    else:
        raise ValueError("unk_vector_type仅支持mean/zero")
    
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        # 首行：词汇量 向量维度（标准格式）
        f.write(f"{vocab_size} {vector_dim}\n")
        # 第一行：<UNK> + 其向量（对应0号索引）
        unk_vector_str = ' '.join([f"{v:.6f}" for v in unk_vector])
        f.write(f"{unk_token} {unk_vector_str}\n")
        
        # 逐词写入：单词 + 空格分隔的768个向量值
        for word in vocab:
            vector = model.wv[word]
            vector_str = ' '.join([f"{v:.6f}" for v in vector])
            f.write(f"{word} {vector_str}\n")
    
    print(f"768维词向量已导出为TXT：{txt_save_path}")
    print(f"词表大小：{vocab_size}，向量维度：{vector_dim}")

def load_txt_vectors(txt_path):
    """验证导出的TXT词向量（可选）"""
    vectors = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        vocab_size, vector_dim = map(int, first_line.split())
        print(f"读取TXT词向量：词汇量={vocab_size}，维度={vector_dim}")
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0]
            vector = np.array([float(v) for v in parts[1:]])
            vectors[word] = vector
    
    return vectors

# ===================== 主执行流程 =====================
if __name__ == "__main__":
    # 替换为你的“每行1句”语料路径
    CORPUS_PATH = r"data_fasttext\\re_tacred\\re_tacred_corpus.txt"
    # 模型保存路径
    MODEL_SAVE_PATH = r"data_fasttext\\re_tacred\\gensim_fasttext_768d_line.model"
    # TXT词向量保存路径
    TXT_SAVE_PATH = r"data_fasttext\\re_tacred\\fasttext_768d_vectors_line.txt"
    # PKL文件保存路径
    WORD2IDX_PKL_PATH = r"data_fasttext\\re_tacred\\word2idx_line.pkl"
    # TXT文件保存路径
    WORD2IDX_TXT_PATH = r"data_fasttext\\re_tacred\\word2idx_line.txt"
    
    # 1. 训练768维模型
    trained_model, word2idx = train_fasttext_768d_gensim(CORPUS_PATH, MODEL_SAVE_PATH, WORD2IDX_PKL_PATH, WORD2IDX_TXT_PATH)
    
    # 2. 导出TXT词向量
    export_vectors_to_txt(trained_model, TXT_SAVE_PATH, unk_token="<UNK>", unk_vector_type="mean")
    
    # 3. 验证TXT文件（可选）
    loaded_vectors = load_txt_vectors(TXT_SAVE_PATH)
    test_word = "computer"
    if test_word in loaded_vectors:
        print(f"\n验证：{test_word} 的向量维度={len(loaded_vectors[test_word])}")
        print(f"{test_word} 前5维：{loaded_vectors[test_word][:5]}")
