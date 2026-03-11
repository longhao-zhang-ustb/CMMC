import gensim
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
import pickle
import os

nltk.download('punkt_tab')
nltk.download('stopwords')

class TqdmCallback(CallbackAny2Vec):
    def __init__(self, epochs):
        self.epochs = epochs
        self.pbar = None

    def on_train_begin(self, model):
        # 训练开始时初始化进度条
        self.pbar = tqdm(total=self.epochs, desc="Training Word2Vec")

    def on_epoch_end(self, model):
        # 每个 epoch 结束更新进度条
        self.pbar.update(1)

    def on_train_end(self, model):
        # 训练结束关闭进度条
        self.pbar.close()
        
tqdm_callback = TqdmCallback(epochs=10)

# 加载英文语料
corpus_path = r'data\\semeval\\corpus.txt'

# 读取语料并分词
sentences = []
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        # 转小写
        line = line.lower()
        # 分词
        tokens = word_tokenize(line)
        # 去掉标点符号
        tokens = [word for word in tokens if word.isalpha()]
        # 去掉停用词
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        if tokens:
            sentences.append(tokens)

epochs = 10
# ====== 2. 训练 Word2Vec ======
model = Word2Vec(
    sentences=sentences,
    vector_size=768,      # 词向量维度
    window=5,             # 上下文窗口大小
    min_count=1,          # 忽略出现次数少于5的词
    sg=1,                 # 1 表示 Skip-gram，0 表示 CBOW
    hs=0,                 # 0 表示使用负采样，1 表示层次softmax
    negative=5,           # 负采样的数量
    workers=4,             # 线程数
    epochs=epochs,
    callbacks=[tqdm_callback]
)

# ====== 新增：生成word2idx.pkl和word2idx.txt ======
def generate_and_save_word2idx(model, pkl_save_path, txt_save_path, unk_token="<UNK>"):
    """
    基于训练好的Word2Vec模型生成word2idx，并保存为pkl和txt
    :param model: 训练好的Word2Vec模型
    :param pkl_save_path: word2idx.pkl保存路径
    :param txt_save_path: word2idx.txt保存路径
    :param unk_token: 未知词标识（索引0）
    """
    # 1. 获取模型词表（和词向量顺序完全一致）
    vocab = model.wv.index_to_key
    
    # 2. 生成word2idx：0号索引给<UNK>，真实单词从1开始
    word2idx = {unk_token: 0}
    for idx, word in enumerate(vocab):
        word2idx[word] = idx + 1
    
    # 3. 确保保存路径的文件夹存在（避免报错）
    for save_path in [pkl_save_path, txt_save_path]:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    # 4. 保存word2idx.pkl（二进制，供程序读取）
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(word2idx, f)
    print(f"\nword2idx.pkl已保存至：{pkl_save_path}")
    
    # 5. 保存word2idx.txt（文本，供人工查看，按索引升序排列）
    sorted_word2idx = sorted(word2idx.items(), key=lambda x: x[1])
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write("# 单词 索引（0=<UNK>，后续为Word2Vec词表顺序）\n")
        for word, idx in sorted_word2idx:
            f.write(f"{word} {idx}\n")
    print(f"word2idx.txt已保存至：{txt_save_path}")
    
    # 打印统计信息
    print(f"词表总大小（含<UNK>）：{len(word2idx)}")
    print(f"示例：<UNK> → 索引{word2idx[unk_token]}")
    print(f"示例：第一个真实单词「{vocab[0]}」→ 索引{word2idx[vocab[0]]}")

# 可选：生成含<UNK>向量的词向量txt（<UNK>向量为所有真实向量的均值）
def export_word2vec_with_unk(model, save_path, unk_token="<UNK>"):
    vocab = model.wv.index_to_key
    vector_dim = model.vector_size
    # 计算<UNK>的均值向量
    unk_vector = model.wv.vectors.mean(axis=0)
    # 写入文件
    with open(save_path, 'w', encoding='utf-8') as f:
        # 首行：总词汇量（含UNK） 维度
        f.write(f"{len(vocab)+1} {vector_dim}\n")
        # 第一行：<UNK> + 均值向量
        f.write(f"{unk_token} {' '.join([f'{v:.6f}' for v in unk_vector])}\n")
        # 后续行：真实单词 + 向量
        for word in vocab:
            f.write(f"{word} {' '.join([f'{v:.6f}' for v in model.wv[word]])}\n")
    print(f"\n含<UNK>的词向量文件已保存：{save_path}")

# ====== 3. 保存模型 ======
model.save(r"data\\semeval\\word2vec_768d.model")
# model.wv.save_word2vec_format(r"data\\re_tacred\\word2vec_768d.txt", binary=False)
export_word2vec_with_unk(
    model=model,
    save_path=r"data\\semeval\\word2vec_768d.txt",
    unk_token="<UNK>"
)
print("训练完成，模型已保存为 word2vec_768d.model 和 word2vec_768d.txt")
# 调用函数生成并保存word2idx
generate_and_save_word2idx(
    model=model,
    pkl_save_path=r"data\\semeval\\word2idx.pkl",
    txt_save_path=r"data\\semeval\\word2idx.txt",
    unk_token="<UNK>"
)

print("\n训练完成，所有文件已保存：")
