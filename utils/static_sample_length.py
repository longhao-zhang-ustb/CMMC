import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter  # 导入刻度格式化器

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

def format_with_thousands_separator(num):
    """
    为数值添加千位分隔符
    - 整数直接格式化
    - 浮点数保留2位小数后格式化
    """
    if isinstance(num, int):
        return f"{num:,}"
    elif isinstance(num, float):
        # 先保留2位小数，再添加千位分隔符
        return f"{num:,.2f}"
    else:
        return str(num)

def y_axis_formatter(x, pos):
    """
    自定义纵坐标刻度格式化函数
    x: 刻度数值
    pos: 刻度位置（无需使用，仅为兼容接口）
    """
    return format_with_thousands_separator(int(x))

# ---------------------- 1. 数据准备 ----------------------
is_train = False
is_test = True
file_path = r'20260218_experimental_data\\semeval\\test.txt'
output_path = r'20260218_experimental_data\\fig_save\\semeval_test.png'
# 读取数据文件，提取句子并计算长度
sentences = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        sentence = eval(line)['sentence']  # 提取句子
        sentences.append(sentence)

# 计算每个句子的长度
sentence_lengths = [len(sent.split(' ')) for sent in sentences]

# ---------------------- 2. 计算统计指标 ----------------------
sample_count = len(sentence_lengths)  # 样本数量
mean_length = np.mean(sentence_lengths)  # 均值
max_length = np.max(sentence_lengths)  # 最大值
min_length = np.min(sentence_lengths)  # 最小值

# ---------------------- 3. 绘制直方图 ----------------------
# 移除中文字体设置（英文无需特殊设置）
plt.rcParams['axes.unicode_minus'] = False  # 仅保留负号显示修复

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制直方图 # #8D0202 #1714C5 #157908
n, bins, patches = ax.hist(sentence_lengths, bins=80, color="#157908" if is_test else ('#8D0202' if is_train else '#1714C5'), rwidth=1, edgecolor=None)

# 添加均值竖线
# 为纵坐标设置自定义格式化器
ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
ax.axvline(mean_length, color='orange', linestyle='--', linewidth=2, label=f'Average Length')

# ---------------------- 4. 添加文本标注 ----------------------
# 构造英文标注文本
label_width = 14
# 冒号右侧数值左对齐，冒号左侧标签右对齐
annotation_text = (
    f'Samples = {format_with_thousands_separator(sample_count)}\n'
    f'Average Length = {format_with_thousands_separator(mean_length)}\n'
    f'Max Length = {format_with_thousands_separator(max_length)}\n'
    f'Min Length = {format_with_thousands_separator(min_length)}'
)

# 在图表上添加文本标注（位置可以根据需要调整）
ax.text(0.7, 0.95, annotation_text, transform=ax.transAxes, fontdict={'fontsize': 18}, linespacing=2,
        verticalalignment='top')

# ---------------------- 5. 图表美化 ----------------------
ax.set_xlabel('Sentence Length', fontsize=18)
ax.set_ylabel('Frequency (Number of Sentences)', fontsize=18)
# 设置横坐标刻度字体大小
ax.tick_params(axis='x', labelsize=18)
# 设置纵坐标刻度字体大小
ax.tick_params(axis='y', labelsize=18)
# ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()  # 自动调整布局

# 显示图表
plt.show()

# 以1000dpi保存图表
fig.savefig(output_path, dpi=1000)
