import torch

def count_non_bert_params(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']  # 因为你的保存格式是 {'state_dict': ...}

    total_non_bert = 0
    # 也可以分组统计，便于了解各模块规模
    module_counts = {}

    for key, tensor in state_dict.items():
        if key.startswith('word_emb.'):
            continue  # 跳过 BERT 部分

        numel = tensor.numel()
        total_non_bert += numel

        # 按模块前缀分组统计
        prefix = key.split('.')[0]  # 取第一个点之前的部分作为模块名
        module_counts[prefix] = module_counts.get(prefix, 0) + numel

    # 打印各模块参数量，以M为单位
    print("非 BERT 各部分参数量：")
    for mod, cnt in module_counts.items():
        print(f"  {mod}: {cnt / 1e6:.4f}M")
    print(f"总计非 BERT 参数量: {total_non_bert / 1e6:.4f}M")
    return total_non_bert

# 使用
file_path = r'experiments\model_exp\bert_semeval_task8\cmmc_semeval_seed42_bert_best.pth.tar'
params = count_non_bert_params(file_path)
