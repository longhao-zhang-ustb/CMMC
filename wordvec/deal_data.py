if __name__ == "__main__":
    # # 读取train.txt, test.txt和val.txt文件，并将它们的内容合并到一个新文件中
    # for filename in [r"data\\semeval\\train.txt", r"data\\semeval\\test.txt", r"data\\semeval\\val.txt"]:
    #     with open(filename, "r", encoding="utf-8") as f:
    #         lines = f.readlines()
    #     with open(r"data\\semeval\\corpus_step1.txt", "a", encoding="utf-8") as f:
    #         f.writelines([line.strip() + "\n" for line in lines])
    # exit()
    # 读取data\re_tacred\corpus_step1.txt文件
    content = []
    with open(r"data\\semeval\\corpus_step1.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = eval(line.strip())['sentence']
            content.append(line)
    # 将content写入data\re_tacred\corpus.txt文件
    with open(r"data\\semeval\\corpus.txt", "w", encoding="utf-8") as f:
        for line in content:
            f.write(line + "\n")
