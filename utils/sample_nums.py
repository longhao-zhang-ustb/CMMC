# 统计下2套数据集中每种关系类别的样本数量，包括训练集、验证集和测试集
import os
# relation_array = ['Message-Topic', 'Product-Producer', 'Instrument-Agency', 'Entity-Destination', 'Cause-Effect', 'Component-Whole', 'Entity-Origin', 'Member-Collection', 'Content-Container']
relation_array = ['org2miscmulti', 'org2locmulti', 'org2org', 'org2per', 'per2miscmulti', 'per2locmulti', 'per2org', 'per2per']
if __name__ == '__main__':
    folder_path = r'20260218_experimental_data\\re_tacred'
    relation_counts = {}
    for split in ['train.txt', 'val.txt', 'test.txt']:
        split_file = os.path.join(folder_path, split)
        for relation in relation_array:
            relation_counts[relation] = 0
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                relation_counts[relation_array[eval(line)['label']]] += 1
        print(f"{split} Relation Counts: {relation_counts}")
        
"""
train.txt Relation Counts: {
    'Message-Topic': 629, 
    'Product-Producer': 642, 
    'Instrument-Agency': 455, 
    'Entity-Destination': 835, 
    'Cause-Effect': 903, 
    'Component-Whole': 894, 
    'Entity-Origin': 649, 
    'Member-Collection': 686, 
    'Content-Container': 504
}
val.txt Relation Counts: {
    'Message-Topic': 61, 
    'Product-Producer': 136, 
    'Instrument-Agency': 85, 
    'Entity-Destination': 77, 
    'Cause-Effect': 167, 
    'Component-Whole': 110, 
    'Entity-Origin': 127, 
    'Member-Collection': 42, 
    'Content-Container': 80
}
test.txt Relation Counts: {
    'Message-Topic': 205, 
    'Product-Producer': 170, 
    'Instrument-Agency': 120, 
    'Entity-Destination': 225, 
    'Cause-Effect': 261, 
    'Component-Whole': 249, 
    'Entity-Origin': 198, 
    'Member-Collection': 195, 
    'Content-Container': 148
}
"""

"""
train.txt Relation Counts: {
    'org2miscmulti': 7990, 
    'org2locmulti': 4809, 
    'org2org': 7102, 
    'org2per': 6363, 
    'per2miscmulti': 10635, 
    'per2locmulti': 4723, 
    'per2org': 3673, 
    'per2per': 13170}
val.txt Relation Counts: {
    'org2miscmulti': 2350, 
    'org2locmulti': 1466, 
    'org2org': 2030, 
    'org2per': 1514, 
    'per2miscmulti': 4778, 
    'per2locmulti': 1920, 
    'per2org': 1041, 
    'per2per': 4485}
test.txt Relation Counts: {
    'org2miscmulti': 1413, 
    'org2locmulti': 702, 
    'org2org': 1235, 
    'org2per': 1035, 
    'per2miscmulti': 3012, 
    'per2locmulti': 1431, 
    'per2org': 784, 
    'per2per': 3806}
"""
