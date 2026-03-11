import os
# 统计下2套数据集中每种关系类别的实体之间的距离
# relation_array = ['Message-Topic', 'Product-Producer', 'Instrument-Agency', 'Entity-Destination', 'Cause-Effect', 'Component-Whole', 'Entity-Origin', 'Member-Collection', 'Content-Container']
relation_array = ['org2miscmulti', 'org2locmulti', 'org2org', 'org2per', 'per2miscmulti', 'per2locmulti', 'per2org', 'per2per']
if __name__ == '__main__':
    folder_path = r'20260218_experimental_data\\re_tacred'
    relation_counts = {}
    for split in ['train.txt', 'val.txt', 'test.txt']:
        split_file = os.path.join(folder_path, split)
        for relation in relation_array:
            relation_counts[relation] = []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                relation_counts[relation_array[eval(line)['label']]].append(abs(eval(line)['e2_begin']-eval(line)['e1_begin']))
        # 遍历relation_counts字典，计算每个关系类别的最小值、平均距离、最大值、众数
        print(f"{split} Relation Distance Statistics:")
        for relation in relation_array:
            min_distance = min(relation_counts[relation])
            avg_distance = sum(relation_counts[relation]) / len(relation_counts[relation])
            max_distance = max(relation_counts[relation])
            mode_distance = max(set(relation_counts[relation]), key=relation_counts[relation].count)
            print(f"{relation} - Min: {min_distance}, Avg: {avg_distance:.2f}, Max: {max_distance}, Mode: {mode_distance}")

"""
semeval_2010
train.txt Relation Distance Statistics:
Message-Topic - Min: 1, Avg: 5.25, Max: 16, Mode: 3
Product-Producer - Min: 1, Avg: 5.16, Max: 30, Mode: 4
Instrument-Agency - Min: 1, Avg: 5.81, Max: 22, Mode: 3
Entity-Destination - Min: 2, Avg: 4.75, Max: 15, Mode: 3
Cause-Effect - Min: 1, Avg: 5.56, Max: 33, Mode: 4
Component-Whole - Min: 1, Avg: 4.13, Max: 24, Mode: 3
Entity-Origin - Min: 1, Avg: 4.77, Max: 24, Mode: 5
Member-Collection - Min: 1, Avg: 2.93, Max: 17, Mode: 3
Content-Container - Min: 1, Avg: 4.75, Max: 24, Mode: 4
val.txt Relation Distance Statistics:
Message-Topic - Min: 2, Avg: 5.11, Max: 12, Mode: 3
Product-Producer - Min: 1, Avg: 3.97, Max: 13, Mode: 4
Instrument-Agency - Min: 1, Avg: 4.88, Max: 18, Mode: 3
Entity-Destination - Min: 2, Avg: 4.49, Max: 13, Mode: 3
Cause-Effect - Min: 1, Avg: 4.34, Max: 13, Mode: 3
Component-Whole - Min: 1, Avg: 4.31, Max: 16, Mode: 3
Entity-Origin - Min: 1, Avg: 4.78, Max: 13, Mode: 4
Member-Collection - Min: 2, Avg: 2.95, Max: 8, Mode: 2
Content-Container - Min: 2, Avg: 4.44, Max: 12, Mode: 3
test.txt Relation Distance Statistics:
Message-Topic - Min: 1, Avg: 5.31, Max: 14, Mode: 3
Product-Producer - Min: 1, Avg: 5.29, Max: 20, Mode: 4
Instrument-Agency - Min: 1, Avg: 6.22, Max: 15, Mode: 3
Entity-Destination - Min: 2, Avg: 4.87, Max: 12, Mode: 3
Cause-Effect - Min: 1, Avg: 5.67, Max: 20, Mode: 4
Component-Whole - Min: 1, Avg: 3.88, Max: 13, Mode: 3
Entity-Origin - Min: 1, Avg: 4.60, Max: 16, Mode: 5
Member-Collection - Min: 1, Avg: 3.05, Max: 10, Mode: 3
Content-Container - Min: 1, Avg: 4.93, Max: 16, Mode: 4
"""

"""
org2miscmulti - Min: 1, Avg: 13.43, Max: 74, Mode: 4
org2locmulti - Min: 1, Avg: 14.30, Max: 77, Mode: 4 
org2org - Min: 1, Avg: 12.62, Max: 79, Mode: 4
org2per - Min: 1, Avg: 12.07, Max: 81, Mode: 7
per2miscmulti - Min: 1, Avg: 11.64, Max: 81, Mode: 1
per2locmulti - Min: 1, Avg: 13.95, Max: 71, Mode: 3
per2org - Min: 1, Avg: 12.72, Max: 89, Mode: 3     
per2per - Min: 1, Avg: 12.89, Max: 76, Mode: 3
val.txt Relation Distance Statistics:
org2miscmulti - Min: 1, Avg: 14.11, Max: 72, Mode: 5
org2locmulti - Min: 1, Avg: 14.58, Max: 58, Mode: 5 
org2org - Min: 1, Avg: 12.50, Max: 71, Mode: 5      
org2per - Min: 1, Avg: 12.00, Max: 62, Mode: 6      
per2miscmulti - Min: 1, Avg: 11.31, Max: 79, Mode: 5
per2locmulti - Min: 1, Avg: 12.63, Max: 66, Mode: 3 
per2org - Min: 1, Avg: 12.76, Max: 66, Mode: 4
per2per - Min: 1, Avg: 12.74, Max: 81, Mode: 3
test.txt Relation Distance Statistics:
org2miscmulti - Min: 1, Avg: 13.69, Max: 77, Mode: 4
org2locmulti - Min: 1, Avg: 14.31, Max: 60, Mode: 3
org2org - Min: 2, Avg: 12.47, Max: 81, Mode: 6
org2per - Min: 2, Avg: 11.80, Max: 85, Mode: 6
per2miscmulti - Min: 1, Avg: 11.06, Max: 72, Mode: 3
per2locmulti - Min: 1, Avg: 12.52, Max: 56, Mode: 3
per2org - Min: 1, Avg: 11.83, Max: 58, Mode: 3
per2per - Min: 1, Avg: 11.54, Max: 71, Mode: 3
"""
