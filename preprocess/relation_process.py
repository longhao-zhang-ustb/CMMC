import os
import sys
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_folder)

output_file = r'data\\TEST_FILE_NEW.TXT'
output = open(output_file, 'a+')
with open(r'data\\TEST_FILE_FULL.TXT', 'r') as file:
    lines = file.readlines()
    for line in range(0, len(lines), 4):
        # # 如果关系不是other,则进行保存，并删除实体方向
        if lines[line+1].strip() != 'Other':
            relation = lines[line+1][:-8]
            output.write(lines[line])
            output.write(relation+'\n')
            output.write(lines[line+2])
            output.write(lines[line+3])
        else:
            continue
