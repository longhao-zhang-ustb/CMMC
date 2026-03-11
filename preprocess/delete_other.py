def process_data(input_file, output_file):
    o_f = open(output_file, 'a+')
    with open(input_file, 'r', encoding='utf8') as file:
        for index, row in enumerate(file.readlines()):
            if '"label": 0' in row:
                pass
            else:
                o_f.write(row)
    
if __name__ == '__main__':
    input_file = 'data/train.txt'
    output_file = 'data/train_new.txt'
    input_test_file = 'data/test.txt'
    output_test_file = 'data/tes_new.txt'
    process_data(input_file, output_file)
    process_data(input_test_file, output_test_file)
