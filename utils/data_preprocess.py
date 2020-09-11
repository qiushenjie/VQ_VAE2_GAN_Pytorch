import os
import numpy as np

def preReadfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
        src_data_pairs = []
        for i in res:
            p = i.split(' ') # （图片名称， 类别）
            src_data_pairs.append([p[0], p[1]])
        return src_data_pairs

def create_label(file, src_data_pairs, start, end, channels_number):
    curr_src_data_pairs = np.array(src_data_pairs)
    for i in range(start, end - channels_number + 1):
        continuous_pairs = [curr_src_data_pairs[j][0] for j in range(i, i + channels_number + 1)]
        label = continuous_pairs.pop(channels_number // 2)
        continuous_pairs.append(label)
        file.write(' '.join(continuous_pairs) + ' ' + src_data_pairs[start][1] + '\n')
        # file.write(' '.join(continuous_pairs) + '\n')
        # print(continuous_pairs)

    pass

def pairs_generator(channels_number, src_data_pairs):
    file = open('../data/train_pair.txt', 'w')
    data_length = len(src_data_pairs)
    index = 0
    start = 0
    end = 0
    curr_label = src_data_pairs[0][1]
    label_count = 0
    pair = []
    while index < data_length:
        if src_data_pairs[index][1] == curr_label:
            label_count += 1
        else:
            curr_start = start
            end = index - 1
            start = index
            curr_end = end  # 当前label的最后一个label
            pair.append([curr_start, curr_end])
            if curr_end - curr_start + 1 > channels_number:
                create_label(file, src_data_pairs, curr_start, curr_end, channels_number)
            curr_label = src_data_pairs[index][1]
            label_count = 0
        index += 1

    # 最后一个类别图片
    curr_start = start
    end = index - 1
    start = index
    curr_end = end  # 当前label的最后一个label
    pair.append([curr_start, curr_end])
    if curr_end - curr_start + 1 > channels_number:
        create_label(file, src_data_pairs, curr_start, curr_end, channels_number)

    return pair

def main():
    channels_number = 2
    src_data_pairs = preReadfile('../data/valid_series_label.txt')
    pair = pairs_generator(channels_number, src_data_pairs)

main()