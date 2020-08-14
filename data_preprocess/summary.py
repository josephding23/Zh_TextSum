import json
import math


def nlpcc2017():
    data_path = 'D:/NLP/Datasets/nlpcc2017textsummarization/nlpcc2017textsummarization/train_with_summ.txt'
    save_dir = 'D:/NLP/Datasets/nlpcc2017textsummarization/formatted/'

    data_list = []

    test_src_str = ''
    test_tgt_str = ''

    train_src_str = ''
    train_tgt_str = ''

    val_src_str = ''
    val_tgt_str = ''

    for line in open(data_path, 'r', encoding='UTF-8'):
        txt_dict = json.loads(line)
        data_list.append(txt_dict)

    train_num = math.ceil(len(data_list) * 0.8)
    test_num = math.ceil(len(data_list) * 0.1)
    eval_num = len(data_list) - train_num - test_num
    print(train_num, test_num, eval_num)

    for i, txt_dict in enumerate(data_list):

        summarization = txt_dict['summarization']

        article = txt_dict['article']

        assert len(summarization) > 0 and len(article) > 0

        if i < train_num:
            train_tgt_str += summarization
            if summarization[-1] != '\n':
                train_tgt_str += '\n'
            train_src_str += article
            if article[-1] != '\n':
                train_src_str += '\n'
        elif i < train_num + test_num:
            test_tgt_str += summarization
            if summarization[-1] != '\n':
                test_tgt_str += '\n'
            test_src_str += article
            if article[-1] != '\n':
                test_src_str += '\n'
        else:
            val_tgt_str += summarization
            if summarization[-1] != '\n':
                val_tgt_str += '\n'
            val_src_str += article
            if article[-1] != '\n':
                val_src_str += '\n'

    with open(save_dir + 'train.source', 'w', encoding='UTF-8') as f:
        f.write(train_src_str)
    with open(save_dir + 'train.target', 'w', encoding='UTF-8') as f:
        f.write(train_tgt_str)
    with open(save_dir + 'test.source', 'w', encoding='UTF-8') as f:
        f.write(test_src_str)
    with open(save_dir + 'test.target', 'w', encoding='UTF-8') as f:
        f.write(test_tgt_str)
    with open(save_dir + 'val.source', 'w', encoding='UTF-8') as f:
        f.write(val_src_str)
    with open(save_dir + 'val.target', 'w', encoding='UTF-8') as f:
        f.write(val_tgt_str)


if __name__ == '__main__':
    nlpcc2017()