# -*- coding:utf-8 -*-

import json
import math
from harvesttext import HarvestText
import re


def nlpcc2017():

    data_path = 'D:/NLP/datasets/nlpcc2017textsummarization/nlpcc2017textsummarization/train_with_summ.txt'
    save_dir = '../datasets/nlpcc2017textsummarization/formatted/'

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

        summarization = txt_dict['summarization'].replace('\n', '').replace('\r', '')
        article = txt_dict['article'].replace('\n', '').replace('\r', '')

        summarization = clean_text_whole(summarization)
        article = clean_text_whole(article)

        if len(summarization) == 0 or len(article) == 0:
            continue

        assert len(summarization) > 0 and len(article) > 0

        if i < train_num:
            if i == train_num - 1:
                train_tgt_str += summarization
                train_src_str += article
            else:
                train_tgt_str += summarization + '\n'
                train_src_str += article + '\n'

        elif i < train_num + test_num:
            if i == train_num + test_num - 1:
                test_tgt_str += summarization
                test_src_str += article
            else:
                test_tgt_str += summarization + '\n'
                test_src_str += article + '\n'

        else:
            if i == len(data_list) - 1:
                val_tgt_str += summarization
                val_src_str += article
            else:
                val_tgt_str += summarization + '\n'
                val_src_str += article + '\n'

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


def post_test():
    save_dir = '../datasets/nlpcc2017textsummarization/formatted/'

    with open(save_dir + 'val.target', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line == '\n':
                print(line, i)


def clean_text_whole(original_text):
    ht = HarvestText()

    original_text = re.compile(r'【.*?】').sub('', original_text)  # 去掉方括号
    original_text = re.compile(r'(\d{4}-\d{2}-\d{2})').sub('', original_text)  # 去掉日期
    original_text = re.compile(r'(\d{2}:\d{2}:\d{2})').sub('', original_text)  # 去掉时间
    original_text = re.compile(r'(\d{2}:\d{2})').sub('', original_text)  # 去掉时间
    cleaned_text = ht.clean_text(original_text)

    return cleaned_text


def text_clean_test():

    original_text = '''
航班资料图来源:央视<Paragraph>2014-02-25<Paragraph>20:28【天津飞沈阳航班已经安全降落】据央视消息，之前因起落架故障无法降落BK2870次航班已安全降落，并且是正常降落非迫降。据悉机上载有30名乘客。
    '''

    print(clean_text_whole(original_text))


def test_max_length():
    max_article = 0
    max_article_index = 0
    max_abstract = 0
    max_abstract_index = 0

    save_dir = '../datasets/nlpcc2017textsummarization/formatted/'

    with open(save_dir + 'train.source', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) > max_article:
                max_article = len(line)
                max_article_index = i

    with open(save_dir + 'test.source', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) > max_article:
                max_article = len(line)
                max_article_index = i + 40000

    with open(save_dir + 'val.source', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) > max_article:
                max_article = len(line)
                max_article_index = i + 45000

    with open(save_dir + 'train.target', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) > max_abstract:
                max_abstract = len(line)
                max_abstract_index = i

    with open(save_dir + 'test.target', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) > max_abstract:
                max_abstract = len(line)
                max_abstract_index = i + 40000

    with open(save_dir + 'val.target', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) > max_abstract:
                max_abstract = len(line)
                max_abstract_index = i + 45000

    print(max_article, max_article_index, max_abstract, max_abstract_index)


def text_length_test():
    save_dir = '../datasets/nlpcc2017textsummarization/formatted/'

    over_num = 0
    ref_length = 4096

    with open(save_dir + 'train.source', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) > ref_length:
                over_num += 1

    with open(save_dir + 'test.source', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) > ref_length:
                over_num += 1

    with open(save_dir + 'val.source', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) > ref_length:
                over_num += 1

    print(over_num)


if __name__ == '__main__':
    nlpcc2017()
