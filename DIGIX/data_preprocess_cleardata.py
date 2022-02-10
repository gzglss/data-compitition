import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import os
from tqdm.std import trange
from tqdm import tqdm
import json
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

INDEX = ['人物专栏', '作品分析', '情感解读', '推荐文', '攻略文', '治愈系文章', '深度事件', '物品评测', '科普知识文', '行业解读']
body_cut_len=400
labeled_train_path='../data/raw_data/labeled_train_data.json'
labeled_train_preprocess_path='../data/data_preprocess/labeled_cleared.json'
test_data_path='../data/raw_data/doc_quality_data_test.json'
test_data_preprocess_path='../data/data_preprocess/test_cleared.json'
unlabeled_path='../data/raw_data/unlabeled_train_data.json'
unlabeled_preprocess_path='../data/data_preprocess/unlabeled_cleared.json'

def get_sentences_list(raw_text: str):
    return [s for s in BeautifulSoup(raw_text, 'html.parser')._all_strings()]

def remove_symbol(string: str):
    return string.replace('\t', '').replace('\n', '').replace('\r', '')

def check_duplicate_title(input_path, output_path):
    duplicate = 0
    no_html = 0
    no_duplicate = 0
    print("Processing File: ", input_path)
    with open(input_path, "r", encoding='utf-8') as file, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(file):
            json_data = json.loads(line)
            title = json_data["title"]
            body = get_sentences_list(json_data["body"])
            title_length = len(title)

            # 正文中不含HTML标签
            if len(body) == 1:
                no_html += 1
                tmp_body = body[0]
                # 注意,这边re.sub的pattern使用了re.escape()
                # 是为了转译title中存在的会被re视为元字符的字符(例如"?"","*")
                # 事实上相当于"\".join(title)[将所有字符转译为普通字符]
                new_body = re.sub("(原标题：)?" + re.escape(title), "", tmp_body)
                new_body_length = len(new_body)

                if new_body_length == len(tmp_body):
                    no_duplicate += 1
                else:
                    duplicate += 1
                    # print('sub原标题')

            # 正文中包含HTML标签
            else:
                # print(body)
                # print(len(body))
                i = 0
                # 检查 标题是否出现在前两个元素中 (有可能存在标签<p class=\"ori_titlesource\">,会有"原标题: title"的情况出现)
                for sentence in body[:2]:
                    if title in sentence:
                        i += 1

                new_body = "".join(body[i:])#去除body中title的重复部分

                if i > 0:
                    duplicate += 1
                else:
                    no_duplicate += 1

            rm_whites_body = remove_symbol(new_body)
            rm_whites_title = remove_symbol(title)
            json_data["body"] = rm_whites_body
            json_data["title"] = rm_whites_title
            # json_data["length"] = check_length([len(rm_whites_body), len(rm_whites_title)])
            json.dump(json_data, outfile, ensure_ascii=False)
            outfile.write("\n")

    print("duplicate: {}\t no_html: {}, no_duplicate: {}\n".format(duplicate, no_html, no_duplicate))

def clear_data():
    print('开始初步处理有标注数据...')
    check_duplicate_title(labeled_train_path,labeled_train_preprocess_path)
    print('有标注数据初步处理完成...')
    print('')
    print('开始初步处理无标注数据...')
    check_duplicate_title(unlabeled_path,unlabeled_preprocess_path)
    print('无标注数据初步处理完成...')
    print('')
    print('开始初步处理测试集数据...')
    check_duplicate_title(test_data_path,test_data_preprocess_path)
    print('测试集初步处理完成...')

