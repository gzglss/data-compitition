import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from tqdm.std import trange
from tqdm import tqdm
import random
import json
import sys
import transformers as tfs

warnings.filterwarnings('ignore')

labeled_preprocess_path='../data/data_preprocess/labeled_cleared.json'
unlabeled_preprocess_path='../data/data_preprocess/unlabeled_cleared.json'
test_preprocess_path='../data/data_preprocess/test_cleared.json'

labeled_path_front='../data/data_preprocess/labeled_f512.csv'
unlabeled_path_front='../data/data_preprocess/unlabeled_f512.csv'
test_path_front='../data/data_preprocess/test_f512.csv'

labeled_path_middle='../data/data_preprocess/labeled_m512.csv'
unlabeled_path_midle='../data/data_preprocess/unlabeled_m512.csv'
test_path_midle='../data/data_preprocess/test_m512.csv'

labeled_path_front_behind='../data/data_preprocess/labeled_f_b.csv'#两句输入（title与body未合并）
unlabeled_path_front_behind='../data/data_preprocess/unlabeled_f_b.csv'#两句输入（title与body未合并）
test_path_front_behind='../data/data_preprocess/test_f_b.csv'#两句输入（title与body未合并）

INDEX = ['人物专栏', '作品分析', '情感解读', '推荐文', '攻略文', '治愈系文章', '深度事件', '物品评测', '科普知识文', '行业解读']
punctuation='[？?,。.！!]'

def prepare_sequence(title: str, body: str):
    if len(body)==0:
        return title,title
    else:
        return (title, body[:512])
    
def prepare_sequence_fb(title:str,body:str):
    if len(body)==0:
        return (title,title)
    else:
        return (title, body[:256]+'。'+body[-256:])
    
def split_body(text):
    text1=re.split(punctuation,text)
    return text1

#只用到了前面256和后面256个字，前向提取与中间提取未用到
def extract_front_behind_data():
    labeled_data=pd.read_json(labeled_preprocess_path,orient='records',lines=True)
    unlabeled_data=pd.read_json(unlabeled_preprocess_path,orient='records',lines=True)
    test_data=pd.read_json(test_preprocess_path,orient='records',lines=True)
    
    print('开始提取前后句子...')
    labeled_text_fb=[]#构造两句输入?
    for i in trange(len(labeled_data)):
        labeled_text_fb.append(prepare_sequence_fb(labeled_data['title'][i],labeled_data['body'][i]))
    labeled_data['text']=labeled_text_fb
    labeled_csv=labeled_data[['id', 'category', 'doctype', 'text']]
    labeled_csv.to_csv(labeled_path_front_behind,index=False,encoding='utf-8')

    labeled_text_fb2=[]#构造两句输入?
    for i in trange(len(unlabeled_data)):
        labeled_text_fb2.append(prepare_sequence_fb(unlabeled_data['title'][i],unlabeled_data['body'][i]))
    unlabeled_data['text']=labeled_text_fb2
    unlabeled_csv=unlabeled_data[['id', 'category', 'doctype', 'text']]
    unlabeled_csv.to_csv(unlabeled_path_front_behind,index=False,encoding='utf-8')

    labeled_text_fb3=[]#构造两句输入?
    for i in trange(len(test_data)):
        labeled_text_fb3.append(prepare_sequence_fb(test_data['title'][i],test_data['body'][i]))
    test_data['text']=labeled_text_fb3
    test_csv=test_data[['id', 'category','text']]
    test_csv.to_csv(test_path_front_behind,index=False,encoding='utf-8')
    print('over')

def prepare_sequence_m(title:str,body:str):
    if len(body)<=max_seq_len:
        return title,body
    elif len(body)>max_seq_len and len(body)<=2*max_seq_len:
        gap_=len(body)-max_seq_len
        return title,body[gap_:]
    elif len(body)>2*max_seq_len and len(body)<=4*max_seq_len:
        return title,body[max_seq_len:2*max_seq_len]
    else:
        textlen=[]
        lensum=0
        begin_id=0
        text_list=split_body(body)
        for i in text_list:
            textlen.append(len(i))
        med_len=np.sum(textlen)//2
        for i in range(len(textlen)):
            lensum+=textlen[i]
            if lensum>=med_len:
                begin_id=i
                break
        get_text=''
        for i in text_list[begin_id:]:
            get_text+=i
            get_text_='。'
            if len(get_text)>=512:
                break
        return title,get_text

def extract_front_data():
    labeled_data=pd.read_json(labeled_preprocess_path,orient='records',lines=True)
    unlabeled_data=pd.read_json(unlabeled_preprocess_path,orient='records',lines=True)
    test_data=pd.read_json(test_preprocess_path,orient='records',lines=True)

    print('提取文章前512个字')
    labeled_text=[]#构造两句输入?
    for i in trange(len(labeled_data)):
        labeled_text.append(prepare_sequence(labeled_data['title'][i],labeled_data['body'][i]))
    labeled_data['text']=labeled_text
    labeled_csv=labeled_data[['id', 'category', 'doctype', 'text']]
    labeled_csv.to_csv(labeled_path_front,index=False,encoding='utf-8')

    labeled_text2=[]#构造两句输入?
    for i in trange(len(unlabeled_data)):
        labeled_text2.append(prepare_sequence(unlabeled_data['title'][i],unlabeled_data['body'][i]))
    unlabeled_data['text']=labeled_text2
    unlabeled_csv=unlabeled_data[['id', 'category', 'doctype', 'text']]
    unlabeled_csv.to_csv(unlabeled_path_front,index=False,encoding='utf-8')

    labeled_text3=[]#构造两句输入?
    for i in trange(len(test_data)):
        labeled_text3.append(prepare_sequence(test_data['title'][i],test_data['body'][i]))
    test_data['text']=labeled_text3
    test_csv=test_data[['id', 'category','text']]
    test_csv.to_csv(test_path_front,index=False,encoding='utf-8')
    print('over')

def extract_med_data():
    labeled_data=pd.read_json(labeled_preprocess_path,orient='records',lines=True)
    unlabeled_data=pd.read_json(unlabeled_preprocess_path,orient='records',lines=True)
    test_data=pd.read_json(test_preprocess_path,orient='records',lines=True)
    
    print('开始提取中间句子')
    labeled_text=[]#构造两句输入?
    for i in trange(len(labeled_data)):
        labeled_text.append(prepare_sequence_m(labeled_data['title'][i],labeled_data['body'][i]))
    labeled_data['text']=labeled_text
    labeled_csv=labeled_data[['id', 'category', 'doctype', 'text']]
    labeled_csv.to_csv(labeled_path_middle,index=False,encoding='utf-8')

    labeled_text2=[]#构造两句输入?
    for i in trange(len(unlabeled_data)):
        labeled_text2.append(prepare_sequence_m(unlabeled_data['title'][i],unlabeled_data['body'][i]))
    unlabeled_data['text']=labeled_text2
    unlabeled_csv=unlabeled_data[['id', 'category', 'doctype', 'text']]
    unlabeled_csv.to_csv(unlabeled_path_midle,index=False,encoding='utf-8')

    labeled_text3=[]#构造两句输入?
    for i in trange(len(test_data)):
        labeled_text3.append(prepare_sequence_m(test_data['title'][i],test_data['body'][i]))
    test_data['text']=labeled_text3
    test_csv=test_data[['id', 'category','text']]
    test_csv.to_csv(test_path_midle,index=False,encoding='utf-8')
    print('over')