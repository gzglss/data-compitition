# from transformers import TFBertModel

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer,TFBertForSequenceClassification
import jieba
import tensorflow as tf
from tqdm.std import trange
import numpy as np


maxlen = 128
bert_config = ('D:\\BERT预训练模型\\Bert_base_chinese\\bert_config.json')
bert_init_ckpt = ('D:\\BERT预训练模型\\Bert_base_chinese\\bert_model.ckpt')
bert_vocab = ('D:\\BERT预训练模型\\Bert_base_chinese\\vocab.txt')
model_path='D:\\BERT预训练模型\\Bert_base_chinese\\'

path_train='./raw_data/train_dataset.xlsx'
path_company='./raw_data/company_name_entity.xlsx'
data_1=pd.read_excel(path_train)
data_1_com=pd.read_excel(path_company)
data_1=data_1.drop(0,axis=0)
train_list=['NEWS_BASICINFO_SID','NEWS_TITLE','CONTENT','COMPANY_NM','LABEL']
data_2=data_1[train_list]
data_2.reset_index(inplace=True,drop=True)
#为风险标签编码
data_2=data_2.sample(frac=1.0)#pandas自带的打乱函数，frac=1.0:表示返回的比例，1.0表示返回全部的数据框数据
data_2['LABEL']=LabelEncoder().fit_transform(data_2['LABEL'])
#去停用词语料库,其实这个停用词也把很多重要的词去掉了，所以慎重考虑
fenci=open(r'E:\\BERT预训练模型\\停用词.txt','r',encoding='utf-8')
stopkey = [line.strip() for line in fenci.readlines()]
def div_words(seg):
    words=jieba.cut(seg)
    return ' '.join(list(words))

com_jieba=data_1_com['公司名'].apply(div_words)
title_jieba=data_2['NEWS_TITLE'].apply(div_words)

train_x,val_x,train_label,val_label=train_test_split(title_jieba,data_2['LABEL'],random_state=100,train_size=0.8)


tokenizer=BertTokenizer.from_pretrained('E:\\BERT预训练模型\\Bert_base_chinese\\vocab.txt')#@classmethod
train_encoding=tokenizer(list(train_x),truncation=True,padding=True,max_length=32,return_tensors='tf')
val_encoding=tokenizer(list(val_x),truncation=True,padding=True,max_length=32,return_tensors='tf')
train_label=tf.constant(train_label.values)
val_label=tf.constant(val_label.values)
dict_input={}
dict_input.update({'input_ids':train_encoding['input_ids'],'token_type_ids':train_encoding['token_type_ids'],'attention_mask':train_encoding['attention_mask']})
#计算位置向量

# max_len=title_jieba.apply(len).max()
data_len=[len(i) for i in train_encoding['input_ids']]
max_len=max(data_len)
position_ids=[]
# org_posit = tf.Variable(tf.zeros((max_len,)))
print(max_len)
train_len=len(train_x)
for i in trange(train_len):
    arr_train=[]
    for j in range(32):
        if train_encoding['input_ids'][i,j]!=0:
            arr_train.append(j+1)
        else:
            arr_train.append(0)
    position_ids.append(arr_train)
position_ids=tf.constant(position_ids)
# train_position=train_x.apply(position_vec)
dict_input.update({'position_ids':position_ids,'labels':train_label})

position_ids_val=[]
val_len=len(val_x)
for i in trange(val_len):
    arr_val=[]
    for j in range(32):
        if val_encoding['input_ids'][i,j]!=0:
            arr_val.append(j+1)
        else:
            arr_val.append(0)
    position_ids_val.append(arr_val)
position_ids_val=tf.constant(position_ids_val)
dict_val={}
dict_val.update({'input_ids':val_encoding['input_ids'],'token_type_ids':val_encoding['token_type_ids'],'attention_mask':val_encoding['attention_mask']})
# val_position=val_x.apply(position_vec)
dict_val.update({'position_ids':position_ids_val,'labels':val_label})
df_val=pd.DataFrame(dict_val)
print(df_val)

bert=TFBertForSequenceClassification.from_pretrained(model_path)
model=bert(dict_input)
print(model)


