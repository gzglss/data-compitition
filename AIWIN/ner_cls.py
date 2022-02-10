import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from transformers import BertTokenizer
import Levenshtein
from transformers import TFBertForSequenceClassification
import re
from collections import Counter

def data_space(s):
    str_s=s.replace(' ',',')
    return str_s

#相似度计算
def get_equal_rate(str1, str2):
    return Levenshtein.ratio(str1, str2)

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
le=LabelEncoder()
le.fit(data_2['LABEL'])
map_cls=le.classes_
wu_ids=list(map_cls).index('无')
# print(wu_ids)
data_2['LABEL']=LabelEncoder().fit_transform(data_2['LABEL'])
# print(data_2['LABEL'])
data_2['NEWS_TITLE']=data_2['NEWS_TITLE'].apply(data_space)
train_x,val_x,train_label,val_label=train_test_split(data_2['NEWS_TITLE'],data_2['LABEL'],random_state=100,train_size=0.8)
val_x_val,val_x_test,val_label_val,val_label_test=train_test_split(val_x,val_label,random_state=100,train_size=0.5)

tokenizer=BertTokenizer.from_pretrained('./bert_pretrain_model/')

model_path='bert_pretrain_model'
model_cls=TFBertForSequenceClassification.from_pretrained(model_path,num_labels=13)
laearning_rate=1e-5
number_of_epochs=10
optimizer=tf.keras.optimizers.Adam(learning_rate=laearning_rate,epsilon=1e-8)
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)#返回值是否进行softmax？为True会使结果更加稳定
metric=tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model_cls.compile(optimizer=optimizer,loss=loss,metrics=[metric])

model_cls.load_weights('./model_and_trainingcode/tf_cls_weight.h5')

#文本处理
def data_deal_ner(s):
#     print(str(s))
    global data_str
    data_str=''
    if '|' in s:
        pattern1=re.compile(r'(.*)\|(.*)')
        s=pattern1.match(s).group(2)

    pattern=re.compile(r'\D*')
    s1=pattern.findall(s)
#     print(s1)
    s2=[i for i in s1 if i!='']
#     print(s2)
    if len(s2)==1:
        s=s2[0]
#         print(s)
        return s
    if len(s2)>1:
        for i in s2:
            data_str+=i
        s=data_str
        return s
    else:
        return '/'

#风险预测函数
def predict(s):
    seg_token=tokenizer.encode_plus(s,add_special_tokens=True,max_length=64,padding='max_length',return_attention_mask=True)
    seg_dataset={'input_ids':tf.constant([seg_token['input_ids']]),'attention_mask':tf.constant([seg_token['attention_mask']])\
                 ,'token_type_ids':tf.constant([seg_token['token_type_ids']])}
    pred_score=model_cls.predict(seg_dataset)
    cls_encode=pred_score[0][0]#取CLS的得分，用来表示整个句子
    pred_cls_ids=np.argmax(cls_encode)
    pred_cls=map_cls[int(pred_cls_ids)]
    return pred_cls_ids

###########################################################################################
#ner部分
###########################################################################################

#后缀
postfix_list=['控股股份公司','股份有限公司','有限责任公司','投资合伙企业','有限公司','合作联社','分公司','子公司','孙公司','总公司'
              ,'财政厅','财政局','公司','集团','分行','支行','协会','医院','银行']

ids_count = 0
right_count = 0
pred_right = 0
map_right = 0

df_company_name=data_1_com
c_name=df_company_name['公司名']
com_name=c_name.drop_duplicates(keep='first',inplace=False)
com_name=com_name.reset_index(drop=True)

#映射函数
def new_map(text):
    if len(set(text)) != len(text):
        counter_word = Counter(text).most_common(1)
        text = [counter_word[0][0]]
    if text == ['/']:
        return text
    company_name_map = []
    for i in text:
        ner_name = '/'
        for name in com_name:
            if i in name:
                ner_name = name
            #                 print(f'预测实体名为：{text}')
            #                 print(f'映射实体名为：{name}')
            #                 return name
            else:
                continue
        if ner_name == '/':
            temporary_name = []
            i_len = len(i)
            for name in com_name:
                word_count = 0
                name_len = len(name)
                if i_len <= name_len:
                    for j in i:
                        if j in name:
                            word_count += 1
                    if word_count / i_len > 0.9:
                        temporary_name.append(name)
                        continue
                if i_len >= name_len:
                    for j in name:
                        if j in i:
                            word_count += 1
                    if word_count / name_len > 0.9:
                        temporary_name.append(name)
                        continue
                else:
                    continue
            if temporary_name != []:
                name_index = []
                similarity = []
                for name_2 in temporary_name:
                    sim_name_i = get_equal_rate(i, name_2)
                    similarity.append(sim_name_i)
                ids_max = np.argmax(similarity)
                ner_name = temporary_name[int(ids_max)]
        #                     name_2_len=len(name_2)
        #                     count_2=0
        #                     if i_len<=name_2_len:
        #                         for k in i:
        #                             if k in name_2:
        #                                 count_2+=1
        #                         name_index.append(count_2)
        #                     if i_len>name_2_len:
        #                         for k in name_2:
        #                             if k in i:
        #                                 count_2+=1
        #                         name_index.append(count_2)
        #                 max_ids=np.argmax(name_index)
        #                 ner_name=temporary_name[max_ids]

        if ner_name == '/':
            i_2 = i[2:]
            for name in com_name:
                if i_2 in name:
                    ner_name = name
                else:
                    continue
        if ner_name == '/':
            i_3 = i[3:]
            for name in com_name:
                if i_3 in name:
                    ner_name = name
                else:
                    continue
        if ner_name == '/':
            i_4 = i[:-2]
            for name in com_name:
                if i_4 in name:
                    ner_name = name
                else:
                    continue
        if ner_name == '/':
            i_5 = i[:-3]
            for name in com_name:
                if i_5 in name:
                    ner_name = name
                else:
                    continue
        company_name_map.append(ner_name)
    ner_count = 0
    ids = 0
    ner_ids = []
    for j in company_name_map:
        ids += 1
        if j != '/':
            ner_count += 1
            ner_ids.append(ids - 1)
    print(f'预测实体名为：{text}')
    print(f'映射实体名为：{company_name_map}')
    print('')
    if ner_count == 0:
        return '/'
    if ner_count == 1:
        return company_name_map[ner_ids[0]]
    else:
        similarity_2 = []
        for i, j in zip(company_name_map, text):
            sim_name = get_equal_rate(i, j)
            similarity_2.append(sim_name)
        ids_max_sim = np.argmax(similarity_2)
        name_sim = company_name_map[int(ids_max_sim)]
        return name_sim

config_path = 'D:\\BERT预训练模型\\Bert_base_chinese\\bert_config.json'
checkpoint_path = 'D:\\BERT预训练模型\\Bert_base_chinese\\bert_model.ckpt'
dict_path = 'D:\\BERT预训练模型\\Bert_base_chinese\\vocab.txt'
tokenizer_bert4keras = Tokenizer(dict_path, do_lower_case=True)
categories=['LOC', 'ORG', 'PER']
categories = list(sorted(categories))
categories=set(categories)

maxlen = 64
epochs = 10
batch_size = 32
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必class data_generator(DataGenerator):
model_bert4keras_crf = build_transformer_model(
    config_path,
    checkpoint_path,
#     return_keras_model=False
)
# for i in model.layers:
#     print(i.name)
output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers-1)#Transformer-2-FeedForward-Norm
output = model_bert4keras_crf.get_layer(output_layer).output
output = Dense(7)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model_bert4keras_crf =Model(model_bert4keras_crf.input, output)
model_bert4keras_crf.summary()


model_bert4keras_crf.compile(loss=CRF.sparse_loss,
              optimizer=Adam(learing_rate),
              metrics=[CRF.sparse_accuracy])

model_bert4keras_crf.load_weights('./model_and_trainingcode/bert4keras_ner.h5')

class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer_bert4keras.tokenize(text, maxlen=128)
        mapping = tokenizer_bert4keras.rematch(text, tokens)
        token_ids = tokenizer_bert4keras.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model_bert4keras_crf.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

# 预测
# text = '“黑龙江黑化集团有限公司尿素厂发生爆燃事故'
# named_entity_recognize(text)
NER.trans = K.eval(CRF.trans)
rg=NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
def ner_text(text):
    a=rg.recognize(text)
    str_p=''
#     str_mid=''
    str_org_list=[]
    for i in a:
        str_o=''
        str_l=''
        if i[-1]=='ORG':
            str_o+=text[i[0]:i[1]+1]
            str_org_list.append(str_o)
            print(f'机构:{str_o}')
            str_o=''
        if i[-1]=='LOC':
            str_l+=text[i[0]:i[1]+1]
#             if str_l[-1]=='市' or '省':
#                 str_mid+=str_l
#                 str_p=str_mid
            print(f'位置：{str_l}')
#         str_org_list.append(str_p)
#         str_mid=str_p
#         str_p=''
    print('')
    return str_org_list

def del_none(text):
    if text==[]:
        return ['/']
    else:
        return text

#清理html
def clean_html(text):
    pattern=re.compile(r'<[^>]+>',re.S)
    result=re.sub(pattern,'',text)
    result=''.join(result.split())
    result=result[:128]
    return result

def clean_char(text):
    text_2=[]
    for i in text:
        if i in postfix_list:
            text_2.append('/')
        else:
            text_2.append(i)
    return text_2

data_in=pd.read_excel('in.xlsx')
name_list=['NEWS_BASICINFO_SID','NEWS_TITLE','CONTENT']
data_2=data_in[name_list]
data_drop_nan=data_2.dropna(subset=['NEWS_TITLE'])
data_id=data_drop_nan['NEWS_BASICINFO_SID']
data_id_int=data_id.apply(int)

data_title=data_drop_nan['NEWS_TITLE']
data_content=data_drop_nan['CONTENT']

#调用ner的各个函数
def ner_and_map(data1,data2):
    data_deal=data1.apply(data_deal_ner)
    data_ner=data_deal.apply(ner_text)
    data_dropna=data_ner.apply(del_none)
    data_map=data_dropna.apply(new_map)
    data_index=data_map.index.tolist()
    count=0
    nan_ids_list=[]
    for i in data_index:
        if data_map[i] != ['/'] and data_map[i]!='/':
            count+=1
        else:
            nan_ids_list.append(i)
    data_of_content=data2[nan_ids_list]
    data_content_nonhtml=data_of_content.apply(clean_html)
    data_content_deal=data_content_nonhtml.apply(data_deal_ner)
    data_content_ner=data_content_deal.apply(ner_text)
    data_content_clear_char=data_content_ner.apply(clean_char)
    data_content_map=data_content_clear_char.apply(new_map)
    for j in data_index:
        if j in nan_ids_list:
            data_map[j]=data_content_map[j]
    return data_map

#调用分类函数
def risk_classification(data):
    data_cls_id=data.apply(predict)
    data_cls=[map_cls[i] for i in data_cls_id]
    return data_cls

#处理标签为‘无’的情况
def deal_wu(data1,data2):
    ids=data1.index.tolist()
    for i in ids:
        if data1[int(i)]=='无':
            data2[int(i)]='/'
    return data2

data_ner=ner_and_map(data_title,data_content)
data_cls=risk_classification(data_title)
df_res=pd.DataFrame({'id':data_id_int,'ner':data_ner,'cls':data_cls})
data_ner_dropwu=deal_wu(df_res['cls'],data_ner)

df_res=pd.DataFrame({'id':data_id_int,'ner':data_ner_dropwu,'cls':data_cls})
df_res.to_csv('result.csv',header=None,index=False,encoding='utf-8')