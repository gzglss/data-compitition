#对抗训练，训练集标签为0，测试集标签为1，识别出其他类别的索引

import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
import random
from torch import nn
import torch
import pandas as pd
import transformers as tfs

pretrain_model_path_bert = '../model_pre/bert_base_chinese'
pretrain_model_path_bertwwm='../model_pre/bert_chinese_wwm'
bert_finetune_model_path = '../model/bert_model/finetuned_model.bin'
test_path='../data/data_preprocess/test_f_b.csv'
unlabeled_path='../data/data_preprocess/unlabeled_f_b.csv'


class ElkanotoPuClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def __str__(self):
        return 'Estimator: {}\np(s=1|y=1,x) ~= {}\nFitted: {}'.format(
            self.estimator,
            self.c,
            self.estimator_fitted,
        )

    def permutation_(self, data):
        np.random.permutation(data)
        return data
    
    def shuffle_data(self,x,y):
        xy=list(zip(x,y))
        random.seed(2021)
        random.shuffle(xy)
        x[:],y[:]=zip(*xy)
        return np.array(x),np.array(y)

    def fit(self, pos, neg):
        # 打乱 pos 数据集, 按比例划分 hold_out 部分和非 hold_out 部分
        pos = self.permutation_(pos)
        neg = self.permutation_(neg)

        all_data = np.concatenate([pos, neg], axis=0)
        all_data_label = np.concatenate([np.full(shape=pos.shape[0], fill_value=1, dtype=np.int),
                                             np.full(shape=neg.shape[0], fill_value=-1, dtype=np.int)])
        all_data,all_data_label=self.shuffle_data(all_data,all_data_label)
        self.estimator.fit(all_data, all_data_label)
        return self

    def predict_proba(self, X):
        probabilistic_predictions = self.estimator.predict_proba(X)
        probabilistic_predictions = probabilistic_predictions[:, 1]
        return probabilistic_predictions

    def predict(self, X, threshold):
        return np.array([
            1.0 if p > threshold else -1.0
            for p in self.predict_proba(X)
        ])
    
def rf_classification(labeled_fb_encode,test_encode_,msg):
    print("\nStart fitting...")
    estimator = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        bootstrap=True,
        n_jobs=-1,
    )
    classifier = ElkanotoPuClassifier(estimator)

    neg_label=np.zeros((len(labeled_fb_encode),))
    pos_label=np.ones((len(test_encode_),))
    X=np.concatenate([labeled_fb_encode,test_encode_],axis=0)
    y=np.concatenate([neg_label,pos_label],axis=0)
    
    X_pos = X[y == 1]
    print("len of X_positive: ", X_pos.shape)

    X_neg = X[y == 0]
    print("len of X_neg: ", X_neg.shape)
    classifier.fit(X_pos, X_neg)
    joblib.dump(classifier, '../model/binary_class/binary_classification_{}.bin'.format(msg))
    print("Fitting done!")
    return classifier


class MyBertEncoder(nn.Module):
    """Bert分类器模型"""
    def __init__(self, tokenizer_path, finetuned_bert_path):
        super(MyBertEncoder, self).__init__()
        model_class, tokenizer_class = tfs.BertModel, tfs.BertTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        self.bert = torch.load(finetuned_bert_path)

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,truncation=True,
                                                           max_length=512, padding=True)

        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        token_type_ids = torch.tensor(batch_tokenized['token_type_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()

        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        return bert_cls_hidden_state
    
def predict_two_class():
    labeled_fb_encode=np.load('../data/encode_data/labeled_fb.npy')
    test_fb_encode=np.load('../data/encode_data/test_fb.npy')    
    test_fb_encode1=test_fb_encode[:len(test_fb_encode)//2]
    test_fb_encode2=test_fb_encode[len(test_fb_encode)//2:]
    test_data=pd.read_csv(test_path)
    test_text=test_data['text'].apply(eval)#之前未加
    test_text1=test_text[:len(test_fb_encode)//2]
    test_text2=test_text[len(test_fb_encode)//2:]
    
    unlabeled=pd.read_csv(unlabeled_path)
    unlabeled_fb_sample=unlabeled.sample(frac=0.01)
    unlabeled_fb_sample.to_csv('../data/data_preprocess/unlabeled_fb_sample.csv',index=False,encoding='utf-8')
    unlabeled_fb_sample_text=unlabeled_fb_sample['text'].apply(eval)

    classifier1=rf_classification(labeled_fb_encode,test_fb_encode1,'fb1')#用来对第二部分测试集预测
    print('over')
    classifier2=rf_classification(labeled_fb_encode,test_fb_encode2,'fb2')
    print('over')
    
    c=[0,0]
    with torch.no_grad():
        bert_encoder = MyBertEncoder(pretrain_model_path_bert, finetune_model_path)
        bert_encoder.eval()    

        result2=[]
        for text in tqdm(test_text2):#对测试集第二部分预测
            test_encode=np.array(bert_encoder([text]).tolist())
            a=classifier1.predict_proba(test_encode)
            result2.append(a[0])

        result1=[]
        for text in tqdm(test_text1):
            test_encode=np.array(bert_encoder([text]).tolist())
            a=classifier2.predict_proba(test_encode)
            result1.append(a[0])

        
        unlabel_result2=[]
        for text in tqdm(unlabeled_fb_sample_text):
            unlabel_encode=np.array(bert_encoder([text]).tolist())
            a=classifier1.predict_proba(unlabel_encode)
            unlabel_result2.append(a[0])
        c[1]=np.mean(unlabel_result2)

        unlabel_result1=[]
        for text in tqdm(unlabeled_fb_sample_text):
            unlabel_encode=np.array(bert_encoder([text]).tolist())
            a=classifier2.predict_proba(unlabel_encode)
            unlabel_result1.append(a[0])
        c[0]=np.mean(unlabel_result1)
        
    c_med=[np.median(unlabel_result1),np.median(unlabel_result2)]
    unlabel_result=np.concatenate([unlabel_result1,unlabel_result2],axis=0)
    np.save('unlabel_pred_result.npy',unlabel_result)
    result=np.concatenate([result1,result2],axis=0)
    np.save('test_pred_result.npy',result)
    
    return result,result1,result2,c_med

def get_qita_ids():
    result,result1,result2,c_med=predict_two_class()
    
    threshold1=0.45*(1-c_med[0])
    threshold2=0.45*(1-c_med[1])

    qita_ids_rf=[]
    for i in range(len(result1)):
        if result1[i]>threshold1:
            qita_ids_rf.append(i)

    for i in range(len(result2)):
        if result2[i]>threshold2:
            qita_ids_rf.append(len(result1)+i)

    np.save('../data/encode_data/qita_ids.npy',qita_ids_rf)