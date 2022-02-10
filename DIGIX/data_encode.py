import pandas as pd
import numpy as np
import random
import torch
from torch import nn
import transformers as tfs
from tqdm import tqdm

torch.cuda.set_device(0)

labeled_fb_path='../data/data_preprocess/labeled_f_b.csv'
test_fb_path='../data/data_preprocess/test_f_b.csv'

pretrain_model_path_bert = '../model_pre/bert_base_chinese/'
# pretrain_model_path_bertwwm = '../pretrain_model/bertwwm/'
finetune_model_path='../model/bert_model/finetuned_bert.bin'
batch_size=16

def get_text_and_label_index_iterator(input_path):
    data=pd.read_csv(input_path)
    data_text=data['text'].apply(eval).values
    for i in range(len(data_text)):
        yield data_text[i]


# 迭代器: 生成一个batch的数据
def get_bert_iterator_batch(data_path, batch_size=32):
    keras_bert_iter = get_text_and_label_index_iterator(data_path)
    continue_iterator = True
    while True:
        data_list = []
        for _ in range(batch_size):
            try:
                data = next(keras_bert_iter)
                data_list.append(data)
            except StopIteration:
                break
        text_list = []
        if continue_iterator:
            if len(data_list)<batch_size:
                continue_iterator = False
            for data in data_list:
                text_list.append(data)

            yield text_list
        else:
            return StopIteration
        
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
    
def encode():
    labeled_fb=pd.read_csv(labeled_fb_path)
    test_fb=pd.read_csv(test_fb_path)
    labeled_fb['text']=labeled_fb['text'].apply(eval)
    test_fb['text']=test_fb['text'].apply(eval)
    with torch.no_grad():
    
        bert_encoder = MyBertEncoder(pretrain_model_path_bert, finetune_model_path)
        bert_encoder.eval()
        print('finetune_model_path:',finetune_model_path)
        labeled_fb_iter=get_bert_iterator_batch(labeled_fb_path,16)
        test_fb_iter=get_bert_iterator_batch(test_fb_path,16)

        labeled_fb_encode=None

        for label in tqdm(labeled_fb_iter):
            encoded_label = np.array(bert_encoder(label).tolist())
            if labeled_fb_encode is None:
                labeled_fb_encode=encoded_label
            else:
                labeled_fb_encode=np.concatenate([labeled_fb_encode,encoded_label],axis=0)

        for test_ in tqdm(test_fb_iter):
            encode_test=np.array(bert_encoder(test_).tolist())
            if test_fb_encode is None:
                test_fb_encode=encode_test
            else:
                test_fb_encode=np.concatenate([test_fb_encode,encode_test],axis=0)

        print('测试集编码完毕...')
    
    np.save('../data/encode_data/labeled_fb.npy',labeled_fb_encode)
    np.save('../data/encode_data/test_fb.npy',test_fb_encode)
    print('训练集编码完毕...')