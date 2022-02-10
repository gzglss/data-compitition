import json
import os
import torch
import numpy as np
import transformers as tfs
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
from torch import nn
from bert_transform_classification import BertClassificationModel
import bertwwm_transform_classification

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(0)
softmax = nn.Softmax(dim=1)
# Bert预训练模型
pretrain_model_path_bert = '../model_pre/bert_base_chinese/'
pretrain_model_path_bertwwm = '../model_pre/bert_chinese_wwm/'
pretrain_model_path_macbert = 'hfl/chinese-macbert-base'
bert_model_path='../model/bert_model/bert_model_epoch2.pkl'
bertwwm_model_path='../model/bertwwm/bertwwm_model_epoch2.pkl'
macbert_model_path='../model/macbert/macbert_model_epoch2.pkl'
submission_path='../data/result/submission.csv'
TEST_FILE_PATH='../data/data_preprocess/test_f_b.csv'
INDEX = ['人物专栏', '作品分析', '情感解读', '推荐文', '攻略文', '治愈系文章', '深度事件', '物品评测', '科普知识文', '行业解读']

def predict_(x, index, bert_classifier_model,bertwwm_classifier_model,macbert_classifier_model):
    x1=x[0]+'。'+x[1]

    output_bert = bert_classifier_model([x]).cuda()       
    output_bertwwm = bert_classifier_model([x1]).cuda()
    output_macbert = macbert_classifier_model([x1]).cuda()  
    
    output_bert=output_bert.view((1,-1)).cuda()
    output_bertwwm=output_bertwwm.view((1,-1)).cuda()
    output_macbert=output_macbert.view((1,-1)).cuda()
    
    bert_predicted_proba = softmax(output_bert).cuda().tolist()[0]
    bertwwm_predicted_proba = softmax(output_bertwwm).cuda().tolist()[0]
    macbert_predicted_proba = softmax(output_macbert).cuda().tolist()[0]
    
    predicted_proba=(np.array(bert_predicted_proba)+np.array(bertwwm_predicted_proba)+np.array(macbert_predicted_proba))/3
    predicted_index = np.argmax(predicted_proba)
    predicted_label = index[predicted_index]

    # 预测类别的预测概率
    proba = predicted_proba[predicted_index]

    return [predicted_label, round(proba, 2)]


def predict_test():
    qita_ids=np.load('../data/encode_data/qita_ids.npy')
    test=pd.read_csv(TEST_FILE_PATH)
    test_drop=test.drop(labels=qita_ids)
    test_text=test['text'].apply(eval)
    test_text=test_text.drop(labels=qita_ids)

    print("Start evluation...")
    print("Load bert_classifier model path: ", bert_model_path)

    bert_classifier_model = torch.load(bert_model_path)
    bert_classifier_model = bert_classifier_model.cuda()
    bert_classifier_model.eval()

    bertwwm_classifier_model = torch.load(bertwwm_model_path)
    bertwwm_classifier_model = bertwwm_classifier_model.cuda()
    bertwwm_classifier_model.eval()

    macbert_classifier_model = torch.load(macbert_model_path)
    macbert_classifier_model = macbert_classifier_model.cuda()
    macbert_classifier_model.eval()

    with torch.no_grad():
        test_drop[["predicted_label", "proba"]] = test_text.progress_apply(
            lambda x: pd.Series(predict_(x, INDEX, bert_classifier_model,bertwwm_classifier_model,macbert_classifier_model)))

        # 提取id, predicted_label两列信息,并重命名列名, 最后输出到文件
        csv_data = test[['id']]
        doctype=[]
        for i in range(len(csv_data)):
            if i in qita_ids:
                doctype.append('其他')
            else:
                doctype.append(test_drop.loc[i]['predicted_label'])
        csv_data['predict_doctype']=doctype
        print("\n\n===================   The distribution of predictions   ===================\n")
        print(csv_data["predict_doctype"].value_counts())
        print("\n\n")
        csv_data.to_csv(submission_path, index=False,encoding='utf-8')
    print('done')