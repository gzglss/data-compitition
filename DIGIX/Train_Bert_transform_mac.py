import json
import os
import torch
import transformers as tfs
import random
import pandas as pd
from torch import nn
from torch import optim
from tqdm import tqdm
from logger import Progbar
import Config
from sklearn.preprocessing import LabelEncoder
from macbert_transform_classification import BertClassificationModel

# Bert预训练模型
INDEX = ['人物专栏', '作品分析', '情感解读', '推荐文', '攻略文', '治愈系文章', '深度事件', '物品评测', '科普知识文', '行业解读']
FINETUNED_BERT_ENCODER_PATH = '../model/macbert/finetuned_macbert.bin'
POSITIVE_TRAIN_FILE_PATH='../data/data_preprocess/labeled_f_b.csv'
UNLABELED_TRAIN_FILE_PATH='../data/data_preprocess/unlabeled_f_b.csv'
PRETRAINED_BERT_ENCODER_PATH='hfl/chinese-macbert-base'
BERT_MODEL_SAVE_PATH = '../model/macbert/'
BATCH_SIZE = 16
EPOCH = 3

# 生成数据集对应的标签集以及样本总数
def build_label_set_and_sample_num(input_path, output_path):
    label_set = set()
    sample_num = 0
    
    with open(input_path, 'r', encoding="utf-8") as input_file:
        for line in tqdm(input_file):
            json_data = json.loads(line)
            label_set.add(json_data["label"])
            sample_num += 1
            
    with open(output_path, "w", encoding="UTF-8") as output_file:
        record = {"label_list": sorted(list(label_set)), "total_num": sample_num}
        return record["label_list"], record["total_num"]


# 获取一个epoch需要的batch数
def get_steps_per_epoch(line_count, batch_size):
    return line_count // batch_size if line_count % batch_size == 0 else line_count // batch_size + 1


def get_text_and_label_index_iterator():
    labeled_data=pd.read_csv(POSITIVE_TRAIN_FILE_PATH)
    labeled_data=labeled_data.sample(frac=1.0)
    labeled_text=labeled_data['text'].apply(eval).values
    labeled_label=LabelEncoder().fit_transform(labeled_data['doctype'])
    for i in range(len(labeled_label)):
        yield labeled_text[i],labeled_label[i]


# 迭代器: 生成一个batch的数据
def get_bert_iterator_batch(data_path, batch_size=32):
    keras_bert_iter = get_text_and_label_index_iterator()
    continue_iterator = True
    while continue_iterator:
        data_list = []
        for _ in range(batch_size):
            try:
                data = next(keras_bert_iter)
                data_list.append(data)
            except StopIteration:
                continue_iterator = False
        random.shuffle(data_list)

        text_list = []
        label_list = []

        for data in data_list:
            text, label = data
            text_list.append(text)
            label_list.append(label)

        yield text_list, label_list

    return False


def train_macbert():
    
    labeled_data=pd.read_csv(POSITIVE_TRAIN_FILE_PATH)
    labeled_data=labeled_data.sample(frac=1.0)
    labeled_text=labeled_data['text'].apply(eval).values
    labeled_label=LabelEncoder().fit_transform(labeled_data['doctype'])
    
    labels_set, total_num=INDEX,len(labeled_data)
    torch.cuda.set_device(0)

    print("Start training model...")
    # train the model
    steps = get_steps_per_epoch(total_num, BATCH_SIZE)

    bert_classifier_model = BertClassificationModel(len(labels_set))
    bert_classifier_model = bert_classifier_model.cuda()

    # 不同子网络设定不同的学习率
    Bert_model_param = []
    Bert_downstream_param = []
    number = 0
    for items, _ in bert_classifier_model.named_parameters():
        if "bert" in items:
            Bert_model_param.append(_)
        else:
            Bert_downstream_param.append(_)
        number += _.numel()
    param_groups = [{"params": Bert_model_param, "lr": 1e-5},
                    {"params": Bert_downstream_param, "lr": 1e-4}]
    optimizer = optim.Adam(param_groups, eps=1e-7, weight_decay=0.001)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.6)
    criterion = nn.CrossEntropyLoss()
    bert_classifier_model.train()
    progbar = Progbar(target=steps)

    for epoch in range(EPOCH):
        model_save_path = os.path.join(BERT_MODEL_SAVE_PATH, "macbert_model_epoch{}.pkl".format(epoch))

        dataset_iterator = get_bert_iterator_batch(POSITIVE_TRAIN_FILE_PATH, BATCH_SIZE)

        for i, iteration in enumerate(dataset_iterator):
            # 清空梯度
            bert_classifier_model.zero_grad()
            text = iteration[0]
            labels = torch.tensor(iteration[1]).cuda()
            optimizer.zero_grad()
            output = bert_classifier_model(text)
            loss = criterion(output, labels).cuda()
            loss.backward()

            # 更新模型参数
            optimizer.step()
            # 学习率优化器计数
            StepLR.step()
            progbar.update(i + 1, None, None, [("train loss", loss.item()), ("bert_lr", optimizer.state_dict()["param_groups"][0]["lr"]), ("fc_lr", optimizer.state_dict()["param_groups"][1]["lr"])])

            if i == steps - 1:
                break

        # 保存完整的 BERT 分类器模型
        torch.save(bert_classifier_model, model_save_path)
        # 单独保存经 fune tune 的 BertEncoder模型
        torch.save(bert_classifier_model.bert, FINETUNED_BERT_ENCODER_PATH)
        print("epoch {} is over!\n".format(epoch))

    print("\nTraining is over!\n")
