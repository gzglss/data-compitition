from data_preprocess_divide_train import data_doctype_split
from data_preprocess_cleardata import clear_data
from data_preprocess_split_data import extract_front_behind_data
from bert_transform_classification import BertClassificationModel
import bertwwm_transform_classification
import macbert_transform_classification
from Train_Bert_transform import train_bert
from Train_Bert_transform_wwm import train_bertwwm
from Train_Bert_transform_mac import train_macbert
from data_encode import encode
from binary_classificator import get_qita_ids
from test_predict import predict_test

if __name__ == "__main__":
    data_doctype_split()#原始数据集标签划分
    clear_data()#清洗数据集中的html，空格等
    extract_front_behind_data()#提取body中的前256与后256个字作为输入
    train_bert()#训练bert
    train_bertwwm()#训练bertwwm
    train_macbert()#训练macbert
    encode()#文本编码
    get_qita_ids()#对抗训练得到’其他‘类的样本索引
    predict_test()#预测剩余样本的类别