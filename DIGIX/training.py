from bert_transform_classification import BertClassificationModel
import bertwwm_transform_classification
import macbert_transform_classification
from Train_Bert_transform import train_bert
from Train_Bert_transform_wwm import train_bertwwm
from Train_Bert_transform_mac import train_macbert

if __name__ == "__main__":
    train_bert()#训练bert
    train_bertwwm()#训练bertwwm
    train_macbert()#训练macbert