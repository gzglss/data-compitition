from data_encode import encode
from binary_classificator import get_qita_ids
from test_predict import predict_test

if __name__ == "__main__":
    encode()#文本编码
    get_qita_ids()#对抗训练得到’其他‘类的样本索引
    predict_test()#预测剩余样本的类别