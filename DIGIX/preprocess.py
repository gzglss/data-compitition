from data_preprocess_divide_train import data_doctype_split
from data_preprocess_cleardata import clear_data
from data_preprocess_split_data import extract_front_behind_data

if __name__ == "__main__":
    data_doctype_split()#原始数据集标签划分
    clear_data()#清洗数据集中的html，空格等
    extract_front_behind_data()#提取body中的前256与后256个字作为输入