import pandas as pd
import numpy as np


input_path='../data/raw_data/doc_quality_data_train.json'
output_path1='../data/raw_data/labeled_train_data.json'
output_path2='../data/raw_data/unlabeled_train_data.json'
def data_doctype_split():
    print('开始提取有标签数据...')
    df_data=pd.read_json(input_path,orient='records',lines=True)
    df_data_labeled=df_data[df_data['doctype']!='']
    df_data_labeled=df_data_labeled.sample(frac=1.0)
    df_data_labeled.to_json(output_path1,orient='records',lines=True,force_ascii=False)
    print('有标签数据提取完毕')
    print('')
    print('开始提取无标签数据...')
    df_data_unlabel=df_data[df_data['doctype']=='']
    df_data_unlabel=df_data_unlabel.sample(frac=1.0)
    df_data_unlabel.to_json(output_path2,orient='records',lines=True,force_ascii=False)
    print('无标签数据提取完毕')