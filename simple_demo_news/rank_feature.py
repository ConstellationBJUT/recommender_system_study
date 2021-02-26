"""
@Time : 2021/2/22 10:24 AM 
@Author : Xiaoming
构建排序模型训练数据集
"""
import os

import pandas as pd

from utils_log import Logger
from utils_evaluate import evaluate


# 初始化日志
# log_file = 'my_log.log'
# os.makedirs('log', exist_ok=True)
# log = Logger(f'log/{log_file}').logger
# log.info('rank 特征提取')


if __name__ == '__main__':
    # 召回结果
    df_recall = pd.read_csv('result/recall.csv')
    df_test = pd.read_csv('data/my_test_set.csv')
    df_article = pd.read_csv('data/articles.csv')

    # 添加文章单词数量，文章类别，文章创建时间
    df_recall = df_recall.merge(df_article, how='left')
    # 添加每个用户最后一次点击时间
    df_recall = df_recall.merge(df_test[['user_id', 'click_timestamp']], how='left')
    # 添加用户最后一次点击时间与文章创建时间的差值
    df_recall['last_click_create_diff'] = df_recall['click_timestamp'] - df_recall['created_at_ts']

    df_recall.to_csv('data/rank_train.csv', index=False)
