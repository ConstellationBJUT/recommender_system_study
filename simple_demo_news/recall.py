"""
@Time : 2021/2/19 4:33 PM 
@Author : Xiaoming
将多个召回合并
1、提升评估指标mrr
2、itemcf没有召回数据的新用户，可以用hot去补全
3、可以加入多种召回方法，进行召回（这里没有做）
"""

import os
import warnings
from collections import defaultdict
from itertools import permutations

import pandas as pd
from tqdm import tqdm

from utils_log import Logger
from utils_evaluate import evaluate

warnings.filterwarnings('ignore')


# 初始化日志
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('多路召回合并: ')


def mms(df):
    """
    将score进行最大最小值归一化，将不同召回得分归一化到相同范围
    :param df:
    :return:
    """
    user_score_max = {}
    user_score_min = {}

    # 获取用户下的相似度的最大值和最小值
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans


def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    df_train = pd.read_csv('data/my_train_set.csv')
    df_test = pd.read_csv('data/my_test_set.csv')

    recall_path = 'result'

    # 召回方式
    recall_methods = ['itemcf', 'hot']
    # 每种召回得分权重，这里hot很低，为了补全没有召回数据的用户，较小影响itemcf召回
    weights = {'itemcf': 1, 'hot': 0.1}
    recall_list = []
    # recall_dict = {}
    for recall_method in recall_methods:
        recall_result = pd.read_csv(f'{recall_path}/recall_{recall_method}.csv')
        weight = weights[recall_method]

        recall_result['sim_score'] = mms(recall_result)
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        # recall_dict[recall_method] = recall_result

    # 求相似度
    # for recall_method1, recall_method2 in permutations(recall_methods, 2):
    #     score = recall_result_sim(recall_dict[recall_method1],
    #                               recall_dict[recall_method2])
    #     log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')

    # 合并召回结果
    recall_final = pd.concat(recall_list, sort=False)
    # 一个user多个召回有同一个item，将分数相加
    recall_score = recall_final[['user_id', 'article_id', 'sim_score']].groupby(['user_id', 'article_id'
                                 ])['sim_score'].sum().reset_index()
    # 清理冗余数据
    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
    # 将label拼接回来
    recall_final = recall_final.merge(recall_score, how='left')
    # 将用户的得分按照从大到小排序
    recall_final.sort_values(['user_id', 'sim_score'],
                             inplace=True,
                             ascending=[True, False])
    # 只取每个用户前50个（对结果影响不大）
    recall_final = recall_final.groupby('user_id').head(50)
    log.debug(f'recall_final.shape: {recall_final.shape}')

    # 计算相关指标
    total = df_test.user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        recall_final[recall_final['label'].notnull()], total)

    log.debug(
        f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )

    df = recall_final['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")

    log.debug(
        f"标签分布: {recall_final[recall_final['label'].notnull()]['label'].value_counts()}"
    )

    recall_final.to_csv('result/recall.csv')
