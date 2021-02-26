"""
@Time : 2021/2/23 1:39 PM 
@Author : Xiaoming
lgb排序模型，将召回结果进行排序
"""

import os
import random

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from utils_log import Logger
from utils_evaluate import evaluate

seed = 2020
random.seed(seed)

# 初始化日志
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('rank lgb')


def train_model(df_train_set):
    """
    :param df_train_set: 训练集
    :return:
    """

    ycol = 'label'
    feature_names = list(
        filter(lambda x: x not in [ycol, 'created_at_ts', 'click_timestamp'], df_train_set.columns))
    feature_names.sort()

    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.05,
                               n_estimators=10000,
                               subsample=0.8,
                               feature_fraction=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               importance_type='gain',
                               metric=None)

    oof = []
    prediction = df_train_set[['user_id', 'article_id']]
    prediction['pred'] = 0
    df_importance_list = []

    # 训练模型
    kfold = GroupKFold(n_splits=2)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train_set[feature_names], df_train_set[ycol], df_train_set['user_id'])):
        X_train = df_train_set.iloc[trn_idx][feature_names]
        Y_train = df_train_set.iloc[trn_idx][ycol]

        X_val = df_train_set.iloc[val_idx][feature_names]
        Y_val = df_train_set.iloc[val_idx][ycol]

        log.debug(
            f'\nFold_{fold_id + 1} Training ================================\n'
        )

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=100,
                              eval_metric='auc',
                              early_stopping_rounds=100)

        pred_val = lgb_model.predict_proba(X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        df_oof = df_train_set.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict_proba(
            df_train_set[feature_names], num_iteration=lgb_model.best_iteration_)[:, 1]
        prediction['pred'] += pred_test / 5

        df_importance = pd.DataFrame({
            'feature_name':
            feature_names,
            'importance':
            lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        joblib.dump(model, f'model/lgb{fold_id}.pkl')

    # 特征重要性
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')

    # 生成线下
    df_oof = pd.concat(oof)
    df_oof.sort_values(['user_id', 'pred'],
                       inplace=True,
                       ascending=[True, False])
    log.debug(f'df_oof.head: {df_oof.head()}')

    # 计算相关指标
    total = df_train_set.user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )


if __name__ == '__main__':
    # 用召回数据直接作为训练数据集（感觉怪怪的）
    df_train_feature = pd.read_csv('data/rank_train.csv')

    for f in df_train_feature.select_dtypes('object').columns:
        lbl = LabelEncoder()
        df_train_feature[f] = lbl.fit_transform(df_train_feature[f].astype(str))

    train_model(df_train_feature)
