"""
@Time : 2020/11/12 22:59 AM
@Author : 猿小明
"""
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics import roc_auc_score


def build_predict_result():

    def load_dataset(csv_path, shuffle=True):
        return tf.data.experimental.make_csv_dataset(
            csv_path,
            batch_size=256,  # 设置batch size
            shuffle=shuffle,  # 设置随机排列
            label_name='label',  # 设置标签所在的列名
            na_value='?',
            num_epochs=1,
            ignore_errors=True)

    def load_vec():
        news_vec = np.load('../data_demo/train/train_news_vec.npy')
        user_vec = np.load('../data_demo/train/train_user_vec.npy')
        return tf.constant(news_vec, dtype=tf.float32), tf.constant(user_vec, dtype=tf.float32)

    nvec, uvec = load_vec()

    def embedding(x, y):
        return tf.concat([tf.nn.embedding_lookup(uvec, x['uindex']), tf.nn.embedding_lookup(nvec, x['nindex'])],
                         axis=1), y

    dev_data = load_dataset('../data_demo/dev/dev_index.csv')
    dev_data = dev_data.map(embedding)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'])

    checkpoint_save_path = 'model/dc.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------load the model-------------')
        model.load_weights(checkpoint_save_path)

    # result = model.evaluate(dev_data)
    result = model.predict(dev_data)

    dev_df = pd.read_csv('../data_demo/dev/dev_index.csv')
    dev_df['predict'] = result
    dev_df.to_csv('../data_demo/dev/dev_index_predict.csv', index=False)


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
        k: top k

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def evaluate():
    """
    计算ndcg
    :return:
    """
    top = [5, 10]
    df = pd.read_csv('../data_demo/dev/dev_index_predict.csv')
    y_true = df.groupby(by=['uindex'])['label'].apply(list)
    y_score = df.groupby(by=['uindex'])['predict'].apply(list)

    for k in top:
        ndcg_temp = np.mean([
            ndcg_score(each_labels, each_preds, k) for each_labels, each_preds in zip(y_true, y_score)
        ])
        print('ndcg@' + str(k) + ': ' + str(ndcg_temp))

    group_auc = np.mean([
        roc_auc_score(each_labels, each_preds) for each_labels, each_preds in zip(y_true, y_score)
    ])
    print('group auc: ' + str(group_auc))


evaluate()
