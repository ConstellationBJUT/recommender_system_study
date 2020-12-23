"""
@Time : 2020/11/12 9:59 AM 
@Author : 猿小明
"""
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from din_model import get_model


def build_predict_result():

    def load_dataset(csv_path, shuffle=False):
        return tf.data.experimental.make_csv_dataset(
            csv_path,
            batch_size=256,  # 设置batch size
            shuffle=shuffle,  # 设置随机排列
            label_name='label',  # 设置标签所在的列名
            na_value='?',
            num_epochs=1,
            ignore_errors=True)

    def load_vec():
        news_vec = np.load('dev/dev_news_vec.npy')
        return tf.constant(news_vec, dtype=tf.float32)

    nvec_dev = load_vec()

    def embedding_dev(x, y):
        n1 = tf.nn.embedding_lookup(nvec_dev, x['n1'])
        n1 = tf.expand_dims(n1, axis=1)
        n2 = tf.nn.embedding_lookup(nvec_dev, x['n2'])
        n2 = tf.expand_dims(n2, axis=1)
        n3 = tf.nn.embedding_lookup(nvec_dev, x['n3'])
        n3 = tf.expand_dims(n3, axis=1)
        n4 = tf.nn.embedding_lookup(nvec_dev, x['n4'])
        n4 = tf.expand_dims(n4, axis=1)
        n5 = tf.nn.embedding_lookup(nvec_dev, x['n5'])
        n5 = tf.expand_dims(n5, axis=1)
        candidate = tf.nn.embedding_lookup(nvec_dev, x['nindex'])
        candidate = tf.expand_dims(candidate, axis=1)
        x_news = tf.concat([n1, n2, n3, n4, n5, candidate], axis=1)
        return x_news, y

    dev_data = load_dataset('dev/dev_index.csv')
    dev_data = dev_data.map(embedding_dev)

    model = get_model()

    checkpoint_save_path = 'model/din.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------load the model-------------')
        model.load_weights(checkpoint_save_path)

    result = model.predict(dev_data)

    # model.evaluate(dev_data)
    dev_df = pd.read_csv('dev/dev_index.csv')
    dev_df['predict'] = result
    dev_df.to_csv('dev_index_predict.csv', index=False)


build_predict_result()


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
        k
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
    df = pd.read_csv('dev_index_predict.csv')
    y_true = df.groupby(by=['uid'])['label'].apply(list)
    y_score = df.groupby(by=['uid'])['predict'].apply(list)

    for k in top:
        ndcg_temp = np.mean([
            ndcg_score(each_labels, each_preds, k) for each_labels, each_preds in zip(y_true, y_score)
        ])
        print('ndcg@' + str(k) + ': ' + str(ndcg_temp))

    group_auc = np.mean([
        roc_auc_score(each_labels, each_preds) for each_labels, each_preds in zip(y_true, y_score)
    ])
    print('group auc: ' + str(group_auc))

    auc = roc_auc_score(df['label'], df['predict'])
    print('auc: ' + str(auc))


evaluate()
