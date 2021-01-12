"""
@Time : 2020/12/31 20:27 PM
@Author : 猿小明
"""

import os

import numpy as np
import tensorflow as tf
from dien_model import get_model


def train():
    """
    通过train_index.csv，结构为uindex,nindex,label
    预处理数据，构建数据集 uvec+nvec, label
    :return:
    """

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
        train_news_vec = np.load('../din/train/train_news_vec.npy')
        dev_news_vec = np.load('../din/dev/dev_news_vec.npy')
        return tf.constant(train_news_vec, dtype=tf.float32), tf.constant(dev_news_vec, dtype=tf.float32)

    nvec_train, nvec_dev = load_vec()

    def embedding_train(x, y):
        n1 = tf.nn.embedding_lookup(nvec_train, x['n1'])
        n1 = tf.expand_dims(n1, axis=1)
        n2 = tf.nn.embedding_lookup(nvec_train, x['n2'])
        n2 = tf.expand_dims(n2, axis=1)
        n3 = tf.nn.embedding_lookup(nvec_train, x['n3'])
        n3 = tf.expand_dims(n3, axis=1)
        n4 = tf.nn.embedding_lookup(nvec_train, x['n4'])
        n4 = tf.expand_dims(n4, axis=1)
        n5 = tf.nn.embedding_lookup(nvec_train, x['n5'])
        n5 = tf.expand_dims(n5, axis=1)
        candidate = tf.nn.embedding_lookup(nvec_train, x['nindex'])
        candidate = tf.expand_dims(candidate, axis=1)
        x_news = tf.concat([n1, n2, n3, n4, n5, candidate], axis=1)
        return x_news, y

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

    train_data = load_dataset('../din/train/copy1_5train_index.csv')
    train_data = train_data.map(embedding_train)
    dev_data = load_dataset('../din/dev/dev_index.csv')
    dev_data = dev_data.map(embedding_dev)

    model = get_model()

    class_weight = {0: 0.3, 1: 0.7}

    checkpoint_save_path = 'model/dien.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------load the model-------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True)

    model.fit(train_data, epochs=2, class_weight=class_weight, validation_data=dev_data, callbacks=cp_callback)

    model.summary()


train()

# 当前最好val_auc=0.6380，除了gru，其它网络层全l2，class_weight={0:0.3, 1:0.7}
