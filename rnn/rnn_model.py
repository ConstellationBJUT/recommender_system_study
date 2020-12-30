"""
@Time : 2020/12/28 3:54 PM 
@Author : 猿小明
需要
0、data内生成数据
1、din/din_build_data_set.py生成数据
2、执行rnn_model，调节86行LSTM单元

# GRU(16) 无class_weight 0.6215
# GRU(32) 无class_weight 0.6257
# GRU(64) 无class_weight 0.63
# GRU(64) class_weight={0:0.3, 1:0.7} 0.64
# GRU(64, kr=l2, rr=l2) class_weight={0:0.3, 1:0.7} 0.60 train和dev接近

# LSTM(64) class_weight={0:0.28, 1:0.72} 0.6482 不加l2约束
# LSTM(64, kr=l2, rr=l2) class_weight={0:0.3, 1:0.7} 0.6061
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense, GRU, LSTM


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

    model = tf.keras.Sequential([
        LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(), recurrent_regularizer=tf.keras.regularizers.l2()),
        Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2())
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'])

    class_weight = {0: 0.3, 1: 0.7}

    checkpoint_save_path = 'model/dien.ckpt'
    if os.path.exists(checkpoint_save_path + '.index'):
        print('--------------load the model-------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True)

    model.fit(train_data, epochs=5, class_weight=class_weight, validation_data=dev_data, callbacks=cp_callback)

    model.summary()


train()



