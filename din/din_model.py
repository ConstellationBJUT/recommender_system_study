"""
@Time : 2020/12/3 9:45 AM 
@Author : 猿小明
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras import Model


class Attention(Model):
    """
    cross网络
    """
    def __init__(self, hiddens):
        """
        :param hiddens: DNN隐藏层各层节点数
        """
        super(Attention, self).__init__()
        self.deep_model = tf.keras.models.Sequential()
        self.deep_model.add(tf.keras.layers.BatchNormalization())
        for hidden in hiddens:
            self.deep_model.add(Dense(hidden, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))
        self.deep_model.add(Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2()))

    def call(self, inputs, **kwargs):
        user_click = inputs[:, 0:-1, :]
        news = inputs[:, -1, :]
        result = tf.zeros_like(news)
        for i in range(user_click.shape[1]):
            sub = news - user_click[:, i, :]
            input_emb = tf.concat([user_click[:, i, :], sub, news], axis=-1)
            ai = self.deep_model(input_emb)
            result += ai * user_click[:, i, :]
        return result


class DIN(Model):
    """
    DIN网络
    """
    def __init__(self, hiddens):
        """
        :param hiddens: 每个隐藏层神经元数
        """
        super(DIN, self).__init__()
        self.attention = Attention([128, 64])
        self.deep_model = tf.keras.models.Sequential()
        self.deep_model.add(tf.keras.layers.BatchNormalization())
        for hidden in hiddens:
            self.deep_model.add(Dense(hidden, activation=None, kernel_regularizer=tf.keras.regularizers.l2()))
            self.deep_model.add(ReLU(negative_slope=0.0003))
        self.f = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs, training=None, mask=None):
        user = self.attention(inputs)
        news = inputs[:, -1, :]
        x = tf.concat([user, news], axis=-1)
        x = self.deep_model(x)
        y = self.f(x)
        return y


def get_model():
    model = DIN([128, 64])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'])
    return model

