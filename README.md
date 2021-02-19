# recommender_system_study
Deep Learning Recommender System Reading and Study Notes  
深度学习推荐系统读书和学习笔记。

Implementing algorithm（deep crossing, deepFM e.t.） in book and giving examples（text recommend）.  
实现书中部分算法（如deep crossing, deepFM等）并且给出应用样例（文本推荐）。

TensorFlow1.14 and Bert are used to create text vectors（embedding）.  
TensorFlow1.14和Bert预训练模型应用于文本向量化（embedding）。

TensorFlow2.3 is used to build deep learning net.  
TensorFlow2.3用于搭建深度学习网络。

Need to set up 2 python3.7 virtual environments，first include TensorFlow1.14 and second include TensorFlow2.0.  
需要安装2个python3.7虚拟环境，一个包含tf1.14，一个包含tf2.0.

tf1.14 run bert. tf2.0 run deep net.  
tf1.14用于跑bert程序。tf2.0用于跑深度网络。

Data source. 数据源  
https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_train.zip
https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_dev.zip

data description. 数据说明  
https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md

MIND paper is unrelate to the study notes. MIND paper url  
https://msnews.github.io/  

study notes blog records updating log and experience. 学习笔记博客用户记录更细和学习体会  
https://blog.csdn.net/bjjoy2009/category_10615596.html  

down load tiny bert pre-trained model  
https://github.com/google-research/bert

2020-11-28 update
deep crossing: concat(user_vec, news_vec)->Resnet  
user_vec.shape=(128,), news_vec.shape=(128,)  
train_set.shape=(n, 256), train_label.shape=(n, 1)

2020-12-03 update deep&cross 

2020-12-23 update DIN

2020-12-30 update RNN

2020-01-12 update DIEN 

2020-02-18 add a news recommend system demo