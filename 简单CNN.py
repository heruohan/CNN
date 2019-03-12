# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 00:12:09 2018

@author: hecongcong
"""

##########tensorflow 实现简单的卷积网络
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist=input_data.read_data_sets(r'E:\tensorflow\MNIST',\
                                one_hot=True)
sess=tf.InteractiveSession()

######定义初始化函数
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return(tf.Variable(initial))

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return(tf.Variable(initial))


######卷积层、池化层，定义创建函数
def conv2d(x,W):
    return(tf.nn.conv2d(x,W,strides=[1,1,1,1],padding\
                        ='SAME'))

def max_pool_2x2(x):
    return(tf.nn.max_pool(x,ksize=[1,2,2,1],strides\
                          =[1,2,2,1],padding='SAME'))

#####定义输入的placeholder.
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])




#####定义第一个卷积层
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)



#####定义第二个卷积层
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)


#####定义全连接层
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)


#####为减轻过拟合，使用Dropout层.
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)


#####将Dropout层的输出连接一个Softmax层，得到最后的概率
#####输出.
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


######定义损失函数为cross entropy,优化器使用Adam.
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*\
               tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(\
                                 cross_entropy)

######定义评测准确率的操作.
correct_prediction=tf.equal(tf.argmax(y_conv,1),\
                            tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,\
                                tf.float32))

######训练过程
####训练时Dropout的keep_prob比率为0.5，使用大小为50的mini-batch,
##共进行20000次训练迭代.每训练100次，对准确率进行一次评测
###keep_prob设为1，用以实时监测模型的性能.

tf.global_variables_initializer().run()
for i in range(20000):
    batch=mnist.train.next_batch(50)
    if(i%100==0):
        train_accuracy=accuracy.eval(feed_dict=\
                    {x:batch[0],y_:batch[1],keep_prob:1.0})
        print('step %d, training accuracy %g' % (i,\
                                        train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],\
                              keep_prob:0.5})


#########训练完成后，在最终的测试集上进行全面测试，得到整体
##的分类准确率.
print('test accuracy %g' % accuracy.eval(feed_dict=\
            {x:mnist.test.images,y_:mnist.test.labels,\
             keep_prob:1.0}))

















