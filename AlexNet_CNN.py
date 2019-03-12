# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 01:21:22 2018

@author: hecongcong
"""

############# Tensorflow实现AlexNet.
##AlexNet有8个需要训练的层，前5个卷积层和后3层全连接层.
##LRN层出现在第一个及第二个卷积层后，最大池化出现在两个LRN层
##及最后一个卷积层后，RELU激活函数应用在8层的每一层后面.

from datetime import datetime
import math
import time
import tensorflow as tf


batch_size=32
num_batches=100


#####定义一个用来显示网络每一层结构的函数，展示每一个卷积层
###或池化层输出tensor的尺寸.
def print_activations(t):   #t为一个tensor.
    print(t.op.name , ' ' , t.get_shape().as_list())


#########设计AlexNet的网络结构.
####构建第一个卷积层conv1.

def inference(images):
    parameters=[]
    
    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],\
                dtype=tf.float32,stddev=1e-1),name='weight')
        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding=\
                          'SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[64],\
                dtype=tf.float32),trainable=True,name='bias')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)
        print_activations(conv1)
        parameters+=[kernel,biases]
    
    
    ###在第一个卷积层后再添加LRN层和最大池化层.
    '''
    lrn1=tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,\
                   name='lrn1')
    '''
    pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],\
                         padding='VALID',name='pool1')
    print_activations(pool1)
    
    
    #######设计第二个卷积层.
    with tf.name_scope('conv2') as scope:
        kernel=tf.Variable(tf.truncated_normal([5,5,64,192],\
                dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding=\
                          'SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[192],dtype=\
                        tf.float32),trainable=True,name='bias')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
    
        print_activations(conv2)
    
    
    ####在第二个卷积层后再添加LRN层和最大池化层.
    '''
    lrn2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,\
                   name='lrn2')
    '''
    pool2=tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=\
                [1,2,2,1],padding='VALID',name='pool2')
    print_activations(pool2)
    
    
    
    ######创建第三个卷积层.
    
    with tf.name_scope('conv3') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,192,384],\
            dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding=\
                          'SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[384],\
                dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv3)
        
        
    #######构建第四个卷积层.
    with tf.name_scope('conv4') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],\
            dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding=\
                          'SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[256],\
                    dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv4)
        
        
    #######构建第五个卷积层.
    with tf.name_scope('conv5') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,256,256],\
            dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding=\
                          'SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[256],\
                        dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv5)
    
    
    ####在第五个卷积层后再添加最大池化层.
    pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=\
                    [1,2,2,1],padding='VALID',name='pool5')
    print_activations(pool5)
    return(pool5,parameters)





############实现一个评估AlexNet每轮计算时间的函数time_tensorflow_run.
def time_tensorflow_run(session,target,info_string):
    num_steps_burn_in=10
    total_duration=0.0
    total_duration_squared=0.0
    
    for i in range(num_batches+num_steps_burn_in):
        start_time=time.time()
        _=session.run(target)
        duration=time.time()-start_time
        if(i>=num_steps_burn_in):
            if(not i%10):
                print('%s: step %d, duration = %.3f' % \
                      (datetime.now(),i-num_steps_burn_in,\
                       duration))
            total_duration+=duration
            total_duration_squared+=duration*duration
    mn=total_duration/num_batches
    vr=total_duration_squared/num_batches-mn*mn #方差
    sd=math.sqrt(vr)  #标准差
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' \
          % (datetime.now(),info_string,num_batches,mn,sd))
    
    
            
    
#########构造主函数run_benchmark.
def run_benchmark():
    with tf.Graph().as_default():
        image_size=224
        images=tf.Variable(tf.random_normal([batch_size,\
                                             image_size,\
                                             image_size,3],\
                                             dtype=tf.float32,\
                                             stddev=1e-1))
        pool5,parameters=inference(images)
        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)
        
        
        ######进行前向和后向的评测.
        time_tensorflow_run(sess,pool5,'Forward')
        
        objective=tf.nn.l2_loss(pool5)
        grad=tf.gradients(objective,parameters)
        time_tensorflow_run(sess,grad,'Forward-backward')
        
      
    
    
    
        
        
























