import tensorflow as tf
#from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_50,resnet_v2,resnet_arg_scope
from resnet_v2 import *
from vgg import *
from flags_and_variables import *
from tensorflow.python.keras.initializers import he_normal
import numpy as np

def get_fpn_output(inputs,is_training,is_trainable,backbone='resnet_v2_50'):
    if(backbone=='resnet_v2_50'):
        with tf.contrib.slim.arg_scope(resnet_arg_scope(weight_decay=FLAGS.weight_decay,is_trainable = is_trainable)):
            net,end_points = resnet_v2_50(inputs,num_classes=FLAGS.num_class,
                             is_training=is_training,is_trainable = is_trainable,
                             global_pool=False,
                             output_stride=None)

        C3 = end_points['resnet_v2_50/block1']
        C4 = end_points['resnet_v2_50/block2']
        C5 = end_points['resnet_v2_50/block4']
    if(backbone=='vgg_16'):

        with tf.contrib.slim.arg_scope(vgg_arg_scope(weight_decay=FLAGS.weight_decay)):
            net,end_points = vgg_16(inputs,num_classes=FLAGS.num_class,spatial_squeeze=False,dropout_keep_prob=1,is_training=True)
        C3 = end_points['vgg_16/pool3']
        C4 = end_points['vgg_16/pool4']
        C5 = end_points['vgg_16/pool5']
    
    with tf.variable_scope("output") as scope:
        P3 = tf.layers.conv2d(C3, 256, [1,1], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
        P4 = tf.layers.conv2d(C4, 256, [1,1], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
        P5 = tf.layers.conv2d(C5, 256, [1,1], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
        P6 = tf.layers.conv2d(P5, 256, [3,3], 2, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
        P7 = tf.layers.conv2d(P6, 256, [3,3], 2, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
        P4 = tf.image.resize_bilinear(P5,(P5.shape[1]*2,P5.shape[2]*2))+P4
        P3 = tf.image.resize_bilinear(P5,(P4.shape[1]*2,P4.shape[2]*2))+P3
        P3 = tf.layers.conv2d(P3, 256, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
        P4 = tf.layers.conv2d(P4, 256, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))


    pyramid_dict={}
    pyramid_dict['P3'] = P3
    pyramid_dict['P4'] = P4
    pyramid_dict['P5'] = P5
    pyramid_dict['P6'] = P6
    pyramid_dict['P7'] = P7
    return pyramid_dict



def get_network_output(feature_layer_list,pyramid_dict,feature_size,is_training=True):
    localization = []
    centerness = []
    classes = []
    with tf.variable_scope("output2") as scope:
        for index,p in enumerate(feature_layer_list):
            layer = pyramid_dict[p]
            for i in range(4):#four convolutional layer
                layer = tf.layers.conv2d(layer, 256, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))  
                layer = tf.layers.batch_normalization(layer, training=is_training)
                layer = tf.nn.relu(layer)
            centerness_tensor = tf.layers.conv2d(layer, 1, [3,3], 1, 'same',use_bias=True ,kernel_initializer=he_normal(seed=1),kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay),name=('centerness_'+p))
            classes_tensor = tf.layers.conv2d(layer, FLAGS.num_class, [3,3], 1, 'same', bias_initializer= tf.constant_initializer(-np.log((1-0.01)/0.01)),kernel_initializer=he_normal(seed=1),kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay),name=('classes_'+p))
            layer = pyramid_dict[p]
            for i in range(4):
                layer = tf.layers.conv2d(layer, 256, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
                layer = tf.layers.batch_normalization(layer, training=is_training)
                layer = tf.nn.relu(layer)
            localization_tensor = tf.layers.conv2d(layer, 4, [3,3], 1, 'same', use_bias=True,kernel_initializer=he_normal(seed=1),kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay),name=('localization'+p))
            centerness.append(tf.reshape(centerness_tensor,(-1,(feature_size[index][0]*feature_size[index][1]),1)))
            classes.append(tf.reshape(classes_tensor,(-1,(feature_size[index][0]*feature_size[index][1]),FLAGS.num_class)))
            localization.append(tf.reshape(tf.exp(localization_tensor),(-1,(feature_size[index][0]*feature_size[index][1]),4))) 
    return centerness,classes,localization

