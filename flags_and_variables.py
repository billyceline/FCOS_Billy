import tensorflow as tf

tf.app.flags.DEFINE_integer('image_height', 800, "image height.")
tf.app.flags.DEFINE_integer('image_width', 1024, "image width.")
tf.app.flags.DEFINE_integer('batch_size',6, "Batch size for training.")
tf.app.flags.DEFINE_integer('num_class', 20, "Actual num of class +1.")
tf.app.flags.DEFINE_float('weight_decay', 3e-4, "Weight decay for l2 regularization.")
tf.app.flags.DEFINE_float('learning_rate', 1e-5, "Learning rate for gradient decent.")
tf.app.flags.DEFINE_string('log_dir','./log','tensorboard directory')
tf.app.flags.DEFINE_string('checkpoint_dir','/media/xinje/New Volume/fcos/resnet_v2_50_giou/','The directory where to save the parameters of the network')
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')


inputs = tf.placeholder(tf.float32,[None,800,1024,3],'inputs')
boxes =  tf.placeholder(tf.float32,[None,17064,4],'gt_boxes')
classes = tf.placeholder(tf.float32,[None,17064,20],'gt_classes')
centerness = tf.placeholder(tf.float32,[None,17064,1],'gt_centerness')
state = tf.placeholder(tf.float32,[None,17064],'gt_state')


feature_size=[(100,128),(50,64),(25,32),(13,16),(7,8)]
feature_layer_list = ['P3','P4','P5','P6','P7']
stride=[8,16,32,64,128]
iter_size = 20

corresponding_dict = {0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',5:'bus',6:'car',\
                      7:'chair',8:'cow',19:'diningtable',9:'dog',10:'horse',11:'motorbike',\
                     12:'pottedplant',13:'sheep',14:'sofa',15:'train',16:'tvmonitor',17:'cat',18:'person'}
