import tensorflow as tf

tf.app.flags.DEFINE_integer('image_height', 800, "image height.")
tf.app.flags.DEFINE_integer('image_width', 1024, "image width.")
tf.app.flags.DEFINE_integer('batch_size',6, "Batch size for training.")

tf.app.flags.DEFINE_float('weight_decay', 1e-5, "Weight decay for l2 regularization.")
tf.app.flags.DEFINE_float('learning_rate', 1e-5, "Learning rate for gradient decent.")
tf.app.flags.DEFINE_string('log_Dir','./log','tensorboard directory')
tf.app.flags.DEFINE_string('checkpoint_dir','/media/xinje/New Volume/fcos/resnet_v2_50_freeze_bn/','The directory where to save the parameters of the network')

tf.app.flags.DEFINE_string('dataset_name','voc','voc or coco')
tf.app.flags.DEFINE_string('backbone','resnet_v2_50','resnet_v2_50 or vgg16')

tf.app.flags.DEFINE_string('train_2012_dir','/media/xinje/New Volume/VOC07&12/VOC2012/train_2012.txt','The voc2012 train.txt file location')
tf.app.flags.DEFINE_string('train_2007_dir','/media/xinje/New Volume/VOC07&12/VOC2007/train/train_2007.txt','The voc2007 train.txt file location')
tf.app.flags.DEFINE_string('test_2012_dir','/media/xinje/New Volume/VOC07&12/VOC2012/val_2012.txt','The voc2012test.txt file location')
tf.app.flags.DEFINE_string('test_2007_dir','/media/xinje/New Volume/VOC07&12/VOC2007/test/test_2007.txt','The voc2007 test.txt file location')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')

if FLAGS.dataset_name =='voc':
    tf.app.flags.DEFINE_integer('num_class', 20, "Actual num of class.")
    corresponding_dict = {0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',5:'bus',6:'car',\
                      7:'chair',8:'cow',19:'diningtable',9:'dog',10:'horse',11:'motorbike',\
                     12:'pottedplant',13:'sheep',14:'sofa',15:'train',16:'tvmonitor',17:'cat',18:'person'}
if FLAGS.dataset_name =='coco':
    tf.app.flags.DEFINE_integer('num_class', 80, "Actual num of class.")
    corresponding_dict = {}



inputs = tf.compat.v1.placeholder(tf.float32,[None,FLAGS.image_height,FLAGS.image_width,3],'inputs')
boxes =  tf.compat.v1.placeholder(tf.float32,[None,17064,4],'gt_boxes')
classes = tf.compat.v1.placeholder(tf.float32,[None,17064,FLAGS.num_class],'gt_classes')
centerness = tf.compat.v1.placeholder(tf.float32,[None,17064,1],'gt_centerness')
state = tf.compat.v1.placeholder(tf.float32,[None,17064],'gt_state')



feature_size=[(100,128),(50,64),(25,32),(13,16),(7,8)]
feature_layer_list = ['P3','P4','P5','P6','P7']
stride=[8,16,32,64,128]
iter_size = 20
inference_threshold = 0.0
