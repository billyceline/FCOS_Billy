import numpy as np
from flags_and_variables import *
###get ymin ,xmin ,ymax,xmax seperatelty

def encode_boxes(annotation_batch,cls_batch,feature_size,stride,concatenate=True):
    m = np.array([-1,64,128,256,512,1e8])
    ymin_true = np.expand_dims(annotation_batch[...,0],axis=2)#(b,N,1)
    ymax_true = np.expand_dims(annotation_batch[...,2],axis=2)
    xmin_true = np.expand_dims(annotation_batch[...,1],axis=2)
    xmax_true = np.expand_dims(annotation_batch[...,3],axis=2)

    l_t_r_b_true = []
    x_list=[]
    y_list=[]
    matched_true_boxes = []
    matched_true_classes = []
    matched_true_centerness = []
    for feature_index in range(5):
        m_min =m[feature_index]
        m_max =m[feature_index+1]
        offset = np.math.floor(stride[feature_index]/2)
        y_center_mapping = np.array([(i*stride[feature_index]+offset) for i in range(feature_size[feature_index][0])])/FLAGS.image_height #[w]
        x_center_mapping = np.array([(i*stride[feature_index]+offset) for i in range(feature_size[feature_index][1])])/FLAGS.image_width#[H]
#         x_list.append(x_center_mapping)
#         y_list.append(y_center_mapping)

    ####whether center in box
        y_mask = np.logical_and((y_center_mapping>ymin_true),(y_center_mapping<ymax_true))#[b,N,W]
        x_mask = np.logical_and((x_center_mapping>xmin_true),(x_center_mapping<xmax_true))#[b,N,H]
        y_mask_f = y_mask.astype(np.float32)
        x_mask_f = x_mask.astype(np.float32)

    #calculate top,bottom,left,right and concate together  
        t_true = (y_center_mapping - ymin_true)*y_mask_f
        #t_true = np.expand_dims(np.expand_dims(t_true,3),axis=-1) * np.ones([y_mask.shape[0],y_mask.shape[1],y_mask.shape[2],x_mask.shape[2],1])
        t_true = np.tile(np.expand_dims(np.expand_dims(t_true,3),axis=-1), [1,1,1,x_mask.shape[2],1]) #(6, 30, 100, 128, 1)

        b_true = (ymax_true - y_center_mapping)*y_mask_f
        b_true = np.tile(np.expand_dims(np.expand_dims(b_true,3),axis=-1), [1,1,1,x_mask.shape[2],1])#(6, 30, 100, 128, 1)

        l_true = (x_center_mapping - xmin_true)*x_mask_f
        l_true = np.tile(np.expand_dims(np.expand_dims(l_true,2),axis=-1),[1,1,y_mask.shape[2],1,1])#(6, 30, 100, 128, 1)

        r_true = (xmax_true - x_center_mapping)*x_mask_f
        r_true = np.tile(np.expand_dims(np.expand_dims(r_true,2),axis=-1),[1,1,y_mask.shape[2],1,1])#(6, 30, 100, 128, 1)

        tblr = np.concatenate([t_true,b_true,l_true,r_true],axis=-1)
        tblr_temp = np.concatenate([t_true*FLAGS.image_height,b_true*FLAGS.image_height,l_true*FLAGS.image_width,r_true*FLAGS.image_width],axis=-1)#(6, 30, 100, 128, 4)

        tblr_mask = np.expand_dims(np.logical_and((np.max(tblr_temp,axis=-1)>m_min),(np.max(tblr_temp,axis=-1)<=m_max)),axis=-1)#(b,N,W,H,1)
        xy_mask = (np.logical_and(np.expand_dims(y_mask,axis=3),np.expand_dims(x_mask,axis=2))).astype(np.float32)
        xy_mask = np.expand_dims(xy_mask,axis=-1)#(?,?,100,128,1)

        true_boxes = np.expand_dims(np.expand_dims(annotation_batch,axis=2),axis=3)#(?,?,1,1,4)
        true_classes = np.expand_dims(np.expand_dims(cls_batch,axis=2),axis=3)
        #encode true boxes to feature map
        true_boxes = true_boxes * xy_mask * tblr_mask#(?,?,100,128,4)
        true_classes = true_classes * xy_mask * tblr_mask
        tblr = tblr *xy_mask *tblr_mask

        area = (true_boxes[...,2]-true_boxes[...,0])*(true_boxes[...,3]-true_boxes[...,1])
        area[area==0] = 1000000 #exclude 0
        index_min = np.argmin(area,axis=1)
        #get the corresponding minimal area encoded boxes
        pos_true_boxes=[]
        pos_true_classes=[]
        for b in range(FLAGS.batch_size):
            w_boxes=[]
            w_classes=[]
            for w in range(feature_size[feature_index][0]):
                h_boxes = []
                h_classes=[]
                for h in range(feature_size[feature_index][1]):
                    h_boxes.append(tblr[b,:,w,h,:][index_min[b,w,h]])
                    h_classes.append(true_classes[b,:,w,h,:][index_min[b,w,h]])
                w_boxes.append(h_boxes)
                w_classes.append(h_classes)
            pos_true_boxes.append(w_boxes)
            pos_true_classes.append(w_classes)
        pos_true_boxes = np.array(pos_true_boxes)
        pos_true_classes = np.squeeze(np.array(pos_true_classes)).astype(np.int32)
        ###calculate centerness
        tb_max = np.max(pos_true_boxes[...,0:2],axis=-1)
        tb_min = np.min(pos_true_boxes[...,0:2],axis=-1)
        lr_max = np.max(pos_true_boxes[...,2:],axis=-1)
        lr_min = np.min(pos_true_boxes[...,2:],axis=-1)
        centerness = np.expand_dims(np.sqrt((lr_min/(lr_max+1e-8)) * (tb_min/(tb_max+1e-8))),axis=-1)
        matched_true_boxes.append(np.reshape(pos_true_boxes,(-1,feature_size[feature_index][0]*feature_size[feature_index][1],4)))
        matched_true_classes.append(np.reshape(pos_true_classes,(-1,feature_size[feature_index][0]*feature_size[feature_index][1],FLAGS.num_class)))
        matched_true_centerness.append(np.reshape(centerness,(-1,feature_size[feature_index][0]*feature_size[feature_index][1],1)))
    if(concatenate == True):
        matched_true_boxes = np.concatenate(matched_true_boxes,axis=1)
        matched_true_classes = np.concatenate(matched_true_classes,axis=1)
        matched_true_centerness = np.concatenate(matched_true_centerness,axis=1)
    return matched_true_boxes,matched_true_classes,matched_true_centerness

def predict_outputs(centerness_pred,classes_pred,localization_pred,feature_size):
    m_w = np.array([-1,64,128,256,512,np.inf])/FLAGS.image_width
    m_h = np.array([-1,64,128,256,512,np.inf])/FLAGS.image_height
    center_list = []
    m_h_min_list = []
    m_w_max_list = []
    for i in range(5):
        #change top,bottom,left,right to ymin,xmin,ymax,xmax
    # feature_size=[(100,128),(50,64),(25,32),(13,16),(7,8)]
    # stride=[8,16,32,64,128]
        m_w_min =m_w[i]
        m_w_max =m_w[i+1]
        m_h_min =m_h[i]
        m_h_max =m_h[i+1]
        #ensure the predicted boxes max(top,bottom,left,right) is in the domain(m_h_min,m_w_max)
            #the mim(top,bottom,left,right) is bigger than 1 pixel
        offset = np.math.floor(stride[i]/2)
        y_center_mapping = np.array([(j*stride[i]+offset) for j in range(feature_size[i][0])])/FLAGS.image_height
        x_center_mapping = np.array([(j*stride[i]+offset) for j in range(feature_size[i][1])])/FLAGS.image_width
        y_center_mapping = np.expand_dims(np.tile(np.expand_dims(y_center_mapping,axis=-1),[1,feature_size[i][1]]),axis=-1)
        x_center_mapping = np.expand_dims(np.tile(np.expand_dims(x_center_mapping,axis=0),[feature_size[i][0],1]),axis=-1)
        center = np.concatenate([y_center_mapping,x_center_mapping],axis=-1).reshape(-1,(feature_size[i][0]*feature_size[i][1]),2)
        center_list.append(center)
        m_h_min = np.ones_like(y_center_mapping)*m_h_min
        m_w_max = np.ones_like(x_center_mapping)*m_w_max
        m_h_min_list.append(m_h_min.reshape(-1,(feature_size[i][0]*feature_size[i][1])))
        m_w_max_list.append(m_w_max.reshape(-1,(feature_size[i][0]*feature_size[i][1])))
    center_list = np.concatenate(center_list,axis=1) #(1, 17064, 2)
    m_h_min_list = np.concatenate(m_h_min_list,axis=1)#(1, 17064)
    m_w_max_list = np.concatenate(m_w_max_list,axis=1)#(1, 17064)
    
#     localization_mask = tf.expand_dims(tf.logical_and((tf.reduce_min(localization_pred,axis=-1)>m_h_min_list),(tf.reduce_max(localization_pred,axis=-1)<=m_w_max_list)),axis=-1)
    localization_mask_2 = tf.cast(tf.expand_dims((tf.reduce_min(localization_pred,axis=-1)>0.001),axis=-1),tf.float32)
#     localization_mask = tf.cast(localization_mask,tf.float32) * localization_mask_2
    localization_pred = localization_pred * localization_mask_2
    centerness_pred  = centerness_pred *localization_mask_2
    classes_pred = classes_pred * localization_mask_2
    
    
    ymin = tf.expand_dims((center_list[...,0]-localization_pred[...,0]) * FLAGS.image_height,axis=-1)
    ymax = tf.expand_dims((center_list[...,0]+localization_pred[...,1]) * FLAGS.image_height,axis=-1)
    xmin = tf.expand_dims((center_list[...,1]-localization_pred[...,2]) * FLAGS.image_width,axis=-1)
    xmax = tf.expand_dims((center_list[...,1]+localization_pred[...,3]) * FLAGS.image_width,axis=-1)
    localization_pred = tf.concat([ymin,xmin,ymax,xmax],axis=-1)
    centerness_pred = tf.squeeze(centerness_pred,axis=0)
    classes_pred = tf.squeeze(classes_pred,axis=0)
    localization_pred = tf.squeeze(localization_pred,axis=0)
    
    score_pred = centerness_pred*classes_pred
#     score_pred = classes_pred
    mask = (score_pred > 0.3)
    
    _boxes = []
    _classes = []
    _scores = []
    for c in range(FLAGS.num_class):
    ##nms
        _localization_pred = tf.boolean_mask(localization_pred,mask[:,c])
        _scores_pred = tf.boolean_mask(score_pred[:,c],mask[:,c])
        nms_index = tf.image.non_max_suppression(
            _localization_pred, _scores_pred, 10, iou_threshold = 0.5)
        _boxes.append(tf.gather(_localization_pred, nms_index))
        _scores_pred = tf.gather(_scores_pred,nms_index)
        _scores.append(_scores_pred)
        _classes.append(tf.ones_like(_scores_pred, 'int32') * c)
    _boxes = tf.concat(_boxes,axis=0)
    _scores = tf.concat(_scores,axis=0)
    _classes = tf.concat(_classes,axis=0)
    return _scores,_classes,_boxes
