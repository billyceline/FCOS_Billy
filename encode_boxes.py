import numpy as np
from flags_and_variables import *
###get ymin ,xmin ,ymax,xmax seperatelty
def encode_boxes(annotation_batch,cls_batch,feature_size,stride):
    m_w = np.array([-1,64,128,256,512,np.inf])/FLAGS.image_width
    m_h = np.array([-1,64,128,256,512,np.inf])/FLAGS.image_height
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
        m_w_min =m_w[feature_index]
        m_w_max =m_w[feature_index+1]
        m_h_min =m_h[feature_index]
        m_h_max =m_h[feature_index+1]
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

        tblr = np.concatenate([t_true,b_true,l_true,r_true],axis=-1)#(6, 30, 100, 128, 4)

        tblr_mask = np.expand_dims(np.logical_and((np.max(tblr,axis=-1)>m_h_min),(np.max(tblr,axis=-1)<=m_w_max)),axis=-1)#(b,N,W,H,1)
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
    matched_true_boxes = np.concatenate(matched_true_boxes,axis=1)
    matched_true_classes = np.concatenate(matched_true_classes,axis=1)
    matched_true_centerness = np.concatenate(matched_true_centerness,axis=1)
    return matched_true_boxes,matched_true_classes,matched_true_centerness
