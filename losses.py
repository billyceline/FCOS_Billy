import tensorflow as tf

def focal_loss(y_true, y_pred,anchor_state):
    """
    Compute the focal loss given the target tensor and the predicted tensor.
    As defined in https://arxiv.org/abs/1708.02002
    Args
        y_true: Tensor of target data from the generator with shape (B, N, num_classes).
        y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
    Returns
        The focal loss of y_pred w.r.t. y_true.
    """
    with tf.compat.v1.variable_scope("loss/focal") as scope:
        alpha=0.25
        gamma=2.0
        # compute the focal loss
        location_state = anchor_state
        labels = y_true
        # alpha 参与用于调节正负样本的平衡问题
        alpha_factor = tf.ones_like(labels) * alpha
        alpha_factor = tf.where(tf.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # focal_weight 用来使置信度较高容易的样本的 loss 比原来小, 而置信度低的难度大的 loss 变化不大
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(tf.equal(labels, 1), 1 - tf.sigmoid(y_pred), tf.sigmoid(y_pred))
        focal_weight = alpha_factor * focal_weight ** gamma
        # binary_crossentropy 是  -log(p) y=1 -log(1-p) y=others, 那么论文中统一的用 -log(pt) 来表示
        cls_loss = focal_weight * tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=y_pred)
#         cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(target=labels,output=y_pred)
        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(tf.equal(location_state, 1))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)
        loss = tf.reduce_sum(cls_loss) / normalizer
    return loss

def iou_loss(y_true, y_pred,location_state,centerness):
    with tf.compat.v1.variable_scope("loss/iou") as scope:
        # pos location
        indices = tf.where(tf.equal(location_state, 1))
    #     print(indices.shape)
        if tf.size(indices) == 0:
            return tf.constant(0.0)
        y_regr_pred = tf.gather_nd(y_pred, indices)
        y_regr_true = tf.gather_nd(y_true, indices)
        y_centerness_true = tf.gather_nd(centerness,indices)
        # (num_pos, )
        pred_top = y_regr_pred[:, 0]
        pred_bottom = y_regr_pred[:, 1]
        pred_left = y_regr_pred[:, 2]
        pred_right = y_regr_pred[:, 3]

        # (num_pos, )
        target_top = y_regr_true[:, 0]
        target_bottom = y_regr_true[:, 1]
        target_left = y_regr_true[:, 2]
        target_right = y_regr_true[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)

        g_w_intersect = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
        g_h_intersect = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
        ac_union = g_w_intersect * g_h_intersect + 1e-7
        
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect*(800*1024) + 1.0) / (area_union*(800*1024) + 1.0)
        losses = -tf.math.log(ious)

        losses =  tf.reduce_sum(losses)/tf.reduce_sum(location_state)
    return losses

def centerness_loss(y_true, y_pred,location_state):
    with tf.compat.v1.variable_scope("loss/bce") as scope:
    #         y_true = tf.squeeze(y_true,axis=-1)
    #         y_pred = tf.squeeze(y_pred,axis=-1)
        #  pos location
        indices = tf.where(tf.equal(location_state, 1))
        if tf.size(indices) == 0:
            return tf.constant(0.0)
        y_centerness_pred = tf.gather_nd(y_pred, indices)
        y_true = tf.gather_nd(y_true, indices)
        y_centerness_true = y_true
        loss = tf.cond((tf.size(y_centerness_true) > 0),lambda:tf.keras.backend.binary_crossentropy(output=y_centerness_pred,target=y_centerness_true, from_logits=True), lambda:tf.zeros_like(y_centerness_pred))
        loss = tf.reduce_mean(loss)  
    return loss
