# -*- coding: utf8 -*- 
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import math
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, f1_score
import warnings
import sys

# SET ENV
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
config.gpu_options.allow_growth=True

source_x = sys.argv[1]
source_y = sys.argv[2]
target_x = sys.argv[3]
target_y = sys.argv[4]

source_x = pd.read_csv(source_x, dtype=np.float32).values
source_y = pd.read_csv(source_y, dtype=np.float32).values

target_x = pd.read_csv(target_x, dtype=np.float32).values
target_y_df = pd.read_csv(target_y, dtype=np.float32)
target_y = pd.read_csv(target_y, dtype=np.float32).values


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

dir_save = './result/'        
createDirectory(dir_save)

n_features = len(source_x[0])
n_classes = len(source_y[0])

batch_iter = 5.0  
batch_size_s = int(math.ceil(len(source_x)/batch_iter))
batch_size_t = int(math.ceil(len(target_x)/batch_iter))

n_sm_h1= 1024
n_sm_h2= 512

n_sm_h1_classification=256
n_sm_h2_classification=128
n_sm_h3_classification=64

n_sm_h1_disc = 128
n_sm_h2_disc = 64

X = tf.placeholder(tf.float32, [None, n_features])
Y = tf.placeholder(tf.float32, [None, n_classes])
X_latent = tf.placeholder(tf.float32, [None, n_sm_h2])
X_latent_target = tf.placeholder(tf.float32, [None, n_sm_h2])

# for dropout
keep_prob_source = tf.placeholder(tf.float32)
keep_prob_target = tf.placeholder(tf.float32)
keep_prob_disc = tf.placeholder(tf.float32)
keep_prob_classifier = tf.placeholder(tf.float32)

alpha = tf.placeholder(tf.float32,)
group_lambda = tf.placeholder(tf.float32,)

group_max = tf.placeholder(tf.float32,)
mmd_max = tf.placeholder(tf.float32,)

# for batch normalization
phase_source = tf.placeholder(tf.bool, name='phase_source')
phase_target = tf.placeholder(tf.bool, name='phase_target')
phase_classifier = tf.placeholder(tf.bool, name='phase_classifier')

X_target = tf.placeholder(tf.float32, [None, n_features])
Y_target = tf.placeholder(tf.float32, [None, n_classes])

handle = tf.placeholder(tf.string, shape=[])
handle2 = tf.placeholder(tf.string, shape=[])

# Dataset and iterator
dataset_source_batch = tf.data.Dataset.from_tensor_slices((X, Y))
dataset_source_batch = dataset_source_batch.shuffle(buffer_size=len(source_x))
dataset_source_batch = dataset_source_batch.batch(batch_size_s).prefetch(batch_size_s*2)
iterator_source = dataset_source_batch.make_initializable_iterator()
iter = tf.data.Iterator.from_string_handle(handle, dataset_source_batch.output_types, dataset_source_batch.output_shapes)
iter_X, iter_Y = iter.get_next()

dataset_target_batch = tf.data.Dataset.from_tensor_slices((X_target, Y_target))
dataset_target_batch = dataset_target_batch.shuffle(buffer_size=len(target_x))
dataset_target_batch = dataset_target_batch.batch(batch_size_t).prefetch(batch_size_t*2)
iterator_target = dataset_target_batch.make_initializable_iterator()
iter_target = tf.data.Iterator.from_string_handle(handle2, dataset_target_batch.output_types, dataset_target_batch.output_shapes)
iter_target_X, iter_target_Y = iter_target.get_next()

dataset_target = tf.data.Dataset.from_tensor_slices((X_target, Y_target))
dataset_target = dataset_target.batch(len(target_x))
iterator_target_all = dataset_target.make_initializable_iterator()

dataset_source_all = tf.data.Dataset.from_tensor_slices((X, Y))
dataset_source_all = dataset_source_all.batch(len(source_x))
iterator_source_all = dataset_source_all.make_initializable_iterator()

# source encoder latent
dataset_source_batch_latent = tf.data.Dataset.from_tensor_slices((X_latent, Y))
dataset_source_batch_latent = dataset_source_batch_latent.shuffle(buffer_size=len(source_x))
dataset_source_batch_latent = dataset_source_batch_latent.batch(batch_size_s).prefetch(batch_size_s*2)
iterator_source_batch_latent = dataset_source_batch_latent.make_initializable_iterator()
iter_source_latent = tf.data.Iterator.from_string_handle(handle, dataset_source_batch_latent.output_types, dataset_source_batch_latent.output_shapes)
iter_source_latent_X, iter_source_latent_Y = iter_source_latent.get_next()

dataset_unlabel = tf.data.Dataset.from_tensor_slices((X_latent_target, Y_target))
dataset_unlabel = dataset_unlabel.shuffle(buffer_size=len(target_x))
dataset_unlabel = dataset_unlabel.batch(batch_size_t).prefetch(batch_size_t*2)
iterator_unlabel = dataset_unlabel.make_initializable_iterator()
iter_unlabel = tf.data.Iterator.from_string_handle(handle2, dataset_unlabel.output_types, dataset_unlabel.output_shapes)
iter_X_unlabel, iter_Y_unlabel = iter_unlabel.get_next()

dataset_unlabel_all = tf.data.Dataset.from_tensor_slices((X_latent_target, Y_target))
dataset_unlabel_all = dataset_unlabel_all.batch(len(target_x))
iterator_unlabel_all = dataset_unlabel_all.make_initializable_iterator()

dataset_label = tf.data.Dataset.from_tensor_slices((X_latent, Y))
dataset_label = dataset_label.shuffle(buffer_size=len(source_x))
dataset_label = dataset_label.batch(batch_size_s).prefetch(batch_size_s*2)
iterator_label = dataset_label.make_initializable_iterator()
iter_label = tf.data.Iterator.from_string_handle(handle, dataset_label.output_types, dataset_label.output_shapes)
iter_X_label, iter_Y_label = iter_label.get_next()

epochs1 = 1000
epochs2 = 2000
epochs3 = 3000

learning_rate = 1e-4   
max_target_acc_step2 = 0.0
max_target_pred_step2 = []
max_target_acc_step3 = 0.0
max_target_pred_step3 = []

g_lr=tf.placeholder(tf.float32,)
d_lr=tf.placeholder(tf.float32,)
g_lr_value = 1e-5
d_lr_value = 1e-5

learning_rate_classification = 1e-5 

# dropout rate
keep_prob_classifier_rate = 0.7
keep_prob_source_rate = 0.5
keep_prob_target_rate = 0.5
keep_prob_disc_rate = 0.5

a = 0.0
cur_a = 0.0
af = 1.0    
t1 = 0      
t2 = 1000  

disc_pretrain = 0  
smooth = 0.1
unlabel_probability = 0.0 
gaussian_noise_std = 0.2
gaussian_noise_std_strong = 0.8
group_lambda_value = 0.1   

source_latent = []
target_latent = []

def fc_bn(_x, _output, _phase, _scope):
    with tf.variable_scope(_scope):
        h1 = tf.contrib.layers.fully_connected(_x, _output, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.variance_scaling_initializer())
        h2 = tf.contrib.layers.batch_norm(h1, updates_collections=None, fused=True, decay=0.9, center=True, scale=True, is_training=_phase, scope='bn')
    return h2

def s_encoder(inputs, _scope, _keep_prob, _phase, reuse):
    with tf.variable_scope(_scope, reuse=reuse):
        en1 = tf.nn.dropout(tf.nn.relu(fc_bn(inputs, n_sm_h1, _phase, "en1")), _keep_prob)
        en2 = tf.nn.tanh(fc_bn(en1, n_sm_h2, _phase, "en2"))
    return en2 

def classifier(inputs, _scope, _keep_prob, _phase, reuse):
    with tf.variable_scope(_scope, reuse=reuse):
        fc1 = tf.nn.dropout(tf.nn.leaky_relu(fc_bn(inputs, n_sm_h1_classification, _phase, "c_dense1")), _keep_prob)
        fc2 = tf.nn.dropout(tf.nn.leaky_relu(fc_bn(fc1, n_sm_h2_classification, _phase, "c_dense2")), _keep_prob)
        fc3 = tf.nn.dropout(tf.nn.leaky_relu(fc_bn(fc2, n_sm_h3_classification, _phase, "c_dense3")), _keep_prob)
        out = tf.contrib.layers.fully_connected(fc3, n_classes, activation_fn=None, scope='sm', weights_initializer=tf.contrib.layers.variance_scaling_initializer())   #batch normalization 제거    
    return out

def build_classify_ce_loss(logits, labels):
    logits_softmax = tf.nn.softmax(logits)
    c_loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits_softmax + 1e-10), axis = 1))
    return c_loss

# focal loss
def build_classify_loss(logits, labels, gamma=1):
    y_pred = tf.nn.softmax(logits)  
    loss = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
    return loss

def eval(logits, labels):
    softmax = tf.nn.softmax(logits)
    pred = tf.argmax(softmax, 1)
    correct_label_predicted = tf.equal(tf.argmax(labels, 1), pred)
    predicted_accuracy = tf.reduce_mean(tf.cast(correct_label_predicted,tf.float32))
    return predicted_accuracy, pred

def discriminator(inputs, _scope, _phase, _keep_prob, reuse=True):
    with tf.variable_scope(_scope, reuse=reuse):
        fc1 = tf.nn.dropout(tf.nn.leaky_relu(fc_bn(inputs, n_sm_h1_disc, _phase, "disc1")), _keep_prob)
        fc2 = tf.nn.dropout(tf.nn.leaky_relu(fc_bn(fc1, n_sm_h2_disc, _phase, "disc2")), _keep_prob)
        d_logits = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=None, scope='domain_out', weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    return d_logits

def build_ad_loss(feat_s, feat_t):
    pred_disc_source = discriminator(feat_s, "disc_1", True, keep_prob_disc, reuse=False)
    pred_disc_target = discriminator(feat_t, "disc_1", True, keep_prob_disc, reuse=True)
    
    d_loss_source = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_disc_source, labels=tf.ones_like(pred_disc_source)))
    d_loss_target = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_disc_target, labels=tf.zeros_like(pred_disc_target)))
    d_loss = d_loss_source + d_loss_target
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_disc_target, labels=tf.ones_like(pred_disc_target)))
    
    return g_loss, d_loss

def build_group_loss(feat_s, feat_t):
    source_logit_cluster = classifier(feat_s, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)
    target_logit_cluster = classifier(feat_t, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)
    source_pred = tf.argmax(tf.nn.softmax(source_logit_cluster), 1)
    target_pred = tf.argmax(tf.nn.softmax(target_logit_cluster), 1)
    
    centers_loss_list = [[0.0]*1 for _ in range(n_classes)]
    
    for i in range(n_classes): 
        pos_source = tf.where(tf.equal(source_pred, i))
        pos_target = tf.where(tf.equal(target_pred, i))
        
        centers_loss_list[i] = tf.cond(tf.logical_and(tf.not_equal(tf.size(pos_source), 0), tf.not_equal(tf.size(pos_target), 0)), lambda: calculate_group_loss(feat_s, feat_t, pos_source, pos_target), lambda: calculate_empty_center_loss())
        
    mask = tf.greater(centers_loss_list, 0)
    non_zero_list = tf.boolean_mask(centers_loss_list, mask)
    center_loss = tf.reduce_mean(non_zero_list)
        
    return center_loss


def calculate_group_loss(feat_s, feat_t, pos_source, pos_target):
    centers_source = tf.reduce_mean(tf.squeeze(tf.gather(feat_s, pos_source)), 0)
    centers_target = tf.reduce_mean(tf.squeeze(tf.gather(feat_t, pos_target)), 0)
    centers_subtract = tf.subtract(centers_source, centers_target)
    centers_loss = tf.sqrt(tf.reduce_sum(tf.square(centers_subtract)))
    return centers_loss

def build_group_loss2(feat_s, feat_t):
    source_logit_cluster = classifier(feat_s, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)
    target_logit_cluster = classifier(feat_t, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)
    source_pred = tf.argmax(tf.nn.softmax(source_logit_cluster), 1)
    target_pred = tf.argmax(tf.nn.softmax(target_logit_cluster), 1)
    
    st_loss_list = [[0.0]*1 for _ in range(n_classes)]
    t_loss_list = [[0.0]*1 for _ in range(n_classes)]
    
    for i in range(n_classes): 
        pos_source = tf.where(tf.equal(source_pred, i))
        pos_target = tf.where(tf.equal(target_pred, i))

        st_loss_list[i] = tf.cond(tf.logical_and(tf.not_equal(tf.size(pos_source), 0), tf.not_equal(tf.size(pos_target), 0)), lambda: calculate_st_group_loss(feat_s, feat_t, pos_source, pos_target), lambda: calculate_empty_center_loss())
        t_loss_list[i] = tf.cond(tf.logical_and(tf.not_equal(tf.size(pos_source), 0), tf.not_equal(tf.size(pos_target), 0)), lambda: calculate_t_group_loss(feat_s, feat_t, pos_source, pos_target), lambda: calculate_empty_center_loss())

    t_loss = calculate_average_group_loss(t_loss_list)
    return t_loss

def calculate_average_group_loss(loss_list):
    # cell type 개수만큼 다 더해서 한 값으로 리턴
    mask = tf.greater(loss_list, 0)
    non_zero_list = tf.boolean_mask(loss_list, mask)
    loss = tf.reduce_mean(non_zero_list)
    return loss

def calculate_st_group_loss(feat_s, feat_t, pos_source, pos_target):
    centers_source = tf.reduce_mean(tf.squeeze(tf.gather(feat_s, pos_source)), 0)
    target_data = tf.squeeze(tf.gather(feat_t, pos_target))
    
    centers_subtract = tf.subtract(centers_source, target_data)
    
    squ = tf.square(centers_subtract)
    sum_squ = tf.cond(tf.not_equal(tf.rank(squ), 1), lambda:tf.reduce_sum(squ, 1), lambda:tf.reduce_sum(squ))
    sqrt = tf.sqrt(sum_squ)
    centers_loss = tf.reduce_mean(sqrt)
    return centers_loss

def calculate_t_group_loss(feat_s, feat_t, pos_source, pos_target):
    centers_target = tf.reduce_mean(tf.squeeze(tf.gather(feat_t, pos_target)), 0)
    target_data = tf.squeeze(tf.gather(feat_t, pos_target))
    
    centers_subtract = tf.subtract(centers_target, target_data)
    
    squ = tf.square(centers_subtract)
    sum_squ = tf.cond(tf.not_equal(tf.rank(squ), 1), lambda:tf.reduce_sum(squ, 1), lambda:tf.reduce_sum(squ))
    sqrt = tf.sqrt(sum_squ)
    centers_loss = tf.reduce_mean(sqrt)
    return centers_loss

def calculate_empty_center_loss():
    centers_loss = [0.0]
    return centers_loss

def gaussian_noise(input, std):
    noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32) 
    return input + noise

def build_consistency_loss(out, aug_out):
    subtract = tf.subtract(out, aug_out)
    consistency_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(subtract), 1)))
    return consistency_loss


source_encoder = s_encoder(iter_X, "s_encoder", keep_prob_source, phase_source, reuse=False)
source_logits = classifier(source_encoder, "source_classifier", keep_prob_classifier, phase_classifier, reuse=False)

# build training accuracy with training batch
accuracy, pred = eval(source_logits, iter_Y)

## for adversarial training
target_encoder = s_encoder(iter_target_X, "s_encoder", keep_prob_target, phase_target, reuse=True)
target_logit_for_s = classifier(target_encoder, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)

accuracy_target, pred_target = eval(target_logit_for_s, iter_target_Y)

## for semi-supervised learning
sm_ = classifier(iter_X_label, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)
sm_out = tf.nn.softmax(sm_)

noise_sm = gaussian_noise(iter_X_label, gaussian_noise_std)
sm_aug_ = classifier(noise_sm, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)
sm_aug_out = tf.nn.softmax(sm_aug_)

pl_ = classifier(iter_X_unlabel, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)

noise = gaussian_noise(iter_X_unlabel, gaussian_noise_std)
pl_aug_ = classifier(noise, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)
pl_aug_out = tf.nn.softmax(pl_aug_)

noise_strong = gaussian_noise(iter_X_unlabel, gaussian_noise_std_strong)
pl_aug2_ = classifier(noise_strong, "source_classifier", keep_prob_classifier, phase_classifier, reuse=True)
pl_aug2_out = tf.nn.softmax(pl_aug2_)

pl_out = tf.nn.softmax(pl_)
pred_unlabel = tf.argmax(pl_out, 1)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops) :
    # for pre-training
    c_loss = build_classify_loss(source_logits, iter_Y)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(c_loss)
    
    # for adversarial training
    g_loss_1, d_loss_1 = build_ad_loss(iter_source_latent_X, target_encoder)
    group_loss = build_group_loss(iter_source_latent_X, target_encoder)
    t_group_loss = build_group_loss2(iter_source_latent_X, target_encoder)
    g_loss = g_loss_1 + (group_loss + t_group_loss)*group_lambda
    d_loss = d_loss_1
    
    t_vars = tf.trainable_variables() 
    g_vars = [var for var in t_vars if "s_encoder" in var.name]
    d_vars = [var for var in t_vars if "disc_1" in var.name]
    
    optim_g = tf.train.AdamOptimizer(learning_rate=g_lr).minimize(g_loss, var_list=g_vars)
    optim_d = tf.train.AdamOptimizer(learning_rate=d_lr).minimize(d_loss, var_list=d_vars)   
    
    label_cost = build_classify_ce_loss(sm_, iter_Y_label)
    unlabel_cost = build_consistency_loss(pl_aug_out, pl_aug2_out)   #MSE : weak aug + strong aug   
    data_cost = tf.add(label_cost, alpha * unlabel_cost)
    data_op = tf.train.AdamOptimizer(learning_rate=learning_rate_classification, name='Adam2').minimize(data_cost)

# ACCURACY
pred_data = tf.argmax(sm_out, 1)
label = tf.argmax(iter_Y_label, 1)
correct_pred = tf.equal(pred_data, label)
accuracy_data = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# session
sess = tf.Session(config=config)
handle_source = sess.run(iterator_source.string_handle())
handle_target = sess.run(iterator_target.string_handle())
handle_source_all = sess.run(iterator_source_all.string_handle())
handle_target_all = sess.run(iterator_target_all.string_handle())
handle_source_batch_latent = sess.run(iterator_source_batch_latent.string_handle())
handle_unlabel = sess.run(iterator_unlabel.string_handle())
handle_unlabel_all = sess.run(iterator_unlabel_all.string_handle())
handle_label = sess.run(iterator_label.string_handle())

sess.run(tf.global_variables_initializer())

for meta_step in ["pre_training", "adversarial_training", "semi_supervised_learing"]:
    if meta_step == "pre_training":
        print("########## Pre-training ##########")
        for epoch in range(epochs1):
            sess.run(iterator_source.initializer, feed_dict={X: source_x, Y: source_y})
            avg_cost = 0
            avg_acc = 0
            while True: 
                try:
                    _, loss, tr_acc_ = sess.run([train_op, c_loss, accuracy], feed_dict={handle: handle_source, keep_prob_source:keep_prob_source_rate, phase_source:True,
                                                                                         keep_prob_classifier:keep_prob_classifier_rate, phase_classifier:True})
                    avg_cost += loss / batch_iter
                    avg_acc += tr_acc_ / batch_iter
                except tf.errors.OutOfRangeError:
                    break
                
            if epoch % 10 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.3f}'.format(avg_cost), ' acc_train =', '{:.3f}'.format(avg_acc))
                
            if epoch+1 == epochs1:
                sess.run(iterator_source_all.initializer, feed_dict={X: source_x, Y: source_y})
                last_source_acc_test, pred_test, source_latent = sess.run([accuracy, pred, source_encoder], feed_dict={handle: handle_source_all, keep_prob_source:1.0, phase_source:False,
                                                                                                                            keep_prob_classifier:1.0, phase_classifier:False})
                
    if meta_step == "adversarial_training":
        print("########## Adversarial training ##########")
        for epoch in range(epochs2):
            sess.run(iterator_source_batch_latent.initializer, feed_dict={X_latent: source_latent, Y: source_y})
            sess.run(iterator_target.initializer, feed_dict={X_target: target_x, Y_target: target_y})
            avg_d_loss = 0
            avg_g_loss = 0
            
            if epoch < 1000:
                g_lr_value = 1e-6
                d_lr_value = 1e-5
            else:
                g_lr_value = 1e-6
                d_lr_value = 1e-6
            
            while True: 
                try:
                    if epoch % 5 == 0:  
                        _d, _g, d_loss_, g_loss_ = sess.run([optim_d, optim_g, d_loss, g_loss], 
                                                        feed_dict={handle: handle_source_batch_latent, handle2: handle_target, 
                                                            keep_prob_target:keep_prob_target_rate, phase_target:True,
                                                            keep_prob_disc:keep_prob_disc_rate,
                                                            keep_prob_classifier:1.0, phase_classifier:False, 
                                                            g_lr:g_lr_value, d_lr:d_lr_value, group_lambda:group_lambda_value})
                    
                    else:
                        _d, d_loss_, g_loss_ = sess.run([optim_d, d_loss, g_loss], 
                                                    feed_dict={handle: handle_source_batch_latent, handle2: handle_target, 
                                                        keep_prob_target:1.0, phase_target:False,
                                                        keep_prob_disc:keep_prob_disc_rate,
                                                        keep_prob_classifier:1.0, phase_classifier:False, 
                                                        g_lr:g_lr_value, d_lr:d_lr_value, group_lambda:group_lambda_value})
                            
                    avg_d_loss += d_loss_ / batch_iter
                    avg_g_loss += g_loss_ / batch_iter
                    
                except tf.errors.OutOfRangeError:
                    break

            if epoch % 10 == 0 :
                print("step:{}, g_loss:{:.3f}, d_loss:{:.3f} ".format(epoch, avg_g_loss, avg_d_loss), end="")
                
                sess.run(iterator_target_all.initializer, feed_dict={X_target: target_x, Y_target: target_y})
                t_acc, t_pred = sess.run([accuracy_target, pred_target], feed_dict={handle2: handle_target_all, keep_prob_target:1.0, phase_target:False,
                                                                                              keep_prob_classifier:1.0, phase_classifier:False})
                
                print("target acc:{:.4f}".format(t_acc))
                last_target_acc_test = t_acc
                
                if max_target_acc_step2 < t_acc:
                    max_target_acc_step2 = t_acc 
                    max_target_pred_step2 = t_pred
                    
            if epoch+1 == epochs2 :
                sess.run(iterator_target_all.initializer, feed_dict={X_target: target_x, Y_target: target_y})
                target_pred_step2, target_latent = sess.run([pred_target, target_encoder], feed_dict={handle2: handle_target_all, keep_prob_target:1.0, phase_target:False,
                                                                                                          keep_prob_classifier:1.0, phase_classifier:False})
        
    if meta_step == "semi_supervised_learing":
        print("########## Semi-supervised learning ##########")
        for epoch in range(epochs3):
            sess.run(iterator_unlabel_all.initializer, feed_dict={X_latent_target: target_latent, Y_target: target_y})
            target_pred, cur_target_out = sess.run([pred_unlabel, pl_out], 
                                                         feed_dict = {handle2: handle_unlabel_all, keep_prob_target:1.0, phase_target:False, keep_prob_classifier:1.0, phase_classifier:False})
                        
            tmp = {}
            for i in range(n_classes) :
                tmp[i] = []
            for i in range(len(target_pred)) :
                for j in range(n_classes) :
                    if target_pred[i] == j :
                        tmp[j].append(1)
                    else :
                        tmp[j].append(0)
            pred_unlabel_df_tmp = pd.DataFrame(tmp).values	
            
            filteredpred = [[x_ul_sub, y_ul_sub, p] for x_ul_sub, y_ul_sub, p in zip(target_latent, cur_target_out, pred_unlabel_df_tmp) if max(y_ul_sub) > unlabel_probability]
            x_ul_filter = [row[0] for row in filteredpred]
            y_ul_filter = [row[1] for row in filteredpred]
            pred_unlabel_df = [row[2] for row in filteredpred]

            sess.run(iterator_label.initializer, feed_dict={X_latent: source_latent, Y: source_y})
            sess.run(iterator_unlabel.initializer, feed_dict={X_latent_target: x_ul_filter, Y_target: pred_unlabel_df})

            if epoch < t1 :
                a = 0.0
            elif (epoch >= t1) and (epoch < t2) :
                a = (float(epoch - t1)/(t2-t1))*af
            else :
                a = af
            avg_cost = 0
            avg_acc = 0
            while True: 
                try:
                    _, loss, tr_acc_, target_pred, cur_a, pl_aug__, pl__ = sess.run([data_op, data_cost, accuracy_data, pred_unlabel, alpha, pl_aug_out, pl_out], 
                                                                     feed_dict={handle: handle_label, handle2:handle_unlabel, keep_prob_classifier:keep_prob_classifier_rate, phase_classifier:True, alpha: a})
                    avg_cost += loss / batch_iter
                    avg_acc += tr_acc_ / batch_iter
                    
                except tf.errors.OutOfRangeError:
                    break
            if epoch % 10 == 0:
                
                print('Epoch:', '%04d' % (epoch + 1), 'cost = {:.3f}'.format(avg_cost), ' acc_source = {:.3f}'.format(avg_acc), end=" ")
                sess.run(iterator_target_all.initializer, feed_dict={X_target: target_x, Y_target: target_y})
                target_pred_step3, target_latent_step3, target_acc_step3 = sess.run([pred_target, target_encoder, accuracy_target], feed_dict={handle2: handle_target_all, keep_prob_target:1.0, phase_target:False,
                                                                                                          keep_prob_classifier:1.0, phase_classifier:False})
                print("target acc:{:.4f}".format(target_acc_step3))
                if max_target_acc_step3 < target_acc_step3:
                    max_target_acc_step3 = target_acc_step3 
                    max_target_pred_step3 = target_pred_step3
                    
#save predicted label
target_y_df.columns = range(n_classes)
real_target_label = pd.get_dummies(target_y_df).idxmax(1).values
prediction = pd.get_dummies(max_target_pred_step3).idxmax(1).values

pred_result = pd.concat([pd.DataFrame(real_target_label), pd.DataFrame(prediction)], axis=1)
pred_result.columns = ['Label', 'Prediction']
pred_result.to_csv(dir_save+'prediction.csv', index=False)

         
