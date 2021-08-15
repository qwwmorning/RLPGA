import numpy as np
from scipy.stats import ortho_group
import tensorflow as tf
import utils
from scipy.sparse import vstack
import sys
import os
import json
from functools import partial
import amazon_wst_matrix_process as wmp


# Set hyper-parameters instruction -------------------------------------------------------------------------------------
'''
1.  paramless_net_struc: Changing the network which step don't have any operation to deal with parameters or 
    hyper-parameters. 
'''

# Set hyper-parameters -------------------------------------------------------------------------------------------------
hyper_param_dict = {}
hyper_param_dict['source_domain_abv'] = 'd'
hyper_param_dict['target_domain_abv'] = 'e'
hyper_param_dict['num_step'] = 5000
hyper_param_dict['paramless_net_struc'] = 'f'
hyper_param_dict['clf_param'] = 1
hyper_param_dict['dmi_param'] = 0
hyper_param_dict['ndmat_param_s'] = 1e-9
hyper_param_dict['ndmat_param_t'] = 1e-9
hyper_param_dict['ndmat_t'] = 1000
hyper_param_dict['ndmat_nn'] = 3
hyper_param_dict['wd_param'] = 1
hyper_param_dict['wd_w_norm_iterate'] = 1#0
hyper_param_dict['gp_param'] = 10
hyper_param_dict['total_lloss_method'] = 'l2_loss'
hyper_param_dict['noise_add_flag'] = False
hyper_param_dict['noise_amount'] = 0
hyper_param_dict['save_model'] = True

# Determine sys.argv settings ------------------------------------------------------------------------------------------
if len(sys.argv) > 1:
    try:
        # Cause sys.argv[0] is pyfile name
        param_ite = 1
        hyper_param_dict['source_domain_abv'] = str(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['target_domain_abv'] = str(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['num_step'] = int(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['paramless_net_struc'] = str(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['clf_param'] = float(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['dmi_param'] = float(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['ndmat_param_s'] = float(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['ndmat_param_t'] = float(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['ndmat_t'] = float(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['ndmat_nn'] = float(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['wd_param'] = float(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['wd_w_norm_iterate'] = int(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['gp_param'] = float(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['total_lloss_method'] = str(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['noise_add_flag'] = str(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['noise_amount'] = float(sys.argv[param_ite])
        param_ite += 1
        hyper_param_dict['save_model'] = str(sys.argv[param_ite])
        print('All \'nohup\' setting is DONE!')
    except IndexError:
        print('Part of \'nohup\' setting is DONE, but NOT all!')
else:
    print('NOT \'nohup\' process!')

# Set dependant zero
if hyper_param_dict['noise_add_flag'] == 'False':
    hyper_param_dict['noise_add_flag'] = False
elif hyper_param_dict['noise_add_flag'] == 'True':
    hyper_param_dict['noise_add_flag'] = True
else:
    hyper_param_dict['noise_add_flag'] = bool(hyper_param_dict['noise_add_flag'])
if not hyper_param_dict['noise_add_flag']:
    hyper_param_dict['noise_amount'] = 0
if hyper_param_dict['noise_amount'] == 0:
    hyper_param_dict['noise_amount'] = 0
if hyper_param_dict['save_model'] == 'False':
    hyper_param_dict['save_model'] = False
elif hyper_param_dict['save_model'] == 'True':
    hyper_param_dict['save_model'] = True
else:
    hyper_param_dict['save_model'] = bool(hyper_param_dict['save_model'])
if not hyper_param_dict['dmi_param'] == 0:
    hyper_param_dict['clf_param'] = 0

# Get data -------------------------------------------------------------------------------------------------------------
data_folder = './data/'
transfer_direct = str(hyper_param_dict['source_domain_abv']) + ' ---> ' + str(hyper_param_dict['target_domain_abv'])
if hyper_param_dict['source_domain_abv'] == 'd':
    source_name = 'dvd'
elif hyper_param_dict['source_domain_abv'] == 'e':
    source_name = 'electronics'
elif hyper_param_dict['source_domain_abv'] == 'b':
    source_name = 'books'
elif hyper_param_dict['source_domain_abv'] == 'k':
    source_name = 'kitchen'
else:
    print('ERROR: Your choice of \'source_domain_abv\' is NOT correct!')
    sys.exit(0)
if hyper_param_dict['target_domain_abv'] == 'd':
    target_name = 'dvd'
elif hyper_param_dict['target_domain_abv'] == 'e':
    target_name = 'electronics'
elif hyper_param_dict['target_domain_abv'] == 'b':
    target_name = 'books'
elif hyper_param_dict['target_domain_abv'] == 'k':
    target_name = 'kitchen'
else:
    print('ERROR: Your choice of \'target_domain_abv\' is NOT correct!')
    sys.exit(0)
xs, ys, xt, yt, xt_test, yt_test = utils.load_amazon(source_name, target_name, data_folder, verbose=True)

wst_data_dir = './wst_data/'
ws = wmp.load_matrix(source_name, wst_data_dir, 'train', hyper_param_dict['ndmat_t'])
wt = wmp.load_matrix(target_name, wst_data_dir, 'train', hyper_param_dict['ndmat_t'])
wt_test = wmp.load_matrix(target_name, wst_data_dir, 'test', hyper_param_dict['ndmat_t'])

wst_neg_data_dir = './wst_neg_data/'
ws_neg = wmp.load_neg_matrix(source_name, wst_neg_data_dir, 'train', hyper_param_dict['ndmat_t'])
wt_neg = wmp.load_neg_matrix(target_name, wst_neg_data_dir, 'train', hyper_param_dict['ndmat_t'])
wt_test_neg = wmp.load_neg_matrix(target_name, wst_neg_data_dir, 'test', hyper_param_dict['ndmat_t'])

# Set flags
qr_update_flag = True

# Set dirs
if hyper_param_dict['save_model']:
    save_path = 'trained_model/net_' + str(hyper_param_dict['paramless_net_struc']) + '_model/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path += str(source_name) + '_' + str(target_name) + '/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path += str(source_name) + '_' + str(target_name)
log_dir = './log/' + str(sys.argv[0].split('amazon_')[-1][:-3]) + '/'
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
else:
    for i in os.listdir(log_dir):
        file_data = log_dir + "\\" + i
        if os.path.isfile(file_data):
            os.remove(file_data)

# Set parameters
clf_dmi_train_mod = 10
if hyper_param_dict['clf_param'] == 0:
    eval_best_term = 'dmi_acc_xt'
else:
    eval_best_term = 'clf_acc_xt'
matrix_determinant_err = 1e-6
regularizer_weight = 0.5

# hyper-parameter for wdgrl
lr_wd_D = 1e-4
D_train_num = 5

batch_size = 64
l2_param = 1e-4
lr = 1e-4
num_class = 2
n_input = xs.shape[1]
n_hidden = [500]

dmi_clf_joint_flag = True
dmi_clf_joint_dmi_hp = 1e-5

# tf.set_random_seed(0)
# np.random.seed(0)

# Tools ----------------------------------------------------------------------------------------------------------------
def convert_to_type(obj, type='list'):
    if type == 'list:':
        try:
            obj_numpy = (obj.eval(session=sess)).tolist()
            obj_list = obj_numpy.tolist()
        except:
            obj_list = obj.tolist()
        obj_return = obj_list
    else:
        try:
            obj_numpy = (obj.eval(session=sess)).tolist()
        except:
            obj_numpy = obj
        obj_return = obj_numpy
    return obj_return

def generate_orthogonal_matrix(dim):
    # Generate
    # generated_matrix = tf.random_uniform(shape=shape, minval=0., maxval=1.)
    generated_matrix = ortho_group.rvs(dim=dim)
    generated_matrix = generated_matrix.tolist()
    generated_matrix = tf.convert_to_tensor(generated_matrix)
    generated_matrix = tf.cast(generated_matrix, dtype=tf.float32)
    return generated_matrix

def isset(v):
    try:
        type(eval(v))
    except:
        return 0
    else:
        return 1

def l_n_loss_func(choice, tf_var):
    if choice == 'l2_loss':
        l_loss = tf.nn.l2_loss(tf_var)
    elif choice == 'l1_loss':
        l_loss = tf.contrib.layers.l1_regularizer(regularizer_weight)(tf_var)
    elif choice == 'l2_1_loss':
        l_loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.multiply(tf_var, tf_var), axis=1)))
    else:
        print('ERROR: Your choice of\'l_n_loss_func\' is NOT correct!')
        sys.exit(2)
    return l_loss

def tf_confusion_metrics(matrics_name, predict, real):
    # Set default inputs
    predictions = tf.argmax(predict, 1)
    actuals = tf.argmax(real, 1)
    # Set default matrix
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
    # Calc four main matrics
    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )
    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )
    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )
    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )
    tp = tp_op
    tn = tn_op
    fp = fp_op
    fn = fn_op
    # Calc tf_confusion_metrics
    tpr = tf.to_float(tp) / (tf.to_float(tp) + tf.to_float(fn))
    fpr = tf.to_float(fp) / (tf.to_float(fp) + tf.to_float(tn))
    fnr = tf.to_float(fn) / (tf.to_float(tp) + tf.to_float(fn))
    accuracy = (tf.to_float(tp) + tf.to_float(tn)) / (
                tf.to_float(tp) + tf.to_float(fp) + tf.to_float(fn) + tf.to_float(tn))
    recall = tpr
    precision = tf.to_float(tp) / (tf.to_float(tp) + tf.to_float(fp))
    f1_score = (2 * (precision * recall)) / (precision + recall)
    # Save in dict
    return_data = {'tpr': tpr, 'fpr': fpr, 'fnr': fnr, 'accuracy': accuracy, 'recall': recall, 'precision': precision,
                   'f1_score': f1_score}
    return return_data[matrics_name]

def spectral_norm(spectral_norm_w, iteration=1):
    spectral_norm_w_shape = spectral_norm_w.shape.as_list()
    spectral_norm_w = tf.reshape(spectral_norm_w, [-1, spectral_norm_w_shape[-1]])
    spectral_norm_u = tf.Variable(tf.random_normal(shape=[1, spectral_norm_w_shape[-1]]), name='spectral_norm_u',
                                  trainable=False)
    spectral_norm_u_hat = spectral_norm_u
    spectral_norm_v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        spectral_norm_v_ = tf.matmul(spectral_norm_u_hat, tf.transpose(spectral_norm_w))
        spectral_norm_v_hat = tf.nn.l2_normalize(spectral_norm_v_)
        spectral_norm_u_ = tf.matmul(spectral_norm_v_hat, spectral_norm_w)
        spectral_norm_u_hat = tf.nn.l2_normalize(spectral_norm_u_)
    spectral_norm_u_hat = tf.stop_gradient(spectral_norm_u_hat)
    spectral_norm_v_hat = tf.stop_gradient(spectral_norm_v_hat)
    spectral_norm_sigma = tf.matmul(tf.matmul(spectral_norm_v_hat, spectral_norm_w), tf.transpose(spectral_norm_u_hat))
    with tf.control_dependencies([spectral_norm_u.assign(spectral_norm_u_hat)]):
        spectral_norm_w_norm = spectral_norm_w / spectral_norm_sigma
        spectral_norm_w_norm = tf.reshape(spectral_norm_w_norm, spectral_norm_w_shape)
    return spectral_norm_w_norm

def w_matrix_refine(w_batch):
    refined_w_matrix = np.zeros(shape=[int(batch_size / 2), int(batch_size / 2)])
    for nd_i in range(int(batch_size / 2)):
        sub_x_wd_list = []
        for nd_j in range(nd_i + 1, int(batch_size / 2)):
            # Get X vector distance
            real_x_wd = w_batch[nd_i][int(w_batch[nd_j][-1])]
            if not sub_x_wd_list:
                sub_x_wd_list.append([real_x_wd, nd_i, nd_j])
            else:
                insert_flag = False
                for i in range(len(sub_x_wd_list)):
                    if sub_x_wd_list[i][0] > real_x_wd:
                        insert_flag = True
                        sub_x_wd_list.insert(i, [real_x_wd, nd_i, nd_j])
                        break
                if not insert_flag:
                    sub_x_wd_list.append([real_x_wd, nd_i, nd_j])
                while len(sub_x_wd_list) > hyper_param_dict['ndmat_nn']:
                    del sub_x_wd_list[-1]
        # Envalue matrix
        for sub_x_wd in sub_x_wd_list:
            refined_w_matrix[sub_x_wd[1]][sub_x_wd[2]] = sub_x_wd[0]
            refined_w_matrix[sub_x_wd[2]][sub_x_wd[1]] = sub_x_wd[0]
    for nd_k in range(len(refined_w_matrix)):
        refined_w_matrix[nd_k][nd_k] = -(sum(refined_w_matrix[nd_k]))
    return refined_w_matrix

# Init input data and variables ----------------------------------------------------------------------------------------
with tf.name_scope('input'):
    X = tf.sparse_placeholder(dtype=tf.float32)
    y_true = tf.placeholder(dtype=tf.int32)
    y_test_true = tf.placeholder(dtype=tf.int32)
    W = tf.placeholder(dtype=tf.float32)
    W_neg = tf.placeholder(dtype=tf.float32)
    train_flag = tf.placeholder(dtype=tf.bool)
    y_true_one_hot = tf.one_hot(y_true, num_class)
    y_test_true_one_hot = tf.one_hot(y_test_true, num_class)

with tf.name_scope('generator'):
    h1 = utils.fc_layer(X, n_input, n_hidden[0], layer_name='hidden1', input_type='sparse')

with tf.name_scope('slice_data'):
    h1_s = tf.cond(train_flag, lambda: tf.slice(h1, [0, 0], [int(batch_size / 2), -1]), lambda: h1)
    h1_t = tf.cond(train_flag, lambda: tf.slice(h1, [int(batch_size / 2), 0], [int(batch_size / 2), -1]), lambda: h1)
    ys_true = tf.cond(train_flag, lambda: tf.slice(y_true_one_hot, [0, 0], [int(batch_size / 2), -1]),
                      lambda: y_true_one_hot)
    ys_test_true = tf.cond(train_flag, lambda: tf.slice(y_test_true_one_hot, [0, 0], [int(batch_size / 2), -1]),
                           lambda: y_test_true_one_hot)

# Calc clf_loss --------------------------------------------------------------------------------------------------------
if not hyper_param_dict['clf_param'] == 0:
    with tf.name_scope('classifier'):
        W_clf = tf.Variable(tf.truncated_normal([n_hidden[-1], num_class], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)),
                            name='clf_weight')
        b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')
        pred_logit = tf.matmul(h1_s, W_clf) + b_clf
        pred_softmax = tf.nn.softmax(pred_logit)
        # Train COULD be noised labels
        clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=ys_true))
        # Test MUST be real labels
        clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_test_true, 1), tf.argmax(pred_softmax, 1)), tf.float32))
        clf_recall = tf_confusion_metrics('recall', pred_softmax, ys_test_true)
        clf_f1_score = tf_confusion_metrics('f1_score', pred_softmax, ys_test_true)
        clf_loss_sum = tf.summary.scalar('clf_loss', clf_loss)
        clf_acc_sum = tf.summary.scalar('clf_acc', clf_acc)

# Calc dmi_loss --------------------------------------------------------------------------------------------------------
if not hyper_param_dict['dmi_param'] == 0:
    with tf.name_scope('dmi'):
        # Set weight matrix
        W_dmi_clf = tf.Variable(tf.truncated_normal([n_hidden[-1], num_class], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)),
                            name='dmi_clf_weight')
        # Set bias matrix
        b_dmi_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='dmi_clf_bias')
        # . multiply sliced dataset h1_s and weight, then add bias
        dmi_pred_logit = tf.matmul(h1_s, W_dmi_clf) + b_dmi_clf
        outputs = tf.nn.softmax(dmi_pred_logit)
        # Train COULD be noised labels
        dmi_loss = -1.0 * tf.log(tf.abs(tf.matrix_determinant(
            tf.matmul(tf.transpose(ys_true), outputs) + (tf.eye(num_class) * matrix_determinant_err))) + 0.001)
        if dmi_clf_joint_flag:
            dmi_hp = dmi_clf_joint_dmi_hp / (dmi_clf_joint_dmi_hp + 1)
            clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=ys_true))
            dmi_loss = dmi_hp * dmi_loss + (1 - dmi_hp) * clf_loss
        # Test MUST be real labels
        dmi_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_test_true, 1), tf.argmax(outputs, 1)), tf.float32))
        dmi_recall = tf_confusion_metrics('recall', outputs, ys_test_true)
        dmi_f1_score = tf_confusion_metrics('f1_score', outputs, ys_test_true)
        # Remove vars
        del outputs, dmi_pred_logit, b_dmi_clf, W_dmi_clf

# Draw wd_net ----------------------------------------------------------------------------------------------------------
if not hyper_param_dict['wd_param'] == 0:
    alpha = tf.random_uniform(shape=[int(batch_size / 2), 1], minval=0., maxval=1.)
    differences = h1_s - h1_t
    interpolates = h1_t + (alpha*differences)
    h1_whole = tf.concat([h1, interpolates], 0)
    with tf.name_scope('critic'):
        with tf.name_scope('critic_h1'):
            wd_weight_h1 = tf.Variable(
                tf.truncated_normal([n_hidden[-1], 100], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)), name='weight')
            wd_bias_h1 = tf.Variable(tf.constant(0.1, shape=[100]), name='bias')
            critic_h1 = tf.nn.relu(tf.matmul(h1_whole, wd_weight_h1) + wd_bias_h1)
        with tf.name_scope('critic_h2'):
            wd_weight_h2 = tf.Variable(tf.truncated_normal([100, 1], stddev=1. / tf.sqrt(100 / 2.)), name='weight')
            wd_bias_h2 = tf.Variable(tf.constant(0.1, shape=[1]), name='bias')
            critic_out = tf.identity(tf.matmul(critic_h1, wd_weight_h2) + wd_bias_h2)

# Draw ndmat_net -------------------------------------------------------------------------------------------------------
if not hyper_param_dict['ndmat_param_s'] == 0 and not hyper_param_dict['ndmat_param_t'] == 0:
    with tf.name_scope('ndmatnet'):
        # Start process
        # Calc POSITIVE graph ------------------------------------------------------------------------------------------
        # Processing source domain ---------------------
        # Calc measure matrix
        # Sparse matrix is instructed as Indices, Values, Shape!!!! Only points in Indices list have values!!!
        source_x_wd_matrix = tf.cond(train_flag, lambda: tf.slice(W, [0, 0], [int(batch_size / 2), -1]), lambda: W)

        # Processing target domain ---------------------
        # Calc measure matrix
        # Sparse matrix is instructed as Indices, Values, Shape!!!! Only points in Indices list have values!!!
        target_x_wd_matrix = tf.cond(train_flag,
                                     lambda: tf.slice(W, [int(batch_size / 2), 0], [int(batch_size / 2), -1]),
                                     lambda: W)

        # Processing source target domain match loss ---
        ndmat_loss_s_pos = tf.trace(tf.matmul(tf.matmul(tf.transpose(h1_s), source_x_wd_matrix), h1_s))
        ndmat_loss_t_pos = tf.trace(tf.matmul(tf.matmul(tf.transpose(h1_t), target_x_wd_matrix), h1_t))

        # Calc NEGATIVE graph ------------------------------------------------------------------------------------------
        # Processing source domain ---------------------
        # Calc measure matrix
        # Sparse matrix is instructed as Indices, Values, Shape!!!! Only points in Indices list have values!!!
        source_x_wd_neg_matrix = tf.cond(train_flag, lambda: tf.slice(W_neg, [0, 0], [int(batch_size / 2), -1]),
                                     lambda: W_neg)

        # Processing target domain ---------------------
        # Calc measure matrix
        # Sparse matrix is instructed as Indices, Values, Shape!!!! Only points in Indices list have values!!!
        target_x_wd_neg_matrix = tf.cond(train_flag,
                                     lambda: tf.slice(W_neg, [int(batch_size / 2), 0], [int(batch_size / 2), -1]),
                                     lambda: W_neg)

        # Processing source target domain match loss ---
        ndmat_loss_s_neg = tf.trace(tf.matmul(tf.matmul(tf.transpose(h1_s), source_x_wd_neg_matrix), h1_s))
        ndmat_loss_t_neg = tf.trace(tf.matmul(tf.matmul(tf.transpose(h1_t), target_x_wd_neg_matrix), h1_t))

        # Calc nn loss -------------------------------------------------------------------------------------------------
        ndmat_loss_s = tf.log(1 + tf.exp(ndmat_loss_s_pos - ndmat_loss_s_neg))
        ndmat_loss_t = tf.log(1 + tf.exp(ndmat_loss_t_pos - ndmat_loss_t_neg))

# ----------------------------------------------------------------------------------------------------------------------
# **********************************************************************************************************************
# Name variables of name spaces ----------------------------------------------------------------------------------------
theta_G = [v for v in tf.global_variables() if 'generator' in v.name]
theta_C = [v for v in tf.global_variables() if 'classifier' in v.name]
theta_M = [v for v in tf.global_variables() if 'dmi' in v.name]
theta_D = [v for v in tf.global_variables() if 'critic' in v.name]
# ----------------------------------------------------------------------------------------------------------------------
# **********************************************************************************************************************
# ----------------------------------------------------------------------------------------------------------------------

# Calc wd_loss ---------------------------------------------------------------------------------------------------------
if not hyper_param_dict['wd_param'] == 0:
    critic_s = tf.cond(train_flag, lambda: tf.slice(critic_out, [0, 0], [int(batch_size / 2), -1]), lambda: critic_out)
    critic_t = tf.cond(train_flag, lambda: tf.slice(critic_out, [int(batch_size / 2), 0], [int(batch_size / 2), -1]),
                       lambda: critic_out)
    wd_loss = tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t)
    tf.summary.scalar('wd_loss', wd_loss)
    if hyper_param_dict['gp_param']:
        gradients = tf.gradients(critic_out, [h1_whole])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        wd_d_op = tf.train.AdamOptimizer(lr_wd_D).minimize(-wd_loss + hyper_param_dict['gp_param'] * gradient_penalty,
                                                           var_list=theta_D)
    else:
        wd_d_op = tf.train.AdamOptimizer(lr_wd_D).minimize(-wd_loss, var_list=theta_D)

# Calc total loss function ---------------------------------------------------------------------------------------------
all_variables = tf.trainable_variables()
if not l2_param == 0:
    l2_loss = l2_param * tf.add_n(
        [l_n_loss_func(hyper_param_dict['total_lloss_method'], v) for v in all_variables if 'bias' not in v.name])
else:
    l2_loss = 0
# Add as total_loss
total_loss = l2_loss
train_var_list = theta_G
if not hyper_param_dict['wd_param'] == 0:
    total_loss += hyper_param_dict['wd_param'] * wd_loss
if not hyper_param_dict['clf_param'] == 0:
    total_loss += hyper_param_dict['clf_param'] * clf_loss
    train_var_list += theta_C
if not hyper_param_dict['dmi_param'] == 0:
    total_loss += hyper_param_dict['dmi_param'] * dmi_loss
    train_var_list += theta_M
if not hyper_param_dict['ndmat_param_s'] == 0 and not hyper_param_dict['ndmat_param_t'] == 0:
    total_loss += hyper_param_dict['ndmat_param_s'] * ndmat_loss_s
    total_loss += hyper_param_dict['ndmat_param_t'] * ndmat_loss_t

train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, var_list=train_var_list)

merged = tf.summary.merge_all()
if not hyper_param_dict['clf_param'] == 0:
    test_merged = tf.summary.merge([clf_loss_sum, clf_acc_sum])
saver = tf.train.Saver(max_to_keep=100)

# Main running ---------------------------------------------------------------------------------------------------------
# Save trained data func
def save_trained_stats_data(eval_step_log_dict, hyper_param_dict):
    # Get saving dir
    filename = './evaluation_stats/'
    if not os.path.isdir(filename):
        os.mkdir(filename)
    filename = filename + 'eval_stats'
    for filename_param_dict in hyper_param_dict:
        filename += '_' + str(hyper_param_dict[filename_param_dict])
    filename += '.jsonl'
    # Generate json_dump_data
    json_dump_data = hyper_param_dict.copy()
    json_dump_data.update(eval_step_log_dict.copy())
    # Save data
    # try:
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        f.write(json.dumps(json_dump_data))
    # except:
    #     print('ERROR: When processing \'save_trained_stats_data\'!')
    #     sys.exit(1)
    print('\nProcess: \'save_trained_stats_data\' is DONE!\n')
    return json_dump_data

# Save step trained data func
def save_eval_step_log_data(eval_step_log_list, hyper_param_dict):
    # Get saving dir
    filename = './evaluation_stats/'
    if not os.path.isdir(filename):
        os.mkdir(filename)
    filename = filename + 'step_log'
    for filename_param_dict in hyper_param_dict:
        filename += '_' + str(hyper_param_dict[filename_param_dict])
    filename += '.jsonl'
    # Save data
    # try:
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        f.write(json.dumps(eval_step_log_list))
    # except:
    #     print('ERROR: When processing \'save_trained_stats_data\'!')
    #     sys.exit(1)
    print('\nProcess: \'save_eval_step_log_data\' is DONE!\n')
    return True

# Show evaluation and end
def show_end_and_evaluation_stats(json_dump_data_list):
    # Start
    print('\n\nEvaluation Matrix -------------------------------------------------------------------------------------')

    # Content
    for json_dump_data in json_dump_data_list:
        print_content = ''
        for dict_name in json_dump_data:
            print_content += str(dict_name) + ': ' + str(json_dump_data[dict_name]) + '\t'
        print_content = print_content[:-2]
        print(print_content)

    # End
    print('-------------------------------------------------------------------------------------------------------\n\n')
    print('All processes of Amazon_Dataset evaluation are DONE!')

    return True

# TF Main run
with tf.Session() as sess:
    # Initializing variables
    sess.run(tf.global_variables_initializer())

    # Set default dir
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    # Noise determine and addition
    # Create real list
    try:
        ys_real = ys.copy()
    except:
        ys_real = ys
    try:
        yt_real = yt.copy()
    except:
        yt_real = yt
    try:
        yt_test_real = yt_test.copy()
    except:
        yt_test_real = yt_test
    if hyper_param_dict['noise_add_flag']:
        # Uniform noise
        conf_matrix = [[1, 0], [hyper_param_dict['noise_amount'], 1 - hyper_param_dict['noise_amount']]]
        # Add into label
        for idx, target_label in enumerate(ys):
            target_label_new = int(np.random.choice(num_class, 1, p=np.array(conf_matrix[target_label])))
            ys[idx] = target_label_new
        for idx, target_label in enumerate(yt):
            target_label_new = int(np.random.choice(num_class, 1, p=np.array(conf_matrix[target_label])))
            yt[idx] = target_label_new
        for idx, target_label in enumerate(yt_test):
            target_label_new = int(np.random.choice(num_class, 1, p=np.array(conf_matrix[target_label])))
            yt_test[idx] = target_label_new

    # Initializing input datasets
    S_batches = utils.batch_generator([xs, ys, ys_real, ws, ws_neg], int(batch_size / 2), shuffle=True)
    T_batches = utils.batch_generator([xt, yt, yt_real, wt, wt_neg], int(batch_size / 2), shuffle=True)

    # Train one-time
    eval_step_log_list = []
    best_eval_step_log_dict = {}
    # Iterate steps and train datasets
    for i in range(hyper_param_dict['num_step']):
        xs_batch_csr, ys_batch, ys_real_batch, ws_batch, wd_neg_batch = next(S_batches)
        xt_batch_csr, yt_batch, yt_real_batch, wt_batch, wt_neg_batch = next(T_batches)
        batch_csr = vstack([xs_batch_csr, xt_batch_csr])
        xb = utils.csr_2_sparse_tensor_tuple(batch_csr)
        yb = np.hstack([ys_batch, yt_batch])
        yb_real = np.hstack([ys_real_batch, yt_real_batch])

        # Process wb and make it as diag input matrix
        ws_batch = w_matrix_refine(ws_batch)
        wt_batch = w_matrix_refine(wt_batch)
        wb = np.vstack([ws_batch, wt_batch])

        ws_neg_batch = w_matrix_refine(ws_neg_batch)
        wt_neg_batch = w_matrix_refine(wt_neg_batch)
        wb_neg = np.vstack([ws_neg_batch, wt_neg_batch])

        for _ in range(D_train_num):
            # Op wd
            sess.run([wd_d_op], feed_dict={X: xb, train_flag: True})

            # Op wd_weight
            if not hyper_param_dict['wd_w_norm_iterate'] == 0:
                wd_weight_h1 = spectral_norm(wd_weight_h1, iteration=hyper_param_dict['wd_w_norm_iterate'])
                wd_weight_h2 = spectral_norm(wd_weight_h2, iteration=hyper_param_dict['wd_w_norm_iterate'])

        _, train_summary, l_wd = sess.run([train_op, merged, wd_loss],
                                          feed_dict={X: xb, y_true: yb, y_test_true: yb_real, W: wb, W_neg: wb_neg,
                                                     train_flag: True})
        train_writer.add_summary(train_summary, global_step=i)

        if i % clf_dmi_train_mod == 0:
            eval_step_log_dict = {}
            whole_xs_stt = utils.csr_2_sparse_tensor_tuple(xs)
            whole_xt_stt = utils.csr_2_sparse_tensor_tuple(xt_test)
            print('Step: ', str(i))
            print('Transfer direction: ' + transfer_direct)
            print('Wasserstein distance: %f' % l_wd)
            eval_step_log_dict['step'] = int(i)
            eval_step_log_dict['wd'] = float(l_wd)
            if not hyper_param_dict['clf_param'] == 0:
                clf_acc_xs, clf_rec_xs, clf_f1_xs, c_loss_xs = sess.run([clf_acc, clf_recall, clf_f1_score, clf_loss],
                                                                        feed_dict={X: whole_xs_stt, y_true: ys,
                                                                                   y_test_true: ys_real,
                                                                                   train_flag: False})
                test_summary, clf_acc_xt, clf_rec_xt, clf_f1_xt, c_loss_xt = sess.run(
                    [test_merged, clf_acc, clf_recall, clf_f1_score, clf_loss],
                    feed_dict={X: whole_xt_stt, y_true: yt_test, y_test_true: yt_test_real, train_flag: False})
                print('CLF - Source classifier loss: %f, Target classifier loss: %f' % (c_loss_xs, c_loss_xt))
                print('CLF - Source label accuracy: %f, Target label accuracy: %f' % (clf_acc_xs, clf_acc_xt))
                print('CLF - Source label recall: %f, Target label recall: %f' % (clf_rec_xs, clf_rec_xt))
                print('CLF - Source label f1 score: %f, Target label f1 score: %f' % (clf_f1_xs, clf_f1_xt))
                eval_step_log_dict['c_loss_xs'] = float(c_loss_xs)
                eval_step_log_dict['c_loss_xt'] = float(c_loss_xt)
                eval_step_log_dict['clf_acc_xs'] = float(clf_acc_xs)
                eval_step_log_dict['clf_acc_xt'] = float(clf_acc_xt)
                eval_step_log_dict['clf_rec_xs'] = float(clf_rec_xs)
                eval_step_log_dict['clf_f1_xs'] = float(clf_f1_xs)
                eval_step_log_dict['clf_rec_xt'] = float(clf_rec_xt)
                eval_step_log_dict['clf_f1_xt'] = float(clf_f1_xt)
                test_writer.add_summary(test_summary, global_step=i)
            if not hyper_param_dict['dmi_param'] == 0:
                # DMI part train and test
                dmi_acc_xs, dmi_rec_xs, dmi_f1_xs, d_loss_xs = sess.run([dmi_acc, dmi_recall, dmi_f1_score, dmi_loss],
                                                                        feed_dict={X: whole_xs_stt, y_true: ys,
                                                                                   y_test_true: ys_real,
                                                                                   train_flag: False})
                dmi_acc_xt, dmi_rec_xt, dmi_f1_xt, d_loss_xt = sess.run([dmi_acc, dmi_recall, dmi_f1_score, dmi_loss],
                                                                        feed_dict={X: whole_xt_stt, y_true: yt_test,
                                                                                   y_test_true: yt_test_real,
                                                                                   train_flag: False})
                print('DMI - Source dmi loss: %f, Target dmi loss: %f' % (d_loss_xs, d_loss_xt))
                print('DMI - Source label accuracy: %f, Target label accuracy: %f' % (dmi_acc_xs, dmi_acc_xt))
                print('DMI - Source label recall: %f, Target label recall: %f' % (dmi_rec_xs, dmi_rec_xt))
                print('DMI - Source label f1 score: %f, Target label f1 score: %f' % (dmi_f1_xs, dmi_f1_xt))
                eval_step_log_dict['d_loss_xs'] = float(d_loss_xs)
                eval_step_log_dict['d_loss_xt'] = float(d_loss_xt)
                eval_step_log_dict['dmi_acc_xs'] = float(dmi_acc_xs)
                eval_step_log_dict['dmi_acc_xt'] = float(dmi_acc_xt)
                eval_step_log_dict['dmi_rec_xs'] = float(dmi_rec_xs)
                eval_step_log_dict['dmi_f1_xs'] = float(dmi_f1_xs)
                eval_step_log_dict['dmi_rec_xt'] = float(dmi_rec_xt)
                eval_step_log_dict['dmi_f1_xt'] = float(dmi_f1_xt)

            eval_step_log_list.append(eval_step_log_dict)
            if best_eval_step_log_dict:
                if float(eval_step_log_dict[eval_best_term]) > float(best_eval_step_log_dict[eval_best_term]):
                    best_eval_step_log_dict = eval_step_log_dict.copy()
            else:
                best_eval_step_log_dict = eval_step_log_dict.copy()

            if hyper_param_dict['save_model']:
                saver.save(sess, save_path, global_step=i)

    # Save data
    json_dump_data = save_trained_stats_data(best_eval_step_log_dict, hyper_param_dict)

    save_eval_step_log_data(eval_step_log_list, hyper_param_dict)
