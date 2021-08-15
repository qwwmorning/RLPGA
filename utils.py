import numpy as np
import tensorflow as tf
from sklearn.datasets import load_svmlight_files
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_amazon(source_name, target_name, data_folder=None, verbose=False):
    if data_folder is None:
        data_folder = './data/'
    source_file = data_folder + source_name + '_train.svmlight'
    target_file = data_folder + target_name + '_train.svmlight'
    test_file = data_folder + target_name + '_test.svmlight'
    if verbose:
        print('source file:', source_file)
        print('target file:', target_file)
        print('test file:  ', test_file)

    xs, ys, xt, yt, xt_test, yt_test = load_svmlight_files([source_file, target_file, test_file])
    ys, yt, yt_test = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, yt_test))

    return xs, ys, xt, yt, xt_test, yt_test


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]


def csr_2_sparse_tensor_tuple(csr_matrix):
    coo_matrix = csr_matrix.tocoo()
    indices = np.transpose(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = coo_matrix.data
    shape = csr_matrix.shape
    return indices, values, shape


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def group_id_2_label(group_ids, num_class):
    labels = np.zeros([len(group_ids), num_class])
    for i in range(len(group_ids)):
        labels[i, group_ids[i]] = 1
    return labels


def label_2_group_id(labels, num_class):
    tmp = np.arange(num_class)
    group_ids = labels.dot(tmp)
    return group_ids


def spectral_norm(w, iteration=1):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])
   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)
       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)
   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)
   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)
   return w_norm


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, input_type='dense'):
    with tf.name_scope(layer_name):
        weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1. / tf.sqrt(input_dim / 2.)), name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')
        if input_type == 'sparse':
            activations = act(tf.sparse_tensor_dense_matmul(input_tensor, weight) + bias)
        else:
            activations = act(tf.matmul(input_tensor, weight) + bias)
        return activations


def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def plot_embedding(X, y, source_num, file_name):
    pp = PdfPages(file_name)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(5, 5))

    xs = X[:source_num]
    xt = X[source_num:]
    ys = y[:source_num]
    yt = y[source_num:]

    index_neg_s = np.where(ys == 0)[0]
    index_pos_s = np.where(ys == 1)[0]
    xs_pos = xs[index_pos_s]
    xs_neg = xs[index_neg_s]
    plt.scatter(xs_pos[:, 0], xs_pos[:, 1], color='red', alpha=0.7, s=5, label='xs positive samples')
    plt.scatter(xs_neg[:, 0], xs_neg[:, 1], color='blue', alpha=0.7, s=5, label='xs negative samples')

    index_neg_t = np.where(yt == 0)[0]
    index_pos_t = np.where(yt == 1)[0]
    xt_pos = xt[index_pos_t]
    xt_neg = xt[index_neg_t]
    plt.scatter(xt_pos[:, 0], xt_pos[:, 1], color='purple', alpha=0.7, s=5, label='xt positive samples')
    plt.scatter(xt_neg[:, 0], xt_neg[:, 1], color='green', alpha=0.7, s=5, label='xt negative samples')

    pp.savefig()
    pp.close()


def matdistance_tf(x, X, lenX):
    x_tile = tf.tile(x, [lenX, 1])
    Xb = tf.square(tf.subtract(x_tile, X))
    Xc = tf.sqrt(tf.reduce_sum(Xb, 1))
    with tf.Session() as sess:
        A = Xc.eval().tolist
    return A


def constructLines_tf(X):
    print('Process: constructing edges...')
    lenX = int(X.get_shape().as_list()[0])
    B = np.zeros(lenX)
    tFlag = 1 # same label in nns
    cFlag = 0 # decide overlap
    for i in range(lenX):
        print('Subprocess: ' + str(i + 1) + '/' + str(lenX))
        if not B[i] == 0: # arranged
            continue
        temp = X[i, :]
        A = matdistance_tf(temp, X, lenX)
        A[i] = float("inf") # set the self-index is inf
        index = A.index(min(A))
        B[i] = tFlag
        while not float(B[index]) == float(tFlag):
            if not B[index] == 0: # this nn is arranged
                for j in range(lenX):
                    if B[j] == tFlag:
                        B[j] = B[index] # change the label of tFlag to B[index]
                cFlag = 1
                break
            B[index] = tFlag
            tem = X[index, :]
            A = matdistance_tf(tem, X, lenX)
            A[index] = float("inf") # set the self-index is inf
            index = A.index(min(A))
        if cFlag == 0:
            tFlag = tFlag + 1
        else:
            cFlag = 0
    count = tFlag - 1
    return B, count


def matdistance(x, X, lenX):
    x_tile = np.tile(x, [lenX, 1])
    Xb = np.square(np.subtract(x_tile, X))
    Xc = np.sqrt(Xb.sum(axis=1))
    A = Xc.tolist
    return A


def constructLines(X):
    print('Process: constructing edges...')
    lenX = int(np.size(X, 0))
    B = np.zeros(lenX)
    tFlag = 1 # same label in nns
    cFlag = 0 # decide overlap
    for i in range(lenX):
        print('Subprocess: ' + str(i + 1) + '/' + str(lenX))
        if not B[i] == 0: # arranged
            continue
        temp = X[i]
        A = matdistance(temp, X, lenX)
        A[i] = float("inf") # set the self-index is inf
        index = A.index(min(A))
        B[i] = tFlag
        while not float(B[index]) == float(tFlag):
            if not B[index] == 0: # this nn is arranged
                for j in range(lenX):
                    if B[j] == tFlag:
                        B[j] = B[index] # change the label of tFlag to B[index]
                cFlag = 1
                break
            B[index] = tFlag
            tem = X[index]
            A = matdistance(tem, X, lenX)
            A[index] = float("inf") # set the self-index is inf
            index = A.index(min(A))
        if cFlag == 0:
            tFlag = tFlag + 1
        else:
            cFlag = 0
    count = tFlag - 1
    return B, count
