import numpy as np
from sklearn.datasets import load_svmlight_files
import pickle as pkl
import os
import utils


def avoid_eval(data):
    try:
        data = eval(data)
        return data
    except TypeError:
        return data


def load_amazon_x_input(domain_name, usage):
    data_folder = './data/'
    domain_file = data_folder + domain_name + '_' + str(usage) + '.svmlight'
    domain_data, label_data = load_svmlight_files([domain_file])
    return domain_data, label_data


def load_matrix(domain_name, data_folder, usage, ndmat_t):
    if data_folder is None:
        data_folder = './wst_data/'
    domain_wst_file = data_folder + domain_name + '_' + usage + '_' + str(ndmat_t) + '.pkl'
    print('Loading \'' + str(domain_name) + '\' wst matrix...')
    print('From: ' + str(domain_wst_file))
    if os.path.isfile(domain_wst_file):
        with open(domain_wst_file, 'rb') as f:
            domain_w = pkl.load(f)
    else:
        generate_w_matrix(domain_name, usage, domain_wst_file, ndmat_t)
        print('Process \'generate_w_matrix\' is FINISHED!')
        with open(domain_wst_file, 'rb') as f:
            domain_w = pkl.load(f)
    return domain_w


def generate_w_matrix(domain_name, usage, save_data_folder, ndmat_t):
    print('Not exists!')
    print('Generating wst matrix...')
    domain_data, label_data = load_amazon_x_input(domain_name, usage)
    try:
        domain_data = domain_data.todense()
    except:
        print('domain_data is already dense matrix!')
    # Start process
    # Calc X measure matrix
    domain_w_matrix = np.zeros(shape=[int(len(domain_data)), int(len(domain_data)) + 1])
    outer_loop_len = len(domain_data)
    for nd_i in range(int(len(domain_data))):
        inner_loop_len = outer_loop_len - nd_i - 1
        for nd_j in range(nd_i + 1, int(len(domain_data))):
            print(
                'Outer loop: ' + str(nd_i + 1) + '/' + str(outer_loop_len) + '\tInner loop: ' + str(
                    nd_j + 1) + '/' + str(inner_loop_len))
            # Calc X vector distance
            x_wd = -(np.exp(-(np.square(np.linalg.norm(domain_data[nd_j] - domain_data[nd_i])) / float(ndmat_t))))
            # Envalue matrix
            domain_w_matrix[nd_i][nd_j] = float(x_wd)
            domain_w_matrix[nd_j][nd_i] = float(x_wd)
        domain_w_matrix[nd_i][-1] = int(nd_i)
    # Save data
    with open(save_data_folder, 'wb') as f:
        pkl.dump(domain_w_matrix, f)


def load_neg_matrix(domain_name, data_folder, usage, ndmat_t):
    if data_folder is None:
        data_folder = './wst_neg_data/'
    domain_wst_file = data_folder + domain_name + '_' + usage + '_' + str(ndmat_t) + '.pkl'
    print('Loading \'' + str(domain_name) + '\' wst neg matrix...')
    print('From: ' + str(domain_wst_file))
    if os.path.isfile(domain_wst_file):
        with open(domain_wst_file, 'rb') as f:
            domain_w = pkl.load(f)
    else:
        generate_neg_w_matrix(domain_name, usage, domain_wst_file, ndmat_t)
        print('Process \'generate_neg_w_matrix\' is FINISHED!')
        with open(domain_wst_file, 'rb') as f:
            domain_w = pkl.load(f)
    return domain_w


def generate_neg_w_matrix(domain_name, usage, save_data_folder, ndmat_t):
    print('Not exists!')
    print('Generating wst matrix...')
    domain_data, label_data = load_amazon_x_input(domain_name, usage)
    try:
        domain_data = domain_data.todense()
    except:
        print('domain_data is already dense matrix!')
    # Start process
    # Get neg edge matrix
    B, count = utils.constructLines(domain_data)
    # Calc X measure matrix
    domain_w_matrix = np.zeros(shape=[int(len(domain_data)), int(len(domain_data)) + 1])
    outer_loop_len = len(domain_data)
    for nd_i in range(int(len(domain_data))):
        inner_loop_len = outer_loop_len - nd_i - 1
        for nd_j in range(nd_i + 1, int(len(domain_data))):
            print(
                'Outer loop: ' + str(nd_i + 1) + '/' + str(outer_loop_len) + '\tInner loop: ' + str(
                    nd_j + 1) + '/' + str(inner_loop_len))
            # Calc X vector distance
            if B[nd_i] == B[nd_j]:
                x_wd = 0
            else:
                x_wd = -(np.exp(-(np.square(np.linalg.norm(domain_data[nd_j] - domain_data[nd_i])) / float(ndmat_t))))
            # Envalue matrix
            domain_w_matrix[nd_i][nd_j] = float(x_wd)
            domain_w_matrix[nd_j][nd_i] = float(x_wd)
        domain_w_matrix[nd_i][-1] = int(nd_i)
    # Save data
    with open(save_data_folder, 'wb') as f:
        pkl.dump(domain_w_matrix, f)
