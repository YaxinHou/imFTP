import os
import torch
import argparse
import numpy as np
import pandas as pd

from model.model import Classify
from utils.transformer import DataTransformer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

cuda = True if torch.cuda.is_available() else False


def getnewf1(y_test, pre_label, min_class):
    labels = np.unique(y_test)
    p_class_macro, r_class_macro, f_class_macro, support_macro = \
        precision_recall_fscore_support(y_test, pre_label, labels=labels, average='macro', )
    p_class_micro, r_class_micro, f_class_micro, support_micro = \
        precision_recall_fscore_support(y_test, pre_label, labels=labels, average='micro', )
    p_class_weighted, r_class_weighted, f_class_weighted, support_weighted = \
        precision_recall_fscore_support(y_test, pre_label, labels=labels, average='weighted', )
    p_class_none, r_class_none, f_class_none, support_micro_none = \
        precision_recall_fscore_support(y_test, pre_label, labels=labels, average=None)
    pAR = 0
    for i in min_class:
        pAR = (pAR + r_class_none[i]) / len(min_class)

    acc = accuracy_score(y_test, pre_label)
    g_mean = 1
    for i in r_class_none:
        g_mean = g_mean * i
    g_mean = g_mean ** (1 / len(r_class_none))
    g_mean = round(g_mean, 4)

    return acc, f_class_macro, f_class_micro, f_class_weighted, g_mean, r_class_macro, pAR


def ags_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Epochs', default=501, type=int, help='Number of S_training epochs')
    parser.add_argument('--dataset', type=str, help='dataset setting', default='mfcc')
    parser.add_argument('--train_dir', type=str, dest='train_dir', help='the path of train data',
                        default='./data/spilted_data')
    parser.add_argument('--model_dir', default='./model', help='Number of training epochs')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = ags_parse()
    # 读取训练数据

    all_data_filename = ['abalone', 'balance', 'car', 'clave', 'dermatology', 'ecoli', 'flare', 'glass',
                         'mfcc', 'new-thyroid', 'nursery', 'pageblocks', 'satimage', 'shuttle', 'thyroid']

    all_min_class_id = [[0, 1, 13, 14, 15, 16, 17, 18, 19], [1], [2, 3], [0],
                        [0, 2, 3, 4, 5], [1, 2, 3, 4], [4, 5], [0, 3, 4, 5],
                        [3], [0, 2], [1], [0, 2, 3], [2, 4, 5],
                        [0, 2], [1, 2]]

    f = all_data_filename.index(args.dataset)
    data_filename = all_data_filename[f] + '.xlsx'
    min_class_id = all_min_class_id[f]

    data_name = data_filename.split('.')[0]

    model_name = data_filename.split('.')[0]

    filepath = os.path.join(args.train_dir, data_filename)

    # 模型保存路径
    args.model_dir = './model' + os.sep + model_name

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # train_data
    tr_df = pd.read_excel(filepath, index_col=None, header=None, sheet_name='Sheet1')
    tr_data = np.array(tr_df)
    feature = tr_data[:, 0:-1]
    label = tr_data[:, -1]

    # test_data
    te_df = pd.read_excel(filepath, index_col=None, header=None, sheet_name='Sheet2')
    te_data = np.array(te_df)
    x = te_data[:, 0:-1]
    y = te_data[:, -1]

    # all_data
    all_df = pd.concat([tr_df, te_df], axis=0)
    all_data = np.array(all_df)
    all_x = all_data[:, 0:-1]
    all_y = all_data[:, -1]

    label_nums = len(np.unique(all_y))  # 类别数量

    # 统计每一类的数目及比重
    per_class_count = []
    for c in range(len(np.unique(label))):
        num = np.sum(np.array(label == c))
        per_class_count.append(num)

    trans = DataTransformer()

    # 高斯采样
    trans.fit(feature)
    feature = trans.transform(feature)
    x = trans.transform(x)

    rows_org, cols_org = feature.shape

    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    S = Classify(cols_org, label_nums)
    S_filename = '{}/S_{}.pth'.format(args.model_dir, args.dataset)

    if cuda:
        S.cuda()
        S.load_state_dict(torch.load(S_filename, map_location=torch.device('GPU')))
        _, _, pred = S(x.cuda())
    else:
        S.load_state_dict(torch.load(S_filename, map_location=torch.device('cpu')))

        _, _, pred = S(x)

    _, pred = torch.max(torch.softmax(pred, dim=1), 1)
    Acc, F_class_macro, F_class_micro, F_class_weighted, G_mean, R_class, PAR = \
        getnewf1(y, pred.cpu(), min_class_id)
