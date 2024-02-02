import os
import time

import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from math import cos, pi
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import normal

from utils.log import get_log
from model.model import Classify
from utils.transformer import DataTransformer
from utils.dataset import MyDataset, DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

cuda = True if torch.cuda.is_available() else False


def adjusted_rand_score(features, targets):
    label_unique = targets.unique()
    num_classes = len(label_unique)
    # 根据特征向量计算每个类的类中心
    center_features = torch.zeros(label_unique.size(0), features.size(1))

    for i in range(label_unique.size(0)):
        label = label_unique[i]
        same_class_features = features[targets == label]
        center_features[i] = same_class_features.mean(dim=0)
    if cuda:
        center_features = center_features.cuda()
    # 计算特征向量与类中心的距离---→预测出来的簇
    distmat1 = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(features.size(0), num_classes)
    distmat2 = torch.pow(center_features, 2).sum(dim=1, keepdim=True).expand(num_classes, features.size(0)).t()
    distmat = distmat1 + distmat2
    distmat.addmm_(mat1=features, mat2=center_features.t(), beta=1, alpha=-2)

    per_max, _ = torch.max(distmat, dim=1, keepdim=True)
    per_max = per_max.expand(features.size(0), num_classes)
    logit = per_max - distmat

    other_target = torch.FloatTensor(
        [((label_unique == per_label).nonzero(as_tuple=True)[0]).item() for per_label in targets])
    if cuda:
        other_target = other_target.cuda()
    loss = F.cross_entropy(logit, other_target.long())

    return loss


def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p.detach()) ** gamma * input_values

    return loss.mean()


class GCLLoss(nn.Module):
    def __init__(self, cls_num_list, m=0.5, s=30, train_cls=False, noise_mul=1., gamma=0.):
        super(GCLLoss, self).__init__()
        cls_list = torch.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max() - m_list
        self.m_list = m_list
        if cuda:
            self.m_list = self.m_list.cuda()
        assert s > 0
        self.m = m
        self.s = s
        self.simpler = normal.Normal(0, 1 / 3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma

    def forward(self, cosine, target):
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        noise = self.simpler.sample(cosine.shape).clamp(-1, 1)
        if cuda:
            noise = noise.cuda()

        cosine = cosine - self.noise_mul * noise.abs() / self.m_list.max() * self.m_list
        output = torch.where(index, cosine - self.m, cosine)
        if self.train_cls:
            return focal_loss(F.cross_entropy(self.s * output, target, reduction='none'), self.gamma)
        else:
            return F.cross_entropy(self.s * output, target)


class MultiCenterLoss(nn.Module):
    def __init__(self, num_classes, n_center=3, dim=1):
        super(MultiCenterLoss, self).__init__()
        self.dim = dim
        self.n_center = n_center
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        self.lin = nn.Linear(64, self.dim)
        self.bias = nn.Parameter(torch.ones(self.num_classes, self.n_center))
        self.centers = nn.Parameter(torch.zeros(self.num_classes, self.n_center, 64))
        self.linear = nn.Parameter(torch.randn(self.num_classes, self.dim, self.n_center))

    def forward(self, x, labels):
        # x→(200, 64) labels→(200, )
        maps_detach = x.detach()
        maps_detach_p = self.lin(maps_detach)  # lin(64,1) (200, 64)→(200, 1)
        target_select = labels.unsqueeze(1)
        # linear = self.linear[labels.long(), :, :]
        linear = self.linear[target_select.long(), :, :].view(-1, self.dim, self.n_center)  # (200, 1, 3)
        # bias = self.bias[labels.long(), :]
        bias = self.bias[target_select.long(), :].view(-1, self.n_center)  # (200, 3)
        # (200, 1, 1) * (200, 1, 3) + (200, 3)
        maps_detach_fc = torch.bmm(maps_detach_p.unsqueeze(1), linear).view(-1, self.n_center) + bias
        gamma = self.softmax(maps_detach_fc)  # (200, 3)
        # (200, 3, 64)
        centers_ = self.centers[target_select.long(), :, :].view(-1, self.n_center, 64)
        # (200, 3)→(200, 1 ,3) (200, 1 ,3)*( (200, 3, 64) - (200, 3, 64) )**2→(200, 1 ,3)*(200, 3, 64)→(200, 1, 64)
        loss = torch.sum(torch.bmm(gamma.unsqueeze(1),
                                   torch.pow((x.unsqueeze(1).expand(-1, centers_.size()[1], -1) - centers_), 2).view(
                                       x.size(0), self.n_center, -1))) / (x.size(0))

        return loss


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

    log_file = './result/' + str(data_filename.split('.')[0]) + '/'
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    train_result_logger = get_log('result/' + str(data_filename.split('.')[0]) + '/train_log.txt',
                                  'main-result' + str(data_filename.split('.')[0]))

    test_logger = get_log(
        './result/' + str(data_filename.split('.')[0]) + '/re_log-' + str(data_filename.split('.')[0]) + '.txt',
        'Main-result-' + str(data_filename.split('.')[0]))
    start_time = time.time()
    data_name = data_filename.split('.')[0]

    train_result_logger.info('数据集：' + data_name)

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

    data_of_transform = np.concatenate([feature, label.reshape(-1, 1)], axis=1)


    # 权重初始化
    def weight_init(m):
        # 是否为线性层
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            nn.init.constant_(m.bias, 0)
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight, mean=1, std=0.02)
            nn.init.constant_(m.bias, 0)


    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    # 损失函数
    CEL = nn.CrossEntropyLoss()
    MulCL = MultiCenterLoss(num_classes=label_nums)
    GCLN = GCLLoss(cls_num_list=per_class_count, m=0., s=30, noise_mul=0.5)
    GCLS = GCLLoss(cls_num_list=per_class_count, m=0.1, s=20, train_cls=True, noise_mul=0.5, gamma=1.)

    S = Classify(cols_org, label_nums)

    S.apply(weight_init)
    if cuda:
        S.cuda()
        CEL.cuda()
        MulCL.cuda()
        GCLN.cuda()
        GCLS.cuda()

    optimizer_S = optim.Adam(list(S.parameters()) + list(MulCL.parameters()), lr=2e-4, betas=(0.5, 0.9))

    for i in range(1, args.Epochs):
        alpha = 0.5 * (cos((i / 500) * pi) + 1)
        current_data = MyDataset(data_of_transform, i, 361)
        current_dataloader = DataLoader(current_data, batch_size=200, shuffle=True)
        for j, (classic_meta, tanh_meta) in enumerate(current_dataloader):
            if cuda:
                feature_a, feature_b = tanh_meta["sample_feature"].cuda(), classic_meta["sample_feature"].cuda()
                label_a, label_b = tanh_meta["sample_label"].cuda(), classic_meta["sample_label"].cuda()
            else:
                feature_a, feature_b = tanh_meta["sample_feature"], classic_meta["sample_feature"]
                label_a, label_b = tanh_meta["sample_label"], classic_meta["sample_label"]
            if i < 361:
                med_feature_1, med_feature_2, med_feature_3 = S(feature_a)
                loss_1 = GCLN(med_feature_3, label_a.long())
                loss_4 = MulCL(med_feature_1, label_a)
                loss_2 = torch.norm((med_feature_2 - feature_a), p=2, dim=1).mean()
                loss_5 = adjusted_rand_score(med_feature_1, label_a.long())
            else:
                med_feature_1, med_feature_2, med_feature_3 = S(feature_b)
                loss_1 = GCLS(med_feature_3, label_b.long())
                loss_4 = MulCL(med_feature_1, label_b)
                loss_2 = torch.norm((med_feature_2 - feature_b), p=2, dim=1).mean()
                loss_5 = adjusted_rand_score(med_feature_1, label_b.long())

            total_loss = loss_1 + alpha * (loss_4 + loss_2 + loss_5)
            optimizer_S.zero_grad()
            total_loss.backward()
            optimizer_S.step()
            print('epoch:', i, 'step:', j, 'loss1:', loss_1.item(),
                  'loss2:', loss_2.item(),
                  'loss4:', loss_4.item(), 'loss5:', loss_5.item())
        if i % 50 == 0:
            train_result_logger.info('epoch:' + str(i) + ' loss1:' + str(loss_1.item()) +
                                     ' loss2:' + str(loss_2.item()) +
                                     ' loss4:' + str(loss_4.item()) + ' loss5:' + str(loss_5.item()))
        if cuda:
            _, _, pred = S(x.cuda())
        else:
            _, _, pred = S(x)
        _, pred = torch.max(torch.softmax(pred, dim=1), 1)
        Acc, F_class_macro, F_class_micro, F_class_weighted, G_mean, R_class, PAR = \
            getnewf1(y, pred.cpu(), min_class_id)

        end_time = time.time()
        if i > 360:
            torch.save(S.state_dict(), '{}/S-gen_{}.pth'.format(args.model_dir, i))

            test_logger.info(str(i) + '  ' + str(Acc) + '  ' + str(F_class_macro) + '  ' + str(F_class_micro)
                             + '  ' + str(F_class_weighted) + '  ' + str(G_mean) + '  ' + str(R_class) + '  '
                             + str(PAR) + '  ' + str(end_time - start_time))

    test_logger.info(' ')
    end_time = time.time()
    totaltime = end_time - start_time
    train_result_logger.info(str(totaltime) + ' (' + str(totaltime / 60) + ')')
    train_result_logger.info('—————————————————————————————分割线—————————————————————————————')
    train_result_logger.info(' ')
