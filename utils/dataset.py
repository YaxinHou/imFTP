import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MyDataset(Dataset):
    def __init__(self, data, current_epoch, epochs):
        super(MyDataset, self).__init__()
        self.data = data
        self.current_epoch = current_epoch - 1
        self.epochs = epochs - 2
        self.cls_num = len(np.unique(data[:, -1]))
        self.features = data[:, 0:-1]
        self.labels = data[:, -1]
        self.tanh_class_weight, self.tanh_sum_weight = self.tanh_get_weight(self.get_annotations(), self.cls_num)
        self.class_dict = self._get_class_dict()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: (feature, label) where label is index of the target class.
        """
        tanh_meta = dict()
        classic_meta = dict()

        # classic
        classic_feature, classic_label = self.features[index], self.labels[index]
        classic_meta['sample_feature'] = torch.tensor(classic_feature, dtype=torch.float32)
        classic_meta['sample_label'] = torch.tensor(classic_label, dtype=torch.float32)

        # TANH
        tanh_sample_class = self.tanh_sample_class_index_by_weight()
        tanh_sample_indexes = self.class_dict[tanh_sample_class]
        tanh_sample_index = random.choice(tanh_sample_indexes)

        tanh_sample_feature, tanh_sample_label = \
            self.features[tanh_sample_index], self.labels[tanh_sample_index]
        tanh_meta['sample_feature'] = torch.tensor(tanh_sample_feature, dtype=torch.float32)
        tanh_meta['sample_label'] = torch.tensor(tanh_sample_label, dtype=torch.float32)

        return classic_meta, tanh_meta

    def tanh_sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.tanh_sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.tanh_class_weight[i]
            if rand_number <= now_sum:
                return i

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)

        return class_dict

    def tanh_get_weight(self, annotations, num_classes):
        beta_a = -30
        beta_b = 30
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        delta = np.log(np.array(num_list).astype(np.float32))
        delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))
        beta = beta_a + delta * (beta_b - beta_a)
        effective_num_a = sigmoid(beta / 9) * 2
        effective_num_b = sigmoid(- beta / 9) * 2
        class_weight = effective_num_a + (effective_num_b - effective_num_a) * (self.current_epoch / self.epochs)
        sum_weight = sum(class_weight)

        return class_weight, sum_weight

    def get_num_classes(self):

        return self.cls_num

    def get_annotations(self):
        annos = []
        for target in self.labels:
            annos.append({'category_id': int(target)})

        return annos
