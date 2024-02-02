import torch
import numpy as np
import torch.nn as nn
from itertools import product
import torch.nn.functional as F
from torch.nn import Dropout, LeakyReLU, Linear, Sequential, Parameter

cuda = True if torch.cuda.is_available() else False


class FusedFuzzy(nn.Module):
    def __init__(self, input_vector_size, mid_vector_size, m, num_class):
        super(FusedFuzzy, self).__init__()
        self.input_vector_size = input_vector_size
        self.m = m
        self.fuzz_vector_size = 1

        for i in range(self.input_vector_size):
            self.fuzz_vector_size = self.fuzz_vector_size * m[i]

        self.mid_vector_size = mid_vector_size
        self.num_class = num_class

        self.linear_layer1 = nn.Linear(self.fuzz_vector_size, self.mid_vector_size)
        self.linear_layer2 = nn.Linear(self.mid_vector_size, self.fuzz_vector_size)
        self.linear_layer3 = nn.Linear(self.fuzz_vector_size, self.num_class)

        self.lerelu = LeakyReLU(0.2)
        self.dr = Dropout(0.5)

        # 隶属度函数的中心值和宽度值
        self.c = nn.Parameter(torch.Tensor(self.input_vector_size, 3))
        self.b = nn.Parameter(torch.Tensor(self.input_vector_size, 3))
        self.wei = nn.Parameter(torch.Tensor(self.fuzz_vector_size * self.input_vector_size, self.fuzz_vector_size))
        self.bais = nn.Parameter(torch.Tensor(1, ))
        nn.init.xavier_uniform_(self.c)
        nn.init.ones_(self.b)
        nn.init.xavier_uniform_(self.wei)
        nn.init.ones_(self.bais)

    def forward(self, input):
        # 模糊层 n为特征维度 i 某一个特征分量(属性) m[1]~m[n] 对应于 x[1]~x[n]进行模糊分级个数
        # 每个节点均代表一个模糊言语变量值，用于计算输入向量各分量属于各言语变量模糊集合的隶属函数
        u = torch.zeros((input.shape[0], input.shape[1], 3))
        for i in range(self.input_vector_size):
            uu = torch.zeros((input.shape[0], self.m[i]))
            for j in range(self.m[i]):
                uu[:, j] = torch.exp(-(input[:, i] - self.c[i][j]) ** 2 / self.b[i][j] ** 2)
            u[:, i, :] = uu

        # 实际输出 各结点均代表一条模糊规则 用来匹配模糊规则的前件 计算出每条规则的实用度 节点数为m[1]*m[2]*~*m[n-1]*m[n]
        # n个分组的隶属函数 不重复地从每个分组中取一个隶属函数组合在一起

        dd = torch.zeros((input.shape[0], self.fuzz_vector_size, input.shape[1]))
        for i in range(input.shape[0]):
            dd[i] = torch.FloatTensor(np.array(list(product(*(u[i].detach().numpy())))))

        if cuda:
            dd = dd.cuda()
        a = torch.mm(dd.view(input.shape[0], -1), self.wei) + self.bais

        # 归一化层 对每条规则的适用度进行归一化计算
        add_a = torch.sum(a, dim=1, keepdim=True)
        a_mean = a / add_a

        if cuda:
            a_mean = a_mean.cuda()

        # 输出层 实现清晰化的计算
        output_of_linear_layer1 = self.linear_layer1(a_mean)
        output_of_linear_layer2 = self.linear_layer2(output_of_linear_layer1)
        output_of_linear_layer3 = self.linear_layer3(a_mean + output_of_linear_layer2)
        output = self.lerelu(output_of_linear_layer3)
        output = self.dr(output)

        return output


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))

        return out


class Block(nn.Module):
    def __init__(self, i, o):
        super(Block, self).__init__()
        self.fc1 = Linear(i, o)
        self.fc2 = Linear(o, int(o / 2))
        self.fc3 = Linear(int(o / 2), o)
        self.lerelu = LeakyReLU(0.2)
        self.dr = Dropout(0.5)

    def forward(self, input):
        out1 = self.fc1(input)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out = self.lerelu(out1 + out3)
        out = self.dr(out)
        return out


class Attention(nn.Module):
    def __init__(self, i, o):
        super(Attention, self).__init__()
        self.fc1 = Linear(i, o)
        self.fc2 = Linear(i, o // 4)
        self.fc3 = Linear(i, o // 4)
        self.gamma = Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input):
        q = self.fc1(input)
        k = self.fc2(input)
        v = self.fc3(input)

        B, W = q.size()
        q = q.view(B, 1, 1 * W)  # query
        k = k.view(B, 1, 1 * W // 4)  # key
        v = v.view(B, 1, 1 * W // 4)  # value

        # attention weights
        w = F.softmax(torch.bmm(q.transpose(1, 2), k), -1)
        # attend and project
        o = torch.bmm(v, w.transpose(1, 2)).view(B, W)
        return self.gamma * o + input


class Classify(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(Classify, self).__init__()
        self.target_dim = target_dim
        dim = input_dim
        forward_fea_seq = []
        backward_fea_seq = []
        if input_dim >= 256:
            forward_fea_seq += [Attention(dim, dim)]
            forward_fea_seq += [Block(dim, 256)]
            forward_fea_seq += [Block(256, 128)]
            forward_fea_seq += [Block(128, 64)]
            backward_fea_seq += [Block(64, 128)]
            backward_fea_seq += [Block(128, 256)]
            backward_fea_seq += [Block(256, dim)]
        elif input_dim >= 128:
            forward_fea_seq += [Attention(dim, dim)]
            forward_fea_seq += [Block(dim, 128)]
            forward_fea_seq += [Block(128, 64)]
            backward_fea_seq += [Block(64, 128)]
            backward_fea_seq += [Block(128, dim)]
        else:
            forward_fea_seq += [Attention(dim, dim)]
            forward_fea_seq += [Block(dim, 64)]
            backward_fea_seq += [Block(64, dim)]

        self.forward_fea_seq = Sequential(*forward_fea_seq)
        self.backward_fea_seq = Sequential(*backward_fea_seq)

        dim = 64

        lab_seq = []
        fuzzy_seq = []
        if self.target_dim >= 16:
            lab_seq += [Linear(dim, 32)]
            fuzzy_seq += [Linear(32, 16)]
            fuzzy_seq += [Linear(16, 8)]
            fuzzy_seq += [Linear(8, 2)]
            dim = 32
            fuzzy_dim = 2
        elif self.target_dim >= 8:
            lab_seq += [Block(dim, 32)]
            lab_seq += [Linear(32, 16)]
            fuzzy_seq += [Linear(16, 8)]
            fuzzy_seq += [Linear(8, 2)]
            dim = 16
            fuzzy_dim = 2
        else:
            lab_seq += [Block(dim, 32)]
            lab_seq += [Block(32, 16)]
            lab_seq += [Linear(16, 8)]
            fuzzy_seq += [Linear(8, 2)]
            dim = 8
            fuzzy_dim = 2

        self.m = [3] * fuzzy_dim
        self.lab_seq = Sequential(*lab_seq)
        self.fuzzy_seq = Sequential(*fuzzy_seq)
        self.fuzzy = FusedFuzzy(fuzzy_dim, int((dim + fuzzy_dim) / 2), self.m, dim)
        self.nl = NormedLinear(dim, self.target_dim)

    def forward(self, input):
        emboutput = self.forward_fea_seq(input)
        reboutput = self.backward_fea_seq(emboutput)
        output1 = self.lab_seq(emboutput)
        output1_2 = self.fuzzy_seq(output1)
        output2 = self.fuzzy(output1_2)
        output = self.nl(output1 + output2)

        return emboutput, reboutput, output
