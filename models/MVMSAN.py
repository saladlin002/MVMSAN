import torch
import torch.nn as nn
from torch.autograd import Variable
from .Model import Model
import torchvision.models as models

import timm

from models.Softpool import SoftPooling2D
from models.MBConv import MBConvBlock

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1, -1, -1),
                                                    ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class SVMSAN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='resnet18'):
        super(SVMSAN, self).__init__(name)

        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        # self.classnames = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table',
        # 'toilet']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name

        self.use_resnest = cnn_name.startswith('resnest')
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_densenet = cnn_name.startswith('densenet')

        # self.mean = Variable(torch.FloatTensor([0.0142, 0.0142, 0.0142]), requires_grad=False).cuda()
        # self.std = Variable(torch.FloatTensor([0.0818, 0.0818, 0.0818]), requires_grad=False).cuda()

        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        if self.use_resnest:
            if cnn_name == 'resnest14d':
                self.net_1 = timm.create_model('resnest14d', pretrained=True, num_classes=0, global_pool='')
                self.net_1.head = nn.Identity()
                self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                self.net_2 = nn.Linear(in_features=2048, out_features=40)
            elif cnn_name == 'resnest26d':
                self.net_1 = timm.create_model('resnest26d', pretrained=True, num_classes=0, global_pool='')
                self.net_1.head = nn.Identity()
                self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                self.net_2 = nn.Linear(in_features=2048, out_features=10)
            elif cnn_name == 'resnest50d':
                self.net_1 = timm.create_model('resnest50d', pretrained=True, num_classes=0, global_pool='')
                self.net_1.head = nn.Identity()
                self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                self.net_2 = nn.Linear(in_features=2048, out_features=10)
        elif self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 10)
        elif self.use_densenet:
            if self.cnn_name =='densenet121':
                self.net = models.densenet121(pretrained=self.pretraining)
                self.net.classifier = nn.Linear(1024, 10)

    def forward(self, x):
        if self.use_resnest:
            return self.net(x)
        elif self.use_resnet:
            return self.net(x)
        elif self.use_densenet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))

class MVMSAN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='resnet18', num_views=20):
        super(MVMSAN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        # self.classnames = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table',
        # 'toilet']

        self.nclasses = nclasses
        self.num_views = num_views

        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnest = cnn_name.startswith('resnest')
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_densenet = cnn_name.startswith('densenet')

        self.pool_mode = 'soft_att_MBC'

        self.MBConv_lay = MBConvBlock(ksize=3, input_filters=1, output_filters=1, image_size=20)
        self.relu = nn.ReLU()
        dropout = 0.
        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(2048, 2048)

        if self.use_resnest:
            if cnn_name == 'resnest14d':
                self.net_1 = timm.create_model('resnest14d', pretrained=True, num_classes=0, global_pool='')
                self.net_1.head = nn.Identity()
                self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                # self.net_2 = nn.Linear(in_features=2048, out_features=40)
                self.net_2 = nn.Sequential(nn.Conv2d(2048, 40, kernel_size=1))
            elif cnn_name == 'resnest26d':
                self.net_1 = timm.create_model('resnest26d', pretrained=True, num_classes=0, global_pool='')
                self.net_1.head = nn.Identity()
                self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                self.net_2 = nn.Linear(in_features=2048, out_features=40)
                # self.net_2 = nn.Sequential(nn.Conv2d(2048, 10, kernel_size=1))
            elif cnn_name == 'resnest50d':
                self.net_1 = timm.create_model('resnest50d', pretrained=True, num_classes=0, global_pool='')
                self.net_1.head = nn.Identity()
                self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                # self.net_2 = nn.Linear(in_features=2048, out_features=10)
                self.net_2 = nn.Sequential(nn.Conv2d(2048, 10, kernel_size=1))
        elif self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.net_3 = nn.Sequential(*list(model.net.children())[:4])
            self.net_2 = nn.Sequential(nn.Conv2d(2048, 10, kernel_size=1))
        elif self.use_densenet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # self.net_2 = nn.Sequential(nn.Linear(1024, 10))
            self.net_2 = nn.Sequential(nn.Conv2d(2048, 10, kernel_size=1))

        if self.pool_mode == 'soft_att_MBC':
            self.conv1 = nn.Conv2d(1, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(1, 1, 1, bias=False)
            self.conv3 = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, x):
        y = self.net_1(x)
        y = self.pool(y)

        if self.pool_mode == 'soft_att_MBC':
            y = y.view((int(x.shape[0] / self.num_views), 1, y.shape[-3], self.num_views))

            q = SoftPooling2D(1,1)(y)

            # q = self.conv1(y)
            k = self.conv2(y)
            v = self.conv3(y)

            s = torch.matmul(torch.transpose(q, 2, 3), k)

            s = self.MBConv_lay(s)

            beta = torch.nn.functional.softmax(s)

            o = torch.matmul(v, beta)

            gamma = torch.autograd.Variable(torch.FloatTensor([[1.]]), requires_grad=True).cuda()
            y = y + gamma * o

            y = torch.max(y, 3)[0].view(y.shape[0], -1)

            y = y.unsqueeze(-1).unsqueeze(-1)

            y = self.net_2(y)
            y = y.squeeze(-1).squeeze(-1)

        elif self.pool_mode == 'max':
            y = y.view((int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
            y = torch.max(y, 1)[0].view(y.shape[0], -1)

            y = y.unsqueeze(-1).unsqueeze(-1)
            y = self.net_2(y)
            y = y.squeeze(-1).squeeze(-1)

        return y
