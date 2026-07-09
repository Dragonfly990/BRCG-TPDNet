import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import h5py, math
import os
from resnest import resnest50 


class BasicConv2d0(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d0, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1./ math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d0(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d0(in_channel, out_channel, 1),
            BasicConv2d0(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d0(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d0(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d0(in_channel, out_channel, 1),
            BasicConv2d0(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d0(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d0(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d0(in_channel, out_channel, 1),
            BasicConv2d0(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d0(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d0(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d0(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d0(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    
class ResNeSt(nn.Module):
    def __init__(self):
        super(ResNeSt, self).__init__()
       
        self.resnet = resnest50(pretrained=False)
        self.resnet.load_state_dict(torch.load('./model/resnest50-528c19ca.pth'))
     
        
    def forward(self, x):
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)   
        x1 = self.resnet.layer1(x0)  
        x2 = self.resnet.layer2(x1)  
        x3 = self.resnet.layer3(x2)    
        x4 = self.resnet.layer4(x3)  
      
        return x0,x1,x2,x3,x4
    
class MLFM0(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MLFM0, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.CBR = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x,y):
        if x.size() == y.size():
            z = x*y
            z2 = torch.cat([z,x,y],dim=1)
            z3 = self.CBR(z2)+x
            z4 = self.relu(z3)
        else:
            x = self.up(x)
            z = x*y
            z2 = torch.cat([z,x,y],dim=1)
            z3 = self.CBR(z2)+x
            z4 = self.relu(z3)    
        return z4
    
class MLFM1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MLFM1, self).__init__()
        self.MLFM1=MLFM0(ch_in, ch_out)
        self.MLFM2=MLFM0(ch_in, ch_out)

    def forward(self, x,y,z):
        out1 = self.MLFM1(z,y)
        out2= self.MLFM2(out1,x)
        return out2


class GraphNet(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    def init(self, initcache):
        if not os.path.exists(initcache):
            print(initcache + ' not exist!!!\n')
        else:
            with h5py.File(initcache, mode='r') as h5:
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                self.init_params(clsts, traindescs)
                del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H*W
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) 
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        soft_assign = F.softmax(soft_assign, dim=1)

        return soft_assign

    def forward(self, x):
        B, C, H, W = x.size()
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        sigma = torch.sigmoid(self.sigma)
        soft_assign = self.gen_soft_assign(x, sigma) 
        #
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)

        nodes = F.normalize(nodes, p=2, dim=2)
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1) 

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign


class GraphConvNet(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous() 
        support = torch.matmul(x_t, self.weight) 

        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous() 
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        self.gcn1 = GraphConvNet(dim, dim)
        self.gcn2 = GraphConvNet(dim, dim)
        self.gcn3 = GraphConvNet(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        assert(loop == 1 or loop == 2 or loop == 3)
        self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x,adj):
        for gcn in self.gcns:
            x = gcn(x, adj) 
        x = self.relu(x)
        return x

    
class MLP(nn.Module):  
    def __init__(self, input_size, hidden_sizes, num_classes):  
        super(MLP, self).__init__()  
        
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])  
        
        layers = []  
        for i in range(1, len(hidden_sizes)):  
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))  
            layers.append(nn.ReLU())  
        self.hidden_layers = nn.Sequential(*layers)  
        
        self.layer_out = nn.Linear(hidden_sizes[-1], num_classes)  
  
    def forward(self, x):  
        x = F.relu(self.layer1(x))  
        x = self.hidden_layers(x)  
        x = self.layer_out(x)  
        return x 
    
class Adj(nn.Module):  
    def __init__(self, channels, input_size, hidden_sizes, num_classes):  
        super(Adj, self).__init__()  
          
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False) 
        
        self.fc3 = nn.Conv1d(num_classes, num_classes, kernel_size=1, bias=False)
        self.fc4 = nn.Conv1d(num_classes, num_classes, kernel_size=1, bias=False) 
        
        self.softmax = nn.Softmax()
        self.mlp = MLP(input_size, hidden_sizes, num_classes)
        
    
    def forward(self, x,y):  
        x0 = F.relu(self.fc1(self.mlp(x)))  
        y0 = F.relu(self.fc1(self.mlp(y))) 
        A = torch.bmm(x0.permute(0,2,1),y0)
        A = self.softmax(A)  
        x1= torch.bmm(self.mlp(x), F.relu(self.fc3(A)))
        y1= torch.bmm(self.mlp(y), F.relu(self.fc4(A)))
        x1_t = x1.permute(0, 2, 1).contiguous() 
        A_s = torch.matmul(x1_t, x1)  
        y1_t = y1.permute(0, 2, 1).contiguous() 
        A_e = torch.matmul(y1_t, y1)
        return A,A_s,A_e
    
class Sub_MGAI(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dim=64, num_clusters=8, dropout=0.1):
        super(Sub_MGAI, self).__init__()

        self.dim = dim

        self.rgb_proj0 = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)
        self.t_proj0 = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)
        self.adj = Adj(64,num_clusters,[100,50],num_clusters)
        
        self.gcn = CascadeGCNet(dim, loop=2)
        self.conv = nn.Sequential(BasicConv2d(dim, dim, BatchNorm, kernel_size=1, padding=0))

        self.pred = nn.Conv2d(self.dim, 1, kernel_size=1)

        self.up = nn.Upsample(scale_factor=8)
        
    def forward(self, rgb, t):

        rgb_graph, rgb_assign = self.rgb_proj0(rgb)
        t_graph, t_assign = self.t_proj0(t)
        
        A,A_s,A_e = self.adj(rgb_graph,t_graph)
        n_rgb_x = self.gcn(rgb_graph,A_s)
        n_rgb_x = n_rgb_x.bmm(rgb_assign)
        n_rgb_x = self.conv(n_rgb_x.unsqueeze(3)).squeeze(3)
        rgb_x = rgb + n_rgb_x.view(rgb.size()).contiguous()
        
        n_t_x = self.gcn(t_graph,A_e)
        n_t_x = n_t_x.bmm(t_assign)
        n_t_x = self.conv(n_t_x.unsqueeze(3)).squeeze(3)
        t_x = t + n_t_x.view(t.size()).contiguous()

        return   rgb_x,t_x
    

from torch.nn import Conv2d, Parameter, Softmax

def weight_init(module):
    for n, m in module.named_children():
        try:
            #print('initialize: '+n)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                weight_init(m)
            elif isinstance(m, (nn.ReLU,nn.PReLU, nn.Unfold, nn.Sigmoid, nn.AdaptiveAvgPool2d,nn.AvgPool2d, nn.Softmax,nn.Dropout2d)):
                pass
            else:
                m.initialize()
        except:
            pass
        
class GCM(nn.Module):
    def __init__(self, dim,in_dim):
        super(GCM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )


        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU()  #if batch=1 ，batchnorm will deliver bug
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        
        self.in_channels = in_channels =in_dim
        self.key_channels = key_channels =in_dim
        self.head_count = 1
        self.value_channels = value_channels =in_dim

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def initialize(self):
        weight_init(self)

    def forward(self, x):
        x = self.down_conv(x)

        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear', align_corners=True) #if batch=1 ，batchnorm will deliver bug

        y = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

        n, _, h, w = y.size()

        keys = self.keys(y).reshape((n, self.key_channels, h * w))
        queries = self.queries(y).reshape(n, self.key_channels, h * w)
        values = self.values(y).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        
        y = self.gamma * attention + y
        return y

class EDGModule(nn.Module):
    def __init__(self, channel,norm_layer = nn.BatchNorm2d):
        super(EDGModule, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d0(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d0(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d0(2 * channel, channel, 3, padding=1)

        self.conv_1 = BasicConv2d0(channel, channel, 3, padding=1)
        self.conv_2 = BasicConv2d0(channel, channel, 3, padding=1)
        self.conv_3 = BasicConv2d0(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d0(2*channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d0(2*channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d0(2 * channel, 2 * channel, 3, padding=1)

        self.conv5 = nn.Conv2d(2*channel, channel, 1)
       
        self.mlp = nn.Conv2d(channel,1,1)
      

    def forward(self, x1, x2, x3): 
        up_x1 = self.upsample(x1)
        conv_x2 = self.conv_1(x2)
        out12=up_x1*conv_x2
        cat_x2 = torch.cat((out12, x2), 1) 

        up_x1_1 = self.conv_upsample1(self.upsample(up_x1)) 
        up_x2 = self.conv_upsample2(self.upsample(x2)) 
        out123= up_x1_1*up_x2*x3 
      

        up_cat_x2 = self.conv_upsample3(self.upsample(cat_x2)) 
        cat_x4 = self.conv_concat4(torch.cat((up_cat_x2, out123), 1)) 
        x = self.conv5(cat_x4)
       
        return x     


    
class EEblock1(nn.Module):
    def __init__(self, channel):
        super(EEblock1, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.sconv13 = nn.Conv2d(channel,channel, kernel_size=(1,5), padding=(0,2))
        self.sconv31 = nn.Conv2d(channel,channel, kernel_size=(5,1), padding=(2,0))
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
             
    def forward(self, y, x):
       
        b, c, H, W = x.size()

        x1 = self.sconv13(x)
        x2 = self.sconv31(x)

        y1 = self.sconv13(y)
        y2 = self.sconv31(y)

        map_y13 = torch.sigmoid(self.avg_pool(y1).view(b,c,1,1))
        map_y31 = torch.sigmoid(self.avg_pool(y2).view(b,c,1,1))

        k = x1*map_y31 + x2*map_y13
        k = self.upsample(k)+y

        return k
    
class Net(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dim=64, num_clusters=8, dropout=0.1, pretrained=True, args=True):
        super(Net, self).__init__()
        self.layer_rgb = ResNeSt()
        self.rfb0 = RFB_modified(64, 64)
        self.rfb1 = RFB_modified(256, 64)
        self.rfb2 = RFB_modified(512, 64)
        self.rfb3 = RFB_modified(1024, 64)
        self.rfb4 = RFB_modified(2048, 64)
        
        self.agg = MLFM0(64*3,64)
        self.agg1 = MLFM1(64*3,64)
        
        self.gcn = Sub_MGAI()
        self.edge = EEblock1(64)
        
        self.gcm = GCM(64,64)
        self.dfs = EDGModule(64)
        
        self.final_conv = nn.Conv2d(64,1,1)
        
    def forward(self, x):
        x_0, x_1, x_2, x_3, x_4 = self.layer_rgb(x)
        x_4 = self.rfb4(x_4) 
        x_3 = self.rfb3(x_3)
        x_2 = self.rfb2(x_2)
        x_1 = self.rfb1(x_1)
        x_0 = self.rfb0(x_0)
        
        x_e = self.agg1(x_0,x_1,x_2)
        x_s = self.agg1(x_2,x_3,x_4)
        
        x_e = F.interpolate(x_e,size=[64,64], mode='bilinear') 
        x_s = F.interpolate(x_s,size=[32,32], mode='bilinear') 
        
        s_gcn, e_gcn = self.gcn(x_s,x_e)
      
        
        out_e = self.edge(e_gcn, s_gcn)
        out_1 = F.interpolate(self.final_conv(out_e),x.size()[2:], mode='bilinear') 
        
        x_g = self.gcm(x_4)
        out_2=F.interpolate(self.final_conv(x_g),x.size()[2:], mode='bilinear')
        
        out = self.dfs(x_g,x_s,out_e) 
        
        out = F.interpolate(self.final_conv(out),x.size()[2:], mode='bilinear') 
        
        return out_1,out_2,out
    

    
if __name__ == '__main__':
    net =  Net().cuda()
    
    img = torch.randn(2,3,256,256).cuda()
    x_0, x_1, x_2 = net(img)
    print(x_0.shape, x_1.shape,x_2.shape)
    

    
    