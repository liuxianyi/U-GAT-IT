import paddle.fluid as fluid 
from paddle.fluid import dygraph as nn 
import numpy as np
from nn import *
class ResnetGenerator(nn.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [ReflectionPad2d(3),
                    nn.Conv2D(input_nc, ngf, filter_size=7,  stride=1, padding=0, bias_attr=False),
                    InstanceNorm(),
                    ReLU()]
       

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [ReflectionPad2d(1),
                          nn.Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(),
                          ReLU()]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
           
            DownBlock += [
                ResnetBlock(ngf * mult, use_bias=False)
            ]
        
        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1,bias_attr=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1,bias_attr=False)
        self.conv1x1 = nn.Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1,bias_attr=True)
        self.relu = ReLU()

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, act='relu',bias_attr=False),
                  ReLU(),
                  nn.Linear(ngf * mult, ngf * mult, act='relu',bias_attr=False),
                  ReLU()]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult,act='relu',bias_attr=False),
                  ReLU(),
                  nn.Linear(ngf * mult, ngf * mult,act='relu',bias_attr=False),
                  ReLU()
                  ]
                  
        self.gamma = nn.Linear(ngf * mult, ngf * mult,bias_attr=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult,bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2+=[
                Upsample(2),
                ReflectionPad2d(1),
                nn.Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0,bias_attr=False),
                ILN(int(ngf * mult / 2)), 
                ReLU()
            ]
               
       
        UpBlock2+=[
            ReflectionPad2d(3),
            nn.Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0,bias_attr=False),
            Tanh()
        ]
            

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)
    def forward(self, input):
        x = self.DownBlock(input)

        #print(x.shape)  #[2, 256, 63, 63]
        gap = fluid.layers.adaptive_pool2d(input=x, pool_size=(1,1), pool_type='avg')
        #print(gap.shape)  #[2, 256, 1, 1]
        #print(paddle.reshape(x=gap, shape=[x.shape[0], -1]).shape) #[2, 256]
        gap_logit = self.gap_fc(fluid.layers.reshape(x=gap, shape=[x.shape[0], -1]))
        #print(gap_logit.shape) #[2, 1]
        gap_weight = self.gap_fc.parameters()
        #print(gap_weight[0].shape) #[256,1]
        #print(paddle.unsqueeze(input=gap_weight[0], axes=[0,2]).shape) #[1, 256, 1, 1]
        gap = x * fluid.layers.unsqueeze(gap_weight[0], [0,2])
        
        gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=(1,1), pool_type='avg')
        gmp_logit = self.gmp_fc(fluid.layers.reshape(x=gmp, shape=[x.shape[0], -1]))
        gmp_weight = self.gmp_fc.parameters()
        gmp = x * fluid.layers.unsqueeze(gmp_weight[0], [0,2])

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat([gap, gmp], axis=1)
        x = self.relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(input=x, dim=1, keep_dim=True)
        #print(x.shape) #[2, 256, 64, 64]
        if self.light:
            x_ = fluid.layers.adaptive_pool2d(input=x, pool_size=(1,1), pool_type='avg')
            x_ = self.FC(fluid.layers.reshape(x=x_, shape=[x_.shape[0],-1]))
        else:
            #print(x.shape)
            x_ = self.FC(fluid.layers.reshape(x=x, shape=[x.shape[0],-1]))
        gamma, beta = self.gamma(x_), self.beta(x_)

        
        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)
        #print('G')
        #print(out.shape)
        return out, cam_logit, heatmap


class ResnetBlock(nn.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                       nn.Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(),
                       ReLU()]
        
        conv_block += [ReflectionPad2d(1),
                       nn.Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm()]

        
        self.conv_block = nn.Sequential(*conv_block)
       
    def forward(self, x):
        out = x + self.conv_block(x)
        
        return out


class ResnetAdaILNBlock(nn.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = nn.Conv2D(dim, dim, filter_size=3, stride=1, padding=0,bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU()

        self.pad2 = ReflectionPad2d(1)
        self.conv2 = nn.Conv2D(dim, dim, filter_size=3, stride=1, padding=0,bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x

# def var(input, dim, keep_dim=False):
#     in_mean1  = fluid.layers.reduce_mean(input=input, dim=dim, keep_dim=False)
#     in_sub = fluid.layers.elementwise_sub(input, in_mean1, axis=0)
#     in_square = fluid.layers.square(in_sub)
#     in_div = fluid.layers.elementwise_div(in_square, nn.to_variable(np.array(input.shape[0]).astype('float32')),axis=0)
#     in_var = fluid.layers.reduce_sum(in_div, dim=dim, keep_dim=keep_dim)
#     return in_var
# class var(nn.Layer):
#     def __init__(self,  dim, keep_dim=False):
#         super(var, self).__init__()
#         self.dim=dim
#         self.keep_dim=keep_dim
#     def forward(self,input):   
#         in_mean1  = fluid.layers.reduce_mean(input=input, dim=self.dim, keep_dim=True)
#         in_sub = fluid.layers.elementwise_sub(input, in_mean1, axis=0)
#         in_square = fluid.layers.square(in_sub)
#         in_div = fluid.layers.elementwise_div(in_square, nn.to_variable(np.array(input.shape[0]).astype('float32')),axis=0)
#         in_var = fluid.layers.reduce_sum(in_div, dim=self.dim, keep_dim=self.keep_dim)
#         return in_var
def var(input, axis=None, keepdim=False, unbiased=True, out=None, name=None):
   
    dtype = 'float32'
    if dtype not in ["float32", "float64"]:
        raise ValueError("Layer tensor.var() only supports floating-point "
                         "dtypes, but received {}.".format(dtype))
    rank = len(input.shape)
    axes = axis
    axes = [e if e >= 0 else e + rank for e in axes]
    inp_shape = input.shape
    mean = fluid.layers.reduce_mean(input, dim=axis, keep_dim=True, name=name)
    tmp = fluid.layers.reduce_mean(
        (input - mean)**2, dim=axis, keep_dim=keepdim, name=name)

    if unbiased:
        n = 1
        for i in axes:
            n *= inp_shape[i]

        factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    
    return tmp


class adaILN(nn.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rho = fluid.layers.create_parameter(shape=[1,num_features,1,1], dtype='float32',
        default_initializer=fluid.initializer.Constant(0.9))

    def forward(self, input, gamma, beta):
        #print(self.rho)
        
        # fluid.layers.clip(x=self.rho, min=0, max=1)
        in_mean  = fluid.layers.reduce_mean(input=input, dim=[2, 3], keep_dim=True)
        in_var = var(input, axis=[2,3], keepdim=True)
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean= fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_var = var(input, axis=[1,2,3], keepdim=True)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)

        clip_rho = fluid.layers.clip(x=fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1]), min=0.0, max=1.0)
        out = clip_rho * out_in + (1-clip_rho) * out_ln
        out = out * fluid.layers.unsqueeze(gamma, [2,3]) + fluid.layers.unsqueeze(beta, [2, 3])
       
        return out


class ILN(nn.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter(shape=[1,num_features,1,1], dtype='float32', is_bias=True, 
        default_initializer=fluid.initializer.Constant(0.0))
        self.gamma = fluid.layers.create_parameter(shape=[1,num_features,1,1], dtype='float32',is_bias=True, 
        default_initializer=fluid.initializer.Constant(1.0))
        self.beta = fluid.layers.create_parameter(shape=[1,num_features,1,1], dtype='float32',is_bias=True, 
        default_initializer=fluid.initializer.Constant(0.0))
        
    def forward(self, input):
        #print(self.beta)
        # emb = fluid.layers.Print(self.rho)
        # print(emb)
        

        in_mean = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        in_var = var(input, axis=[2,3], keepdim=True)
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean = fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_var = var(input, axis=[1,2,3], keepdim=True)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var , self.eps)

        clip_rho = fluid.layers.clip(x=fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1]), min=0.0, max=1.0)

        out = clip_rho * out_in + (1.0-clip_rho) * out_ln
        out = out * fluid.layers.expand(x=self.gamma, expand_times=[input.shape[0], 1, 1, 1]) + fluid.layers.expand(x=self.beta, expand_times=[input.shape[0], 1, 1, 1])
        
        return out





# class ResnetGenerator(nn.Layer):
#     def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
#         assert(n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         self.n_blocks = n_blocks
#         self.img_size = img_size
#         self.light = light

#         DownBlock = []

#         DownBlock.append(
#             nn.Conv2D(input_nc, ngf, filter_size=7, stride=1, padding=0,bias_attr=False),
#         )

#         n_downsampling = 2
#         for i in range(n_downsampling):
#             mult = 2**i
         
#             DownBlock.append(
#                 nn.Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0,bias_attr=False)
#             )

#         # Down-Sampling Bottleneck
#         mult = 2**n_downsampling
#         for i in range(n_blocks):
           
#             DownBlock.append(
#                 ResnetBlock(ngf * mult, use_bias=False)
#             )
        
#         # Class Activation Map
#         self.gap_fc = nn.Linear(ngf * mult, 1,bias_attr=False)
#         self.gmp_fc = nn.Linear(ngf * mult, 1,bias_attr=False)
#         self.conv1x1 = nn.Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1)
#         #self.relu = nn.ReLU(True)

#         # Gamma, Beta block
#         if self.light:
#             FC = [nn.Linear(ngf * mult, ngf * mult, act='relu',bias_attr=False),
#                   nn.Linear(ngf * mult, ngf * mult, act='relu',bias_attr=False)
#                 ]
#         else:
#             FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult,act='relu',bias_attr=False),
#                   nn.Linear(ngf * mult, ngf * mult,act='relu',bias_attr=False)
#                   ]
                  
#         self.gamma = nn.Linear(ngf * mult, ngf * mult,bias_attr=False)
#         self.beta = nn.Linear(ngf * mult, ngf * mult,bias_attr=False)

#         # Up-Sampling Bottleneck
#         for i in range(n_blocks):
#             setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

#         # Up-Sampling
#         UpBlock2 = []

#         #self.upsample = nn.UpSample(scale_factor=2, mode='NEAREST')
#         for i in range(n_downsampling):
#             mult = 2**(n_downsampling - i)
     
#             UpBlock2.append(
#                 nn.Sequential(
#                 nn.Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0,bias_attr=False),
#                 ILN(int(ngf * mult / 2)) 
#                 )
#             )
       
#         UpBlock2.append(nn.Sequential(
#             nn.Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0,bias_attr=False),
#         ))

#         #self.DownBlock = nn.Sequential(*DownBlock)
#         self.DownBlock = DownBlock
#         self.FC = nn.Sequential(*FC)
#         self.UpBlock2 = UpBlock2
#     def forward(self, input):
#         #  nn.InstanceNorm(ngf),
#         #     nn.ReLU(True)
#         for i in range(len(self.DownBlock)):
#             if i==0:
#                 x = fluid.layers.pad2d(input=input, paddings=[3,3,3,3], mode='reflect')
#                 #print(x.shape)
#                 x = self.DownBlock[i](x)
#                 x = fluid.layers.instance_norm(x)
#                 x = fluid.layers.relu(x)
#             elif i==1 or i==2:
#                 x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
#                 x = self.DownBlock[i](x)
#                 x = fluid.layers.instance_norm(x)
#                 x = fluid.layers.relu(x)
#             else:
#                 x = self.DownBlock[i](x)

#         #print(x.shape)  #[2, 256, 63, 63]
#         gap = fluid.layers.adaptive_pool2d(input=x, pool_size=(1,1), pool_type='avg')
#         #print(gap.shape)  #[2, 256, 1, 1]
#         # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)

#         #print(paddle.reshape(x=gap, shape=[x.shape[0], -1]).shape) #[2, 256]
#         gap_logit = self.gap_fc(fluid.layers.reshape(x=gap, shape=[x.shape[0], -1]))
#         #print(gap_logit.shape) #[2, 1]
#         # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
#         gap_weight = self.gap_fc.parameters()
#         #print(gap_weight[0].shape) #[256,1]
#         # gap_weight = list(self.gap_fc.parameters())[0]
#         #print(paddle.unsqueeze(input=gap_weight[0], axes=[0,2]).shape) #[1, 256, 1, 1]
#         gap = x * fluid.layers.unsqueeze(gap_weight[0], [0,2])
        
#         # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
#         gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=(1,1), pool_type='avg')
#         # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
#         gmp_logit = self.gmp_fc(fluid.layers.reshape(x=gmp, shape=[x.shape[0], -1]))
#         gmp_weight = self.gmp_fc.parameters()
#         gmp = x * fluid.layers.unsqueeze(gmp_weight[0], [0,2])

#         cam_logit = fluid.layers.concat([gap_logit, gmp_logit], axis=1)
#         x = fluid.layers.concat([gap, gmp], axis=1)
#         #x = self.relu(self.conv1x1(x))
#         x = fluid.layers.relu(self.conv1x1(x))

#         heatmap = fluid.layers.reduce_sum(input=x, dim=1, keep_dim=True)
#         #print(x.shape) #[2, 256, 64, 64]
#         if self.light:
#             # x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
#             x_ = fluid.layers.adaptive_pool2d(input=x, pool_size=(1,1), pool_type='avg')
#             # x_ = self.FC(x_.view(x_.shape[0], -1))
#             x_ = self.FC(fluid.layers.reshape(x=x_, shape=[x_.shape[0],-1]))
#         else:
#             # x_ = self.FC(x.view(x.shape[0], -1))
#             #print(x.shape)
#             x_ = self.FC(fluid.layers.reshape(x=x, shape=[x.shape[0],-1]))
#         gamma, beta = self.gamma(x_), self.beta(x_)

        
#         for i in range(self.n_blocks):
#             x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
#         for i in range(len(self.UpBlock2)):
#             if i<2:
#                 #x = self.upsample(x)
#                 x = fluid.layers.resize_nearest(x, scale=2) #, resample='NEAREST'
#                 x = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
#                 x = self.UpBlock2[i](x)
#                 x = fluid.layers.relu(x)
#             else:
#                 x = fluid.layers.pad2d(input=x, paddings=[3,3,3,3], mode='reflect')
#                 x = self.UpBlock2[i](x)
#                 out = fluid.layers.tanh(x)
#         #print(x.shape)
        
#         # out = self.UpBlock2(x)
#         #print('G')
        
#         return out, cam_logit, heatmap


# class ResnetBlock(nn.Layer):
#     def __init__(self, dim, use_bias):
#         super(ResnetBlock, self).__init__()
#         conv_block = []
#         # conv_block += [nn.ReflectionPad2d(1),
#         #                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
#         #                nn.InstanceNorm2d(dim),
#         #                nn.ReLU(True)]
#         conv_block.append(nn.Sequential(
#             nn.Conv2D(dim, dim, filter_size=3, stride=1, padding=0,bias_attr=use_bias),
#             nn.InstanceNorm(dim),
#         ))
#         # conv_block += [nn.ReflectionPad2d(1),
#         #                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
#         #                nn.InstanceNorm2d(dim)]

#         conv_block.append(nn.Sequential(
#             nn.Conv2D(dim, dim, filter_size=3, stride=1, padding=0,bias_attr=use_bias),
#             nn.InstanceNorm(dim)
#         ))
#         # self.conv_block = nn.Sequential(*conv_block)
#         self.conv_block = conv_block 
#     def forward(self, x):
#         for i in range(len(self.conv_block)):
#             y = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
#             y = self.conv_block[i](y)
#             if i==0:
#                 y = fluid.layers.relu(y)
        
#         # out = x + self.conv_block(x)
#         out = x + y
#         return out


# class ResnetAdaILNBlock(nn.Layer):
#     def __init__(self, dim, use_bias):
#         super(ResnetAdaILNBlock, self).__init__()
#         # self.pad1 = nn.ReflectionPad2d(1)
#         self.conv1 = nn.Conv2D(dim, dim, filter_size=3, stride=1, padding=0,bias_attr=use_bias)
#         self.norm1 = adaILN(dim)
#         #self.relu1 = nn.ReLU(True)

#         # self.pad2 = nn.ReflectionPad2d(1)
#         self.conv2 = nn.Conv2D(dim, dim, filter_size=3, stride=1, padding=0,bias_attr=use_bias)
#         self.norm2 = adaILN(dim)

#     def forward(self, x, gamma, beta):
#         # out = self.pad1(x)
#         out = fluid.layers.pad2d(input=x, paddings=[1,1,1,1], mode='reflect')
#         out = self.conv1(out)
#         #print(out.shape)
#         out = self.norm1(out, gamma, beta)
#         #print(out.shape)
#         #out = self.relu1(out)
#         out = fluid.layers.relu(out)
#         #out = self.pad2(out)
#         out = fluid.layers.pad2d(input=out, paddings=[1,1,1,1], mode='reflect')
#         out = self.conv2(out)
#         out = self.norm2(out, gamma, beta)

#         return out + x

# # def var(input, dim, keep_dim=False):
# #     in_mean1  = fluid.layers.reduce_mean(input=input, dim=dim, keep_dim=False)
# #     in_sub = fluid.layers.elementwise_sub(input, in_mean1, axis=0)
# #     in_square = fluid.layers.square(in_sub)
# #     in_div = fluid.layers.elementwise_div(in_square, nn.to_variable(np.array(input.shape[0]).astype('float32')),axis=0)
# #     in_var = fluid.layers.reduce_sum(in_div, dim=dim, keep_dim=keep_dim)
# #     return in_var
# # class var(nn.Layer):
# #     def __init__(self,  dim, keep_dim=False):
# #         super(var, self).__init__()
# #         self.dim=dim
# #         self.keep_dim=keep_dim
# #     def forward(self,input):   
# #         in_mean1  = fluid.layers.reduce_mean(input=input, dim=self.dim, keep_dim=True)
# #         in_sub = fluid.layers.elementwise_sub(input, in_mean1, axis=0)
# #         in_square = fluid.layers.square(in_sub)
# #         in_div = fluid.layers.elementwise_div(in_square, nn.to_variable(np.array(input.shape[0]).astype('float32')),axis=0)
# #         in_var = fluid.layers.reduce_sum(in_div, dim=self.dim, keep_dim=self.keep_dim)
# #         return in_var
# def var(input, axis=None, keepdim=False, unbiased=True, out=None, name=None):
   
#     dtype = 'float32'
#     if dtype not in ["float32", "float64"]:
#         raise ValueError("Layer tensor.var() only supports floating-point "
#                          "dtypes, but received {}.".format(dtype))
#     rank = len(input.shape)
#     axes = axis
#     axes = [e if e >= 0 else e + rank for e in axes]
#     inp_shape = input.shape
#     mean = fluid.layers.reduce_mean(input, dim=axis, keep_dim=True, name=name)
#     tmp = fluid.layers.reduce_mean(
#         (input - mean)**2, dim=axis, keep_dim=keepdim, name=name)

#     if unbiased:
#         n = 1
#         for i in axes:
#             n *= inp_shape[i]

#         factor = n / (n - 1.0) if n > 1.0 else 0.0
#         tmp *= factor
    
#     return tmp


# class adaILN(nn.Layer):
#     def __init__(self, num_features, eps=1e-5):
#         super(adaILN, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.rho = fluid.layers.create_parameter(shape=[1,num_features,1,1], dtype='float32',is_bias=True, 
#         default_initializer=fluid.initializer.Constant(0.9))
#         # self.var1 = var(dim=[1,2,3], keep_dim=True)
#         # self.var2 = var(dim=[2,3], keep_dim=True)
#     def forward(self, input, gamma, beta):
#         #print(self.rho.gradient())
#         in_mean  = fluid.layers.reduce_mean(input=input, dim=[2, 3], keep_dim=True)
#         # in_var = self.var2(input)
#         in_var = var(input, axis=[2,3], keepdim=True)
#         out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
#         ln_mean= fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)
#         #ln_var = var(input, dim=[1,2,3], keep_dim=True)
#         # ln_var = self.var1(input)
#         ln_var = var(input, axis=[1,2,3], keepdim=True)
#         out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        
#         out = fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1]) * out_in + (1-fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1])) * out_ln
#         # out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
#         out = out * fluid.layers.unsqueeze(gamma, [2,3]) + fluid.layers.unsqueeze(beta, [2, 3])
#         # out1 = (1-fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1])) * out_ln
#         # out = fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1]) * out_in 
#         # out = out+out1
#         # out = out * fluid.layers.unsqueeze(gamma, [2,3]) + fluid.layers.unsqueeze(beta, [2, 3])

#         return out


# class ILN(nn.Layer):
#     def __init__(self, num_features, eps=1e-5):
#         super(ILN, self).__init__()
#         self.eps = eps
#         self.rho = fluid.layers.create_parameter(shape=[1,num_features,1,1], dtype='float32',is_bias=True, 
#         default_initializer=fluid.initializer.Constant(0.0))
#         self.gamma = fluid.layers.create_parameter(shape=[1,num_features,1,1], dtype='float32',is_bias=True, 
#         default_initializer=fluid.initializer.Constant(1.0))
#         self.beta = fluid.layers.create_parameter(shape=[1,num_features,1,1], dtype='float32',is_bias=True, 
#         default_initializer=fluid.initializer.Constant(0.0))
        
#         # self.var1 = var(dim=[1,2,3], keep_dim=True)
#         # self.var2 = var(dim=[2,3], keep_dim=True)
#     def forward(self, input):
#         #print(self.gamma.gradient())
#         in_mean = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)

#         # in_var = self.var2(input)
#         in_var = var(input, axis=[2,3], keepdim=True)
#         out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
 

#         ln_mean = fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)

#         # ln_var = self.var1(input)
#         ln_var = var(input, axis=[1,2,3], keepdim=True)
#         out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var , self.eps)

#         out = fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1]) * out_in + (1.0-fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1])) * out_ln
#         # out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
#         out = out * fluid.layers.expand(x=self.gamma, expand_times=[input.shape[0], 1, 1, 1]) + fluid.layers.expand(x=self.beta, expand_times=[input.shape[0], 1, 1, 1])
#         #print(out)
#         #print(out_in.shape)
#         #print(fluid.layers.expand_as(x=self.rho, target_tensor=[input.shape[0], 1, 1, 1]).shape)
#         # out =  (1.0-fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1]))*out_ln
#         # out1 = fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1])*out_in
#         # #print(1,out1.shape)
#         # out = out1+out
#         #print(out.shape)expand_times
#         #out = fluid.layers.elementwise_add(fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1])*out_in , (1-fluid.layers.expand(x=self.rho, expand_times=[input.shape[0], 1, 1, 1]))*out_ln)

#         # out =  out * fluid.layers.expand(x=self.gamma, expand_times=[input.shape[0], 1, 1, 1])+ fluid.layers.expand(x=self.beta, expand_times=[input.shape[0], 1, 1, 1]) 
#         return out

# # class LinearSNReLU(nn.Layer):
# #     def __init__(self, in_channels, out_channels):
# #         super(LinearSNReLU, self).__init__(in_channels, out_channels)

# #     def forward(self, inputs):
# #         weight_sn = fluid.layers.spectral_norm(self.weight)
# #         x = fluid.layers.matmul(inputs, weight_sn)
# #         #x = fluid.layers.leaky_relu(x, alpha=0.2)

# #         return x
# # class SpectralNormConv2D(nn.Conv2D):
# #     def forward(self, input):
# #         attrs = ('strides', self._stride, 'paddings', self._padding,
# #                     'dilations', self._dilation, 'groups', self._groups
# #                     if self._groups else 1, 'use_cudnn', self._use_cudnn)

# #         weight = fluid.layers.spectral_norm(self.weight, dim=1)
# #         out = fluid.dygraph.core.ops.conv2d(input, weight, *attrs)
# #         pre_bias = out
# #         pre_act = fluid.dygraph_utils._append_bias_in_dygraph(pre_bias, self.bias, 1)
# #         return fluid.dygraph_utils._append_activation_in_dygraph(pre_act, self._act)
class Spectralnorm(nn.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = nn.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out
# class Conv2DSNReLU(nn.Conv2D):
#     def __init__(self, num_channels, num_filters, filter_size, stride=1, padding=0):
#         super(Conv2DSNReLU, self).__init__()
#         self._num_channels = num_channels
#         self._num_filters = num_filters
#         self._filter_size=num_filters
#         self._stride=stride
#         self._padding=padding

#     def forward(self, inputs):
#         x = fluid.layers.conv2d(inputs, self._num_channels, self._num_filters, self._filter_size, self._stride, self._padding)
#         weight_sn = fluid.layers.spectral_norm(x.weight, dim=1)
        
#         x = fluid.layers.leaky_relu(x, alpha=0.2)

#         return x

class Discriminator(nn.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model=[]
        model +=[
                ReflectionPad2d(1),
                Spectralnorm(nn.Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0,bias_attr=True), dim=1),
                LeakyReLU(0.2)
        ]
        
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [
                    ReflectionPad2d(1),
                    Spectralnorm(nn.Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0,bias_attr=True),dim=1),
                    LeakyReLU(0.2)
                    ]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d(1),
                Spectralnorm(nn.Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0,bias_attr=True),dim=1),
                LeakyReLU(0.2)
                ]
        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(nn.Linear(ndf * mult, 1,bias_attr=False))
        self.gmp_fc = Spectralnorm(nn.Linear(ndf * mult, 1,bias_attr=False))
        self.conv1x1 = nn.Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1,bias_attr=True)
        self.leaky_relu = LeakyReLU(0.2)

        self.pad = ReflectionPad2d(1)
        self.conv = Spectralnorm(nn.Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0,bias_attr=False),dim=1)

        self.model = nn.Sequential(*model)
    def forward(self, input):
        x = self.model(input)
        #print(input.shape) #[2, 3, 256, 256]
        gap = fluid.layers.adaptive_pool2d(input=x, pool_size=(1,1), pool_type='avg')
        gap_logit = self.gap_fc(fluid.layers.reshape(x=gap, shape=[x.shape[0], -1]))
        gap_weight = self.gap_fc.parameters()[0]
        gap =  x * fluid.layers.unsqueeze(gap_weight, [0,2])
        
        gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=(1,1), pool_type='avg')
        gmp_logit = self.gap_fc(fluid.layers.reshape(x=gmp, shape=[x.shape[0], -1]))
        gmp_weight = self.gmp_fc.parameters()[0]
        gmp =  x * fluid.layers.unsqueeze(gmp_weight, [0,2])

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(input=x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)
        
        return out, cam_logit, heatmap

# class BCEWithLogitsLoss():
#     def __init__(self, weight=None, reduction='mean'):
#         self.weight = weight
#         self.reduction = 'mean'

#     def __call__(self, x, label):
#         out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
#         if self.reduction == 'sum':
#             return fluid.layers.reduce_sum(out)
#         elif self.reduction == 'mean':
#             return fluid.layers.reduce_mean(out)
#         else:
#             return out
# class RhoClipper(object):

#     def __init__(self, min, max):
#         self.clip_min = min
#         self.clip_max = max
#         assert min < max

#     def __call__(self, module):

#         if hasattr(module, 'rho'):
#             w = module.rho.data
#             w = w.clamp(self.clip_min, self.clip_max)
#             module.rho.data = w
