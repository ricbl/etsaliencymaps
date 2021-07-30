import torch.nn as nn
from .model_utils import unetConv2, init_weights
import torch
import torch.nn.functional as F
from .grid_attention_layer import GridAttentionBlock2D_TORR as AttentionBlock2D
torch.autograd.set_detect_anomaly(True)
class AG_Sononet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=14, in_channels=3, is_batchnorm=True, n_convs=None,
                 nonlocal_mode='concatenation_sigmoid', aggregation_mode='concat'):
        super(AG_Sononet, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes= n_classes

        if n_convs is None:
            n_convs = [3, 3, 3, 2, 2]

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        ####################
        # Feature Extraction
        conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, n=n_convs[0])
        maxpool1 = nn.MaxPool2d(kernel_size=2)

        conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, n=n_convs[1])
        maxpool2 = nn.MaxPool2d(kernel_size=2)

        conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, n=n_convs[2])
        maxpool3 = nn.MaxPool2d(kernel_size=2)

        conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, n=n_convs[3])
        maxpool4 = nn.MaxPool2d(kernel_size=2)

        conv5 = unetConv2(filters[3], filters[3], self.is_batchnorm, n=n_convs[4])
        
        
        # Features
        
        self.features1 = nn.Sequential(conv1,
                                     maxpool1,
                                     conv2,
                                     maxpool2,
                                     conv3)

        self.features2 = nn.Sequential(maxpool3,
                                       conv4)
        
        self.features = nn.Sequential(maxpool4,
                                     conv5)

        ################
        # Attention Maps
        self.compatibility_score1 = AttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        self.compatibility_score2 = AttentionBlock2D(in_channels=filters[3], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        #########################0
        # Aggreagation Strategies
        self.attention_filter_sizes = [filters[2], filters[3]]

        if aggregation_mode == 'concat':
            self.classifier = nn.Sequential(nn.Linear(filters[2]+filters[3]+filters[3], n_classes))
            self.aggregate = self.aggregation_concat

        ####################
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
                
        ####################
        # placeholder for gradients
        self.gradients1 = None
        self.gradients2 = None
        self.gradients3 = None

    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))
        
        
    # hook for the gradients of the activations
    def activations_hook1(self, grad):
        self.gradients1 = grad
        
    # hook for the gradients of the activations
    def activations_hook2(self, grad):
        self.gradients2 = grad
        
    # hook for the gradients of the activations
    def activations_hook3(self, grad):
        self.gradients3 = grad
        
    # hook for the gradients of the activations
    def activations_hook_att1(self, grad):
        self.att1 = grad
    # hook for the gradients of the activations
    def activations_hook_att2(self, grad):
        self.att2 = grad

    def forward(self, inputs):

        conv3 = self.features1(inputs)

        conv4 = self.features2(conv3)
        
        conv5 = self.features(conv4)
        conv5_for_pool = conv5*1
        # register the hook for features
        if(conv5_for_pool.requires_grad):
            h1 = conv5_for_pool.register_hook(self.activations_hook1)
        self.activation1 = conv5_for_pool.detach().clone()
        
        batch_size = inputs.shape[0]
        pooled     = F.adaptive_avg_pool2d(conv5_for_pool, (1, 1)).view(batch_size, -1)

        # Attention Mechanism
        g_conv1, att1 = self.compatibility_score1(conv3, conv5)
        g_conv2, att2 = self.compatibility_score2(conv4, conv5)
        
        # register the hook for attention maps
        if(g_conv1.requires_grad):
            h2 = g_conv1.register_hook(self.activations_hook2)
        self.activation2 = g_conv1.detach().clone()
        if(g_conv2.requires_grad):
            h3 = g_conv2.register_hook(self.activations_hook3)
        self.activation3 = g_conv2.detach().clone()
        self.att1 = att1.detach()
        self.att2 = att2.detach()
        
        # flatten to get single feature vector
        fsizes = self.attention_filter_sizes
        g1 = torch.sum(g_conv1.view(batch_size, fsizes[0], -1), dim=-1)
        g2 = torch.sum(g_conv2.view(batch_size, fsizes[1], -1), dim=-1)

        return self.aggregate(g1, g2, pooled)
    
    def get_activations_gradient(self):
        return [self.gradients1, self.gradients2, self.gradients3]
    
    def get_attentions(self):
        # return [self.compatibility_score1(self.features1(x), self.get_activations1(x))[1].squeeze(0).squeeze(0), self.compatibility_score2(self.features2(self.features1(x)), self.get_activations1(x))[1].squeeze(0).squeeze(0)]
        return [self.att1.squeeze(0).squeeze(0), self.att2.squeeze(0).squeeze(0)]
        
    def get_activations(self, x):
        # return [self.features(self.features2(self.features1(x))), self.compatibility_score1(self.features1(x), self.get_activations1(x))[0], self.compatibility_score2(self.features2(self.features1(x)), self.get_activations1(x))[0]]
        return [self.activation1, self.activation2, self.activation3]
    