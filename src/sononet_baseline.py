import torch.nn as nn
from .model_utils import unetConv2, conv2DBatchNormRelu, conv2DBatchNorm, init_weights
import torch.nn.functional as F
class Sononet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=14, in_channels=3, is_batchnorm=True, n_convs=None):
        super(Sononet, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes= n_classes

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        if n_convs is None:
            n_convs = [3,3,3,2,2]

        # downsampling
        conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, n=n_convs[0])
        maxpool1 = nn.MaxPool2d(kernel_size=2)

        conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, n=n_convs[1])
        maxpool2 = nn.MaxPool2d(kernel_size=2)

        conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, n=n_convs[2])
        maxpool3 = nn.MaxPool2d(kernel_size=2)

        conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, n=n_convs[3])
        maxpool4 = nn.MaxPool2d(kernel_size=2)

        conv5 = unetConv2(filters[3], filters[3], self.is_batchnorm, n=n_convs[4])

        # adaptation layer
        conv5_p = conv2DBatchNormRelu(filters[3], filters[2], 1, 1, 0)
        conv6_p = conv2DBatchNorm(filters[2], filters[1], 1, 1, 0)
        
        self.features = nn.Sequential(conv1,
                                     maxpool1,
                                     conv2,
                                     maxpool2,
                                     conv3,
                                     maxpool3,
                                     conv4,
                                     maxpool4,
                                     conv5,
                                     conv5_p,
                                     conv6_p)
        
        self.flatten = nn.Flatten()
        
        # defining the classifier
        self.classifier = nn.Sequential(nn.Linear(filters[1], self.n_classes))
        
        # placeholder for the gradients
        self.gradients = None

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
                
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad


    def forward(self, inputs):
        # Feature Extraction
        
        conv6_p = self.features(inputs)
        
        # register the hook
        if(conv6_p.requires_grad):
            h = conv6_p.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        batch_size = inputs.shape[0]
        
        pooled     = F.adaptive_avg_pool2d(conv6_p, (1, 1))
        
        out = self.flatten(pooled)
        out = self.classifier(out)

        return out
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return [self.gradients]
    
    # method for the activation exctraction
    def get_activations(self, x):
        return [self.features(x)]