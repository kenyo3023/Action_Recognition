"""
Source: https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks/
"""

import logging

import torch.nn as nn


# Initiate Logger
logger = logging.getLogger(__name__)


def init_xavier_weights(net):
    # The weights of conv layer and fully connected layers are both initilized with Xavier algorithm
    # We set the parameters to random values uniformly drawn from [-a, a] where a = sqrt(6 * (din + dout))
    # For batch normalization layers, y=1, b=0, all bias initialized to 0
    logger.info("Performing Xavier Weight Init!")
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return net
