# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:48:59 2018

@author: shirhe-lyh
"""

from tensorflow.contrib.slim import nets

resnet_v1_block = nets.resnet_v1.resnet_v1_block
resnet_v2_block = nets.resnet_v2.resnet_v2_block


def resnet_v1_17(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_17'):
    """ResNet-17 model. See resnet_v1() for arg and return description.
    
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        num_classes: Number of predicted classes for classification tasks.
            If 0 or None, we return the features before the logit layer.
        is_training: whether batch_norm layers are in training mode. If this 
            is set to None, the callers can specify slim.batch_norm's 
            is_training parameter from an outer slim.arg_scope.
        global_pool: If True, we perform global average pooling before 
            computing the logits. Set to True for image classification, False 
            for dense prediction.
        output_stride: If None, then the output will be computed at the nominal
            network stride. If output_stride is not None, it specifies the 
            requested ratio of input to output spatial resolution.
        reuse: whether or not the network and its variables should be reused. 
            To be able to reuse 'scope' must be given.
        scope: Optional variable_scope.
      
    Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, 
            channels_out]. If global_pool is False, then height_out and 
            width_out are reduced by a factor of output_stride compared to 
            the respective height_in and width_in, else both height_out and 
            width_out equal one. If num_classes is 0 or None, then net is 
            the output of the last ResNet block, potentially after global 
            average pooling. If num_classes a non-zero integer, net contains 
            the pre-softmax activations.
      end_points: A dictionary from components of the network to the 
          corresponding activation.
      
    Raises:
        ValueError: If the target output_stride is not valid.
    """
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=1, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=1, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=2, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=1, stride=1)
    ]
    return nets.resnet_v1.resnet_v1(
        inputs, 
        blocks, 
        num_classes, 
        is_training,
        global_pool=global_pool, 
        output_stride=output_stride,
        reuse=reuse,
        scope=scope)
    

def resnet_v1_20(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 reuse=None,
                 scope='resnet_v1_20'):
    """ResNet-20 model. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=1, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=1, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=2, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)
    ]
    return nets.resnet_v1.resnet_v1(
        inputs, 
        blocks, 
        num_classes, 
        is_training,
        global_pool=global_pool, 
        output_stride=output_stride,
        include_root_block=True, 
        reuse=reuse,
        scope=scope)


def resnet_v2_14(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_14'):
    """ResNet-14 model. See resnet_v2() for arg and return description.
    
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        num_classes: Number of predicted classes for classification tasks. 
            If None we return the features before the logit layer.
        is_training: whether batch_norm layers are in training mode.
        global_pool: If True, we perform global average pooling before 
            computing the logits. Set to True for image classification, 
            False for dense prediction.
        output_stride: If None, then the output will be computed at the 
            nominal network stride. If output_stride is not None, it specifies 
            the requested ratio of input to output spatial resolution.
        reuse: whether or not the network and its variables should be reused. 
            To be able to reuse 'scope' must be given.
        scope: Optional variable_scope.
        
    Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, 
            channels_out]. If global_pool is False, then height_out and 
            width_out are reduced by a factor of output_stride compared to the 
            respective height_in and width_in, else both height_out and 
            width_out equal one. If num_classes is None, then net is the 
            output of the last ResNet block, potentially after global average 
            pooling. If num_classes is not None, net contains the pre-softmax
            activations.
        end_points: A dictionary from components of the network to the 
            corresponding activation.
            
    Raises:
        ValueError: If the target output_stride is not valid.
    """
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=1, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=1, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=1, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=1, stride=1)
    ]
    return nets.resnet_v2.resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)
    
    
def resnet_v2_17(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_17'):
    """ResNet-17 model. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=1, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=1, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=2, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=1, stride=1)
    ]
    return nets.resnet_v2.resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)
                

def resnet_v2_20(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_17'):
    """ResNet-17 model. See resnet_v2() for arg and return description."""
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=1, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=2, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=2, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=1, stride=1)
    ]
    return nets.resnet_v2.resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)
