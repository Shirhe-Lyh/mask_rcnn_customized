# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:08:15 2018

@author: shirhe-lyh
"""

import cv2
import json
import numpy as np
import os

import shape_mask_generator

SHAPE_MASK_GENERATOR = shape_mask_generator.ShapeMaskGenerator(
    max_num_shapes=1)


def generate(num_samples=5000, output_directory=None):
    """Generate images, and masks."""
    if output_directory is None:
        raise ValueError('`output_directory` must be specified.')
        
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    images_dir = os.path.join(output_directory, 'images')
    masks_dir = os.path.join(output_directory, 'masks')
    os.mkdir(images_dir)
    os.mkdir(masks_dir)
        
    annotation_dict = {}
    for i in range(num_samples):
        image_name = 'image_{}.jpg'.format(i)
        image_path = os.path.join(images_dir, image_name)
        image, masks, class_names, _ = SHAPE_MASK_GENERATOR.generate()
        if len(masks) < 1:
            continue
        cv2.imwrite(image_path, image)
        mask_name_prefix = 'image_{}'.format(i)
        annotation_list = []
        for i, (mask, class_name) in enumerate(zip(masks, class_names)):
            mask_name = mask_name_prefix + '_{}_{}.png'.format(
                class_name, i)
            mask_path = os.path.join(masks_dir, mask_name)
            cv2.imwrite(mask_path, mask)
            
            # Boundingbox
            nonzero_y_indices, nonzero_x_indices = np.nonzero(mask)
            ymin = np.min(nonzero_y_indices)
            ymax = np.max(nonzero_y_indices)
            xmin = np.min(nonzero_x_indices)
            xmax = np.max(nonzero_x_indices)
            bbox = [int(ymin), int(xmin), int(ymax), int(xmax)]
            
            annotation_list.append([mask_name, class_name, bbox])
        annotation_dict[image_name] = annotation_list
        
    json_path = os.path.join(output_directory, 'annotations.json')
    with open(json_path, 'w') as writer:
        json.dump(annotation_dict, writer)
        
    
if __name__ == '__main__':
    generate(output_directory='./datasets')
