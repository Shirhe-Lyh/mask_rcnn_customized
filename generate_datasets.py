# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:08:15 2018

@author: shirhe-lyh
"""

import cv2
import json
import os

import shape_mask_generator

SHAPE_MASK_GENERATOR = shape_mask_generator.ShapeMaskGenerator()


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
        image, masks, class_names = SHAPE_MASK_GENERATOR.generate()
        cv2.imwrite(image_path, image)
        mask_name_prefix = 'image_{}'.format(i)
        mask_name_to_class_names = []
        for i, (mask, class_name) in enumerate(zip(masks, class_names)):
            mask_name = mask_name_prefix + '_{}_{}.png'.format(
                class_name, i)
            mask_path = os.path.join(masks_dir, mask_name)
            cv2.imwrite(mask_path, mask)
            mask_name_to_class_names.append([mask_name, class_name])
        annotation_dict[image_name] = mask_name_to_class_names
        
    json_path = os.path.join(output_directory, 'annotations.json')
    with open(json_path, 'w') as writer:
        json.dump(annotation_dict, writer)
        
    
if __name__ == '__main__':
    generate(output_directory='./datasets')
