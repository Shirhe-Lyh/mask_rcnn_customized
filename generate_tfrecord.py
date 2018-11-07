# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:31:00 2018

@author: shirhe-lyh


Convert raw dataset to TFRecord for object_detection.
"""

import glob
import hashlib
import io
import json
import numpy as np
import os
import PIL.Image
import tensorflow as tf

import read_pbtxt_file


flags = tf.app.flags

flags.DEFINE_string('images_dir', 
                    './datasets/images', 
                    'Path to images directory.')
flags.DEFINE_string('masks_dir', 
                    './datasets/masks', 
                    'Path to masks directory.')
flags.DEFINE_string('annotation_json_path', 
                    './datasets/annotations.json', 
                    'Path to annotations .json file.')
flags.DEFINE_string('label_map_path', 
                    './shape_label_map.pbtxt', 
                    'Path to label map proto.')
flags.DEFINE_string('output_train_record_path', 
                    './datasets/train.record', 
                    'Path to the output train tfrecord.')
flags.DEFINE_string('output_val_record_path', 
                    './datasets/val.record', 
                    'Path to the output validation tfrecord.')
flags.DEFINE_float('train_val_split_ratio',
                   0.1,
                   'The ratio of validation images in all images.')

FLAGS = flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(annotation_dict, label_map_dict=None):
    """Converts image and annotations to a tf.Example proto.
    
    Args:
        annotation_dict: A dictionary containing the following keys:
            ['height', 'width', 'filename', 'sha256_key', 'encoded_jpg',
             'format', 'xmins', 'xmaxs', 'ymins', 'ymaxs', 'masks',
             'class_names'].
        label_map_dict: A dictionary maping class_names to indices.
            
    Returns:
        example: The converted tf.Example.
        
    Raises:
        ValueError: If label_map_dict is None or is not containing a class_name.
    """
    if annotation_dict is None:
        return None
    if label_map_dict is None:
        raise ValueError('`label_map_dict` is None')
        
    height = annotation_dict.get('height', None)
    width = annotation_dict.get('width', None)
    filename = annotation_dict.get('filename', None)
    sha256_key = annotation_dict.get('sha256_key', None)
    encoded_jpg = annotation_dict.get('encoded_jpg', None)
    image_format = annotation_dict.get('format', None)
    xmins = annotation_dict.get('xmins', None)
    xmaxs = annotation_dict.get('xmaxs', None)
    ymins = annotation_dict.get('ymins', None)
    ymaxs = annotation_dict.get('ymaxs', None)
    masks = annotation_dict.get('masks', None)
    class_names = annotation_dict.get('class_names', None)
    
    labels = []
    for class_name in class_names:
        label = label_map_dict.get(class_name, None)
        if label is None:
            raise ValueError('`label_map_dict` is not containing {}.'.format(
                class_name))
        labels.append(label)
        
    feature_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(filename.encode('utf8')),
        'image/key/sha256': bytes_feature(sha256_key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/mask': bytes_list_feature(masks),
        'image/object/class/label': int64_list_feature(labels)}
    example = tf.train.Example(features=tf.train.Features(
        feature=feature_dict))
    return example


def get_annotation_dict(images_dir, image_name=None, masks_dir=None,
                        annotation_list=None):  
    """Get boundingboxes and masks.
    
    Args:
        images_dir: Path to images directory.
        image_name: The name of a given image.
        masks_dir: Path to masks directory.
        annotation_list: A list containing the mask_names, class_names, ...,
            corresponding to the given image, with format:
            [[mask_name, class_name], ...].
            
    Returns:
        annotation_dict: A dictionary containing the following keys:
            ['height', 'width', 'filename', 'sha256_key', 'encoded_jpg',
             'format', 'xmins', 'xmaxs', 'ymins', 'ymaxs', 'masks',
             'class_names'].
#            
#    Raises:
#        ValueError: If images_dir, image_name, masks_dir or annotation_list 
#            does not exist.
    """
#    if not os.path.exists(images_dir):
#        raise ValueError('`images_dir` does not exist.')
#    if not os.path.exists(masks_dir):
#        raise ValueError('`masks_dir` does not exist.')
#    if image_name is None:
#        raise ValueError('`images_name` must be specified.') 
#    if annotation_list is None:
#        raise ValueError('`annotation_list` must be specified.')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        return None
    if image_name is None or annotation_list is None:
        return None
    
    image_path = os.path.join(images_dir, image_name)
    image_format = image_name.split('.')[-1].replace('jpg', 'jpeg')
    if not os.path.exists(image_path):
        return None
    
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width, height = image.size
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    masks = []
    class_names = []
    for mask_name, class_name, bbox in annotation_list:
        class_names.append(class_name)
        mask_path = os.path.join(masks_dir, mask_name)
        if not os.path.exists(mask_path):
            print('mask_path: %s does not exist.' % mask_path)
            return None
        with tf.gfile.GFile(mask_path, 'rb') as fid:
            encoded_mask = fid.read()
        
        masks.append(encoded_mask)
        ymin, xmin, ymax, xmax = bbox
        xmins.append(float(xmin) / width)
        xmaxs.append(float(xmax) / width)
        ymins.append(float(ymin) / height)
        ymaxs.append(float(ymax) / height)
        
    annotation_dict = {'height': height,
                       'width': width,
                       'filename': image_name,
                       'sha256_key': key,
                       'encoded_jpg': encoded_jpg,
                       'format': image_format,
                       'xmins': xmins,
                       'xmaxs': xmaxs,
                       'ymins': ymins,
                       'ymaxs': ymaxs,
                       'masks': masks,
                       'class_names': class_names}
    return annotation_dict


def _split_train_val_datasets():
    images_dir = FLAGS.images_dir
    train_val_split_ratio = FLAGS.train_val_split_ratio
    
    if not os.path.exists(images_dir):
        raise ValueError('`images_dir` does not exist.')
        
    image_files = glob.glob(os.path.join(images_dir, '*.*'))
    num_images = len(image_files)
    num_val_images = int(num_images * train_val_split_ratio)
    image_files = np.array(image_files)
    np.random.shuffle(image_files)
    train_image_files = image_files[num_val_images:]
    val_image_files = image_files[:num_val_images]
    return train_image_files, val_image_files


def main(_):
    if not os.path.exists(FLAGS.images_dir):
        raise ValueError('`images_dir` does not exist.')
    if not os.path.exists(FLAGS.masks_dir):
        raise ValueError('`masks_dir` does not exist.')
    if not os.path.exists(FLAGS.annotation_json_path):
        raise ValueError('`annotation_json_path` does not exist.')
    if not os.path.exists(FLAGS.label_map_path):
        raise ValueError('`label_map_path` does not exist.')
        
    train_image_files, val_image_files = _split_train_val_datasets()
    label_map = read_pbtxt_file.get_label_map_dict(FLAGS.label_map_path)
    with open(FLAGS.annotation_json_path) as reader:
        annotation_json_dict = json.load(reader)
    
    train_writer = tf.python_io.TFRecordWriter(FLAGS.output_train_record_path)
    val_writer = tf.python_io.TFRecordWriter(FLAGS.output_val_record_path)
      
    num_skiped = 0
    print('write training record ...')
    for i, image_file in enumerate(train_image_files):
        if i % 100 == 0:
            print('On image %d', i)
            
        image_name = image_file.split('/')[-1]
        annotation_list = annotation_json_dict.get(image_name, None)
        annotation_dict = get_annotation_dict(
            FLAGS.images_dir, image_name, FLAGS.masks_dir, annotation_list)
        if annotation_dict is None:
            num_skiped += 1
            continue
        tf_example = create_tf_example(annotation_dict, label_map)
        train_writer.write(tf_example.SerializeToString())
    print('Skiped %d images in writing training tfrecord.' % num_skiped)
    print('Successfully created TFRecord to {}.'.format(
        FLAGS.output_train_record_path))
    
    num_skiped = 0
    print('write validation record ...')
    for i, image_file in enumerate(val_image_files):
        if i % 100 == 0:
            print('On image %d', i)
            
        image_name = image_file.split('/')[-1]
        annotation_list = annotation_json_dict.get(image_name, None)
        annotation_dict = get_annotation_dict(
            FLAGS.images_dir, image_name, FLAGS.masks_dir, annotation_list)
        if annotation_dict is None:
            num_skiped += 1
            continue
        tf_example = create_tf_example(annotation_dict, label_map)
        val_writer.write(tf_example.SerializeToString())
    print('Skiped %d images in writing validation tfrecord.' % num_skiped)
    print('Successfully created TFRecord to {}.'.format(
        FLAGS.output_val_record_path))


if __name__ == '__main__':
    tf.app.run()

