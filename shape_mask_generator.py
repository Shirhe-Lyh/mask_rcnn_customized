# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:19:48 2018

@author: shirhe-lyh
"""

import cv2
import numpy as np


class ShapeMaskGenerator(object):
    """Generate shape masks."""
    
    def __init__(self, 
                 max_num_shapes=2, 
                 shape_types={'triangle': 3, 
                              'rectangle': 4},
                 size=[480, 480],
                 ignore_area_threshold=0.5):
        """Constructor."""
        self._max_num_shapes = max_num_shapes
        self._shape_types = shape_types
        self._ignore_area_threshold = ignore_area_threshold
        self._size = size
    
    def generate(self):
        """Generate shapes and masks."""
        num_shapes = np.random.randint(1, self._max_num_shapes + 1)
        type_indices = np.random.randint(0, len(self._shape_types), 
                                         size=num_shapes)
        
        image = None
        class_names = []
        vertexes_list = []
        shape_types = list(self._shape_types.keys())
        for type_index in type_indices:
            shape_type = shape_types[type_index]
            num_vertexes = self._shape_types[shape_type]
            vertexes = self._generate_vertexes(num_vertexes)
            image = self.generate_polygon(vertexes, image)
            class_names.append(shape_type)
            vertexes_list.append(vertexes)
            
        masks, class_names, vertexes_list = self.generate_mask(
            vertexes_list, class_names)
        return image, masks, class_names, vertexes_list
    
    def generate_mask(self, vertexes_list, class_names, height=None, 
                      width=None):
        """Generate masks."""
        if height is None:
            height = self._size[0]
        if width is None:
            width = self._size[1]
        masks = []
        valid_class_names = []
        valid_vertexes_list = []
        for i, vertexes in enumerate(vertexes_list):
            mask = np.zeros((height, width))
            cv2.fillPoly(mask, [np.array(vertexes)], 1)
            area = np.sum(mask)
            overlaps = vertexes_list[i+1:]
            overlaps = [np.array(vertexes) for vertexes in overlaps]
            cv2.fillPoly(mask, overlaps, 0)
            area_valid_mask = np.sum(mask)
            if area_valid_mask * 1.0 / area < self._ignore_area_threshold:
                continue
            mask = mask.astype(np.uint8)
            masks.append(mask)
            valid_class_names.append(class_names[i])
            valid_vertexes_list.append(vertexes_list[i])
        return masks, valid_class_names, valid_vertexes_list
            
    def _generate_vertexes(self, num_vertexes):
        """Generate vertexes."""
        vertexes = []
        height, width = self._size
        smallest_side = min(height, width)
        radius = np.random.randint(smallest_side // 5, smallest_side * 2 // 5)
        center_x = np.random.randint(radius * 6 // 5, 
                                     smallest_side - radius * 6 // 5)
        center_y = np.random.randint(radius * 6 // 5, 
                                     smallest_side - radius * 6 // 5)
        init_angle = np.random.uniform(0.0, 360.0) * np.pi / 180
        step_angle = 2 * np.pi / num_vertexes
        for i in range(num_vertexes):
            angle = init_angle + i * step_angle
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            vertexes.append([int(x), int(y)])
        return vertexes
    
    def generate_polygon(self, vertexes, image=None,height=None, width=None, 
                         color=None):
        """Generate a polygon.
        
        
        Args:
            vertexes: The vertexes of a polygon.
            image: An input image.
            height: The height of the generated image.
            width: The width of the generated image.
            color: The color to filling a polygon.
            
        Returns:
            image: The generated image.
            
        Raises:
            ValueError: If vertexes is invalid.
        """
        if vertexes is None:
            return image
        
        if len(vertexes) < 3:
            raise ValueError('`vertexes` is invalid.')
            
        if height is None:
            height = self._size[0]
        if width is None:
            width = self._size[1]
        if image is None:
            image = np.ones((height, width, 3)) * 255
        if color is None:
            color = self._random_color()
        if not isinstance(vertexes, np.ndarray):
            vertexes = np.array(vertexes)
        cv2.fillPoly(image, [vertexes], color)
        return image
    
    def generate_circle(self, center, radius, image=None, height=None, 
                        width=None, color=None):
        """Generate a circle.
        
        Args:
            center: The center of a circle.
            radius: The radius of a circle.
            height: The height of the generated image.
            width: The width of the generated image.
            color: The color to filling a polygon.
            
        Returns:
            image: The generated image.
            
        Raises:
            ValueError: If center or radius is invalid.
        """
        if center is None or len(center) < 2:
            raise ValueError('`center` is invalid.')
            
        if height is None:
            height = self._size[0]
        if width is None:
            width = self._size[1]
        smallest_side = min(height, width)
        if radius > smallest_side / 2.0:
            raise ValueError('`radius` is invalid.')
            
        if image is None:
            image = np.ones((height, width, 3)) * 255
        if color is None:
            color = self._random_color()
        if not isinstance(center, tuple):
            center = tuple(center)
        cv2.circle(image, center, radius, color, -1)
        return image
    
    def _random_color(self):
        """Returns a randomly selected color."""
        colors = [[255, 0, 85],
              [255, 0, 0],
              [255, 85, 0],
              [255, 170, 0],
              [255, 255, 0],
              [170, 255, 0],
              [85, 255, 0],
              [0, 255, 0],
              [0, 255, 85],
              [0, 255, 170],
              [0, 255, 255],
              [0, 170, 255],
              [0, 85, 255],
              [0, 0, 255],
              [255, 0, 170],
              [170, 0, 255],
              [255, 0, 255],
              [85, 0, 255]]
        index = np.random.randint(0, len(colors))
        return colors[index]