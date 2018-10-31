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
                 shape_types=['triangle', 'rectangle'],
                 size=[480, 480],
                 ignore_area_threshold=0.5):
        """Constructor."""
        self._max_num_shapes = max_num_shapes
        self._shape_types = shape_types
        self._ignore_area_threshold = ignore_area_threshold
        self._size = size
        
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
    
    def generate(self):
        """Generate shapes and masks."""
        num_shapes = np.random.randint(1, self._max_num_shapes + 1)
        type_indices = np.random.randint(0, len(self._shape_types), 
                                         size=num_shapes)
        
        image = None
        class_names = []
        vertexes_list = []
        for type_index in type_indices:
            shape_type = self._shape_types[type_index]
            if shape_type == 'triangle':
                vertexes = self._generate_vertexes(3, 100)
                image = self.generate_polygon(vertexes, image)
            elif shape_type == 'rectangle':
                vertexes = self._generate_rect_vertexes()
                image = self.generate_polygon(vertexes, image)
            else:
                raise ValueError('Unknown shape_type: %s' % shape_type)
            if vertexes is not None:
                vertexes_list.append(vertexes)
                class_names.append(shape_type)
        if image is None:
            image, vertexes = self._generate_default_triangle()
            vertexes_list.append(vertexes)
            class_names.append('triangle')
            
        masks = self.generate_mask(vertexes_list)
        return image, masks, class_names
    
    def generate_mask(self, vertexes_list, height=None, width=None):
        """Generate masks."""
        if height is None:
            height = self._size[0]
        if width is None:
            width = self._size[1]
        masks = []
        for i, vertexes in enumerate(vertexes_list):
            mask = np.zeros((height, width))
            print(vertexes)
            cv2.fillPoly(mask, [np.array(vertexes)], 1)
            overlaps = vertexes_list[i+1:]
            overlaps = [np.array(vertexes) for vertexes in overlaps]
            cv2.fillPoly(mask, overlaps, 0)
            mask = mask.astype(np.uint8)
            masks.append(mask)
        return masks
            
    def _generate_vertexes(self, num_vertexes, ignore_threshold=None):
        """Generate vertexes."""
        vertexes = []
        height, width = self._size
        for i in range(num_vertexes):
            y = np.random.randint(height // 6, height * 5 // 6)
            x = np.random.randint(width // 6, width * 5 // 6)
            vertexes.append([x, y])
        if ignore_threshold is not None:
            x0, y0 = vertexes[0]
            for x, y in vertexes[1:]:
                min_diff = min(abs(x - x0), abs(y - y0))
                if min_diff < ignore_threshold:
                    return None
        return vertexes
    
    def _generate_rect_vertexes(self):
        """Generate vertexes for a rectangle."""
        height, width = self._size
        xmin, ymin = [max(height, width)] * 2
        xmax, ymax = [0] * 2
        for i in range(2):
            y = np.random.randint(height // 6, height * 5 // 6)
            x = np.random.randint(width // 6, width * 5 // 6)
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        if xmax - xmin < 50:
            xmax += 50
        if ymax - ymin < 50:
            ymax += 50
        vertexes = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
        return vertexes
    
    def _generate_default_triangle(self):
        vertexes = [[240, 180], [180, 300], [300, 300]]
        random_step = np.random.randint(-100, 100)
        random_axis = np.random.randint(0, 2)
        for i in range(3):
            vertexes[i][random_axis] += random_step
        image = self.generate_polygon(vertexes, None)
        return image, vertexes
    
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