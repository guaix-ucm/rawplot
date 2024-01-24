# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Copyright (c) 2021
#
# See the LICENSE file for details
# see the AUTHORS file for authors
# ----------------------------------------------------------------------

#--------------------
# System wide imports
# -------------------

import os
import fractions
import logging

# ---------------------
# Thrid-party libraries
# ---------------------

import rawpy
import exifread
import numpy as np

# ---------
# Constants
# ---------

log = logging.getLogger(__name__)

# ----------
# Exceptions
# ----------

class UnsupportedCFAError(ValueError):
    '''Unsupported Color Filter Array type'''
    def __str__(self):
        s = self.__doc__
        if self.args:
            s = ' {0}: {1}'.format(s, str(self.args[0]))
        s = '{0}.'.format(s)
        return s

# ----------------------
# Module utility classes
# ----------------------

class Point:
    """ Point class represents and manipulates x,y coords. """

    PATTERN = r'\((\d+),(\d+)\)'

    @classmethod
    def from_string(cls, point_str):
        pattern = re.compile(Point.PATTERN)
        matchobj = pattern.search(Rect_str)
        if matchobj:
            x = int(matchobj.group(1))
            y = int(matchobj.group(2))
            return cls(x,y)
        else:
            return None

    def __init__(self, x=0, y=0):
        """ Create a new point at the origin """
        self.x = x
        self.y = y

    def __add__(self, rect):
        return NotImplementedError

    def __repr__(self):
        return f"({self.x},{self.y})"

class Rect:
    """ Region of interest """

    PATTERN = r'\[(\d+):(\d+),(\d+):(\d+)\]'

    @classmethod
    def from_string(cls, Rect_str):
        '''numpy sections style'''
        pattern = re.compile(Rect.PATTERN)
        matchobj = pattern.search(Rect_str)
        if matchobj:
            y1 = int(matchobj.group(1))
            y2 = int(matchobj.group(2))
            x1 = int(matchobj.group(3))
            x2 = int(matchobj.group(4))
            return cls(x1,x2,y1,y2)
        else:
            return None

    @classmethod
    def from_normalized(cls, width, height, n_x0=None, n_y0=None, n_width=1.0, n_height=1.0, debayered=True):
        if n_x0 is not None:
            assert n_x0 + n_width <= 1.0, f"normalized x0(={n_x0}) + width(={n_width}) = {n_x0 + n_width} exceeds 1.0"
        if n_y0 is not None:
            assert n_y0 + n_height <= 1.0, f"normalized x0(={n_y0}) + width(={n_height}) = {n_y0 + n_height} exceeds 1.0"
        # If debayered, we'll adjust to each image plane dimensions
        if debayered:
            height = height //2  
            width  = width  //2 
        # From normalized ROI to actual image dimensions ROI
        w = int(width * n_width) 
        h = int(height * n_height)
        x0 = (width  - w)//2 if n_x0 is None else int(width * n_x0)
        y0 = (height - h)//2 if n_y0 is None else int(height * n_y0)
        return cls(x0, x0+w ,y0, y0+h)

    @classmethod
    def from_dict(cls, Rect_dict):
        return cls(Rect_dict['x1'], Rect_dict['x2'],Rect_dict['y1'], Rect_dict['y2'])
    
    def __init__(self, x1 ,x2, y1, y2):        
        self.x1 = min(x1,x2)
        self.y1 = min(y1,y2)
        self.x2 = max(x1,x2)
        self.y2 = max(y1,y2)

    def to_dict(self):
        return {'x1':self.x1, 'y1':self.y1, 'x2':self.x2, 'y2':self.y2}
        
    def xy(self):
        '''To use when displaying Rectangles in matplotlib'''
        return(self.x1, self.y1)

    def width(self):
        return abs(self.x2 - self.x1)

    def height(self):
        return abs(self.y2 - self.y1)
        
    def dimensions(self):
        '''returns width and height'''
        return abs(self.x2 - self.x1), abs(self.y2 - self.y1)

    def __add__(self, point):
        return Rect(self.x1 + point.x, self.x2 + point.x, self.y1 + point.y, self.y2 + point.y)

    def __radd__(self, point):
        return self.__add__(point)
        
    def __repr__(self):
        '''string in NumPy section notation'''
        return f"[{self.y1}:{self.y2},{self.x1}:{self.x2}]"
      

# ----------------
# Auxiliar classes
# ----------------

class RawImage:

    LABELS = (('Red', 'R'), ('Green 1','G1'), ('Green 2', 'G2'), ('Blue', 'B') )
    BAYER_LETTER = ['B','G','R','G']
    BAYER_PTN_LIST = ('RGGB', 'BGGR', 'GRBG', 'GBRG')
    CFA_OFFSETS = {
        # Esto era segun mi entendimiento
        'RGGB' : {'R':{'x': 0,'y': 0}, 'G1':{'x': 1,'y': 0}, 'G2':{'x': 0,'y': 1}, 'B':{'x': 1,'y': 1}}, 
        'BGGR' : {'R':{'x': 1,'y': 1}, 'G1':{'x': 1,'y': 0}, 'G2':{'x': 0,'y': 1}, 'B':{'x': 0,'y': 0}},
        'GRBG' : {'R':{'x': 1,'y': 0}, 'G1':{'x': 0,'y': 0}, 'G2':{'x': 1,'y': 1}, 'B':{'x': 0,'y': 1}},
        'GBRG' : {'R':{'x': 0,'y': 1}, 'G1':{'x': 0,'y': 0}, 'G2':{'x': 1,'y': 1}, 'B':{'x': 1,'y': 0}},
    }

    CHANNELS = ('R', 'G1', 'G2', 'B')

    def __init__(self, path):
        self._path = path
        self._color_desc = None
        self._cfa = None
        self._biases = None
        self._white_levels = None
        self._metadata = dict()
        self._minimal_metadata()
       

    def _read_img_metadata(self, img_desc):
        self._color_desc = img_desc.color_desc.decode('utf-8')
        self._cfa = ''.join([ self.BAYER_LETTER[img_desc.raw_pattern[row,column]] for row in (1,0) for column in (1,0)])
        self._biases = img_desc.black_level_per_channel
        self._white_levels = img_desc.camera_white_level_per_channel

    def _img(self):
        '''Gather as much rawpy image access as possible for efficiency'''
        with rawpy.imread(self._path) as img:
            self._read_img_metadata(img)

    def _minimal_metadata(self):
        with open(self._path, 'rb') as f:
            exif = exifread.process_file(f, details=False)
        if not exif:
            raise ValueError('Could not get ExifImageWidth, ExifImageLength tags')
        width  = int(str(exif.get('EXIF ExifImageWidth')))
        height = int(str(exif.get('EXIF ExifImageLength')))
        self._shape = (height, width)
        self._metadata['exposure'] = fractions.Fraction(str(exif.get('EXIF ExposureTime', 0)))
        self._metadata['width'] = width
        self._metadata['height'] = height

    def _exif(self):
        with open(self._path, 'rb') as f:
            exif = exifread.process_file(f, details=True)
        if not exif:
            raise ValueError('Could not open EXIF metadata')
        self._metadata['iso'] = str(exif.get('EXIF ISOSpeedRatings', None))
        self._metadata['camera'] = str(exif.get('Image Model', None)).strip()
        self._metadata['focal_length'] = fractions.Fraction(str(exif.get('EXIF FocalLength', 0)))
        self._metadata['f_number'] = fractions.Fraction(str(exif.get('EXIF FNumber', 0)))
        self._metadata['datetime'] = str(exif.get('Image DateTime', None))
        self._metadata['maker'] = str(exif.get('Image Make', None))
        self._metadata['note'] = str(exif.get('EXIF MakerNote', None)) # Useless fo far ...

    def _check_channels(self, channels, err_msg):
        channels = self.CHANNELS if channels is None else channels
        if 'G' in channels:
            raise NotImplementedError(err_msg)

    def _trim(self, raw_pixels, roi):
        if roi:
            y1 = roi.y1  
            y2 = roi.y2
            x1 = roi.x1 
            x2 = roi.x2
            raw_pixels = raw_pixels[y1:y2, x1:x2]  # Extract ROI 
        return raw_pixels


    def _to_stack(self, initial_list, channels):
        if channels is None:
            output_list = initial_list
        else:
            output_list = list()
            for ch in channels:
                if ch == 'G':
                    # This assumes that initial list is a pixel array list
                    aver_green = (initial_list[1] + initial_list[2]) // 2
                    output_list.append(aver_green)
                else:
                    i = self.CHANNELS.index(ch)
                    output_list.append(initial_list[i])
        return np.stack(output_list)


    # ----------
    # Public API 
    # ----------

    def label(self, i):
        return self.LABELS[i]

    def name(self):
        return os.path.basename(self._path)

    def shape(self):
        return self._shape

    def roi(self, n_x0=None, n_y0=None, n_width=1.0, n_heigth=1.0):
        return Rect.from_normalized(self._shape[1], self._shape[0], n_x0, n_y0, n_width, n_heigth)

    def exposure(self):
        '''Useul for image list sorting by exposure time'''
        return self._metadata['exposure']

    def exif(self):
        if self._metadata.get('camera') is None:
            self._exif()
        return self._metadata

    def cfa_pattern(self):
        '''Returns the Bayer pattern as RGGB, BGGR, GRBG, GBRG strings'''
        if self._color_desc is None:
            self._img()
        if self._color_desc != 'RGBG':
            raise UnsupporteCFAError(self._color_desc)
        return self._cfa

    def saturation_levels(self, channels=None):
        self._check_channels(channels, err_msg="saturation_levels on G=(G1+G2)/2 channel not available")
        if self._white_levels is None:
            self._img()
        if self._white_levels is None:
            raise NotImplementedError("saturation_levels for this image not available using LibRaw")
        return [self._white_levels[self.CHANNELS.index(ch)] for ch in channels]

    def black_levels(self, channels=None):
        self._check_channels(channels, err_msg="black_levels on G=(G1+G2)/2 channel not available")
        if self._biases is None:
            self._img()
        return [self._biases[self.CHANNELS.index(ch)] for ch in channels]

    def debayered(self, roi=None, channels=None):
        '''Get a stack of Bayer colour planes selected by the channels sequence'''
        with rawpy.imread(self._path) as img:
            self._read_img_metadata(img)
            cfa_pattern = self._cfa
            raw_pixels_list = list()
            for channel in self.CHANNELS:
                x = self.CFA_OFFSETS[cfa_pattern][channel]['x']
                y = self.CFA_OFFSETS[cfa_pattern][channel]['y']
                raw_pixels = img.raw_image[y::2, x::2].copy() # This is the real debayering thing
                raw_pixels = self._trim(raw_pixels, roi)
                raw_pixels_list.append(raw_pixels)
        return self._to_stack(raw_pixels_list, channels)
        

    def statistics(self, roi=None, channels=None):
        '''In-place statistics calculation for RPi Zero'''
        self._check_channels(channels, err_msg="In-place statistics on G=(G1+G2)/2 channel not available")
        with rawpy.imread(self._path) as img:
            # very imporatnt to be under the image context manager
            # when doing manipulations on img.raw_image
            self._read_img_metadata(img)
            cfa_pattern = self._cfa
            stats_list = list()
            for channel in channels:
                x = self.CFA_OFFSETS[cfa_pattern][channel]['x']
                y = self.CFA_OFFSETS[cfa_pattern][channel]['y']
                raw_pixels = img.raw_image[y::2, x::2]
                raw_pixels = self._trim(raw_pixels, roi)
                average, stddev = round(raw_pixels.mean(),1), round(raw_pixels.std(),3)
                stats_list.append([average, stddev])
        return self._to_stack(stats_list, channels)


class SimulatedDarkImage(RawImage):

    def __init__(self, path, dk_current=1.0, rd_noise=1.0):
        super().__init__(path)
        self._dk_current = dk_current
        self._rd_noise = rd_noise

    def debayered(self, roi=None, channels=None):
        '''Get a stack of Bayer colour planes selected by the channels sequence'''
        self._check_channels(channels, err_msg="In-place statistics on G=(G1+G2)/2 channel not available")
        with rawpy.imread(self._path) as img:
            self._read_img_metadata(img)
            raw_pixels_list = list()
            rng = np.random.default_rng()
            shape = (self._shape[0]//2, self._shape[1]//2)
            dark = [self._dk_current * self.exposure() for ch in self.CHANNELS]
            log.info("DARK Curent is %s", dark)
            for i, channel in enumerate(self.CHANNELS):
                raw_pixels = self._biases[i] + dark[i]+ self._rd_noise * rng.standard_normal(size=shape)
                raw_pixels = np.asarray(raw_pixels, dtype=np.uint16)
                raw_pixels = self._trim(raw_pixels, roi)
                raw_pixels_list.append(raw_pixels)
        return self._to_stack(raw_pixels_list, channels)


# Convenience function when plotting titles
def imageset_metadata(path, x0, y0, width, height, channels):
    '''returns common metadata for all the image set with different exposure times'''
    image = RawImage(path)
    roi = image.roi(x0, y0, width, height)
    exif = image.exif()
    return {
        'camera': exif['camera'],
        'iso': exif['iso'],
        'roi': roi,
        'cols': image.shape()[0],
        'rows': image.shape()[1],
        'maker': exif['maker'],
        'channels': channels
    }