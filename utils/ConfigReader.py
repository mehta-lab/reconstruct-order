#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:31:41 2019

@author: ivan.ivanov
"""
import yaml
import os.path
from utils.imgIO import GetSubDirName

class ConfigReader:   
    def __init__(self):
        self.dataset = Dataset()
        self.processing = Processing()
        self.plotting = Plotting()
        
    def read_config(self,path):
        with open(path, 'r') as f:
            self.yaml_config = yaml.load(f)
            
        if 'dataset' in self.yaml_config:
            for (key, value) in self.yaml_config['dataset'].items():
                if key == 'processed_dir':
                    self.dataset.processed_dir = value
                elif key == 'data_dir':
                    self.dataset.data_dir = value
                elif key == 'samples':
                    self.dataset.samples = value
                elif key == 'positions':
                    self.dataset.positions = value
                elif key == 'background':
                    self.dataset.background = value
                else:
                    raise NameError('Unrecognized parameter {} passed'.format(key))
        else:
            raise IOError('dataset is a required field in the config yaml file')
             
        if 'processing' in self.yaml_config:
            for (key, value) in self.yaml_config['processing'].items():
                if key == 'output_channels':
                    self.processing.output_channels = value
                elif key == 'circularity':
                    self.processing.circularity = value
                elif key == 'background_correction':
                    self.processing.background_correction = value
                elif key == 'flatfield_correction':
                    self.processing.flatfield_correction = value
                elif key == 'azimuth_offset':
                    self.processing.azimuth_offset = value
                elif key == 'separate_positions':
                    self.processing.separate_positions = value
                else:
                    raise NameError('Unrecognized parameter {} passed'.format(key))
         
        if 'plotting' in self.yaml_config:
            for (key, value) in self.yaml_config['plotting'].items():
                if key == 'normalize_color_images':
                    self.plotting.normalize_color_images = value
                elif key == 'save_birefringence_fig':
                    self.plotting.save_birefringence_fig = value
                elif key == 'save_stokes_fig':
                    self.plotting.save_stokes_fig = value
                elif key == 'save_polarization_fig':
                    self.plotting.save_polarization_fig = value
                elif key == 'save_micromanager_fig':
                    self.plotting.save_micromanager_fig = value
                else:
                    raise NameError('Unrecognized parameter {} passed'.format(key))
                    
        if self.dataset.background and 'processing' not in self.yaml_config \
            or 'background_correction' not in self.yaml_config['processing']:
            self.processing.background_correction = 'Input'
                    
        assert self.dataset.processed_dir, \
            'Please provde processed_dir in config file'
        assert self.dataset.data_dir, \
            'Please provde data_dir in config file'
        assert self.dataset.samples, \
            'Please provde samples in config file'
            
        if self.dataset.samples[0] == 'all':
            self.dataset.samples = GetSubDirName(self.dataset.data_dir)         
            
        if not any(isinstance(i, list) for i in self.dataset.positions):
            self.dataset.positions = [self.dataset.positions]*len(self.dataset.samples)
            
        if len(self.dataset.background) == 1:
            self.dataset.background = self.dataset.background * len(self.dataset.samples)
                
        assert len(self.dataset.samples) == len(self.dataset.background) == len(self.dataset.positions), \
            'Length of the background directory list must be one or same as sample directory list'
            
    def write_config(self,path):
        with open(path, 'w') as f:
            yaml.dump(self.yaml_config, f, default_flow_style=False)
            
    def __repr__(self):
        out = str(self.__class__) + '\n'
        for (key, value) in self.dataset.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        for (key, value) in self.processing.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        for (key, value) in self.plotting.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        return out


class Dataset:    
    def __init__(self):
        self.n_samples = 0
        self._processed_dir = []
        self._data_dir = []
        self._samples = []
        self._positions = 'all'
        self._background = []
    
    @property
    def processed_dir(self):
        return self._processed_dir
    
    @property
    def data_dir(self):
        return self._data_dir
    
    @property
    def samples(self):
        return self._samples
    
    @property
    def positions(self):
        return self._positions
    
    @property
    def background(self):
        return self._background
    
    @processed_dir.setter
    def processed_dir(self, value):
        assert os.path.exists(value), 'processed_dir path {} does not exist'.format(value)
        self._processed_dir = value
        
    @data_dir.setter
    def data_dir(self, value):
        assert os.path.exists(value), 'data_dir path {} does not exist'.format(value)
        self._data_dir = value
    
    @samples.setter
    def samples(self, value):
        if not isinstance(value, list):
            value = [value]
        for sm in value:
            assert os.path.exists(os.path.join(self.data_dir,sm)), 'sample directory {} does not exist'.format(sm)
        self.n_samples = len(value)
        self._samples = value
        
    @positions.setter
    def positions(self, value):   
        if not isinstance(value, list):
            value = [value]
        self._positions = value
        
    @background.setter
    def background(self, value):
        if not isinstance(value, list):
            value = [value]
        for bg in value:
            assert os.path.exists(os.path.join(self.data_dir,bg)), 'background directory {} does not exist'.format(bg)
        self._background = value
        
    def __repr__(self):
        out = str(self.__class__) + '\n'
        for (key, value) in self.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        return out
        
class Processing:        
    _allowed_output_channels = ['Transmission', 'Retardance', 'Orientation', 'Polarization',
                                'Orientation_x', 'Orientation_y',
                                'Pol_State_0', 'Pol_State_1', 'Pol_State_2', 'Pol_State_3', 'Pol_State_4',
                                'Stokes_0', 'Stokes_1', 'Stokes_2', 'Stokes_3',
                                '405', '488', '568', '640',
                                'Retardance+Orientation', 'Polarization+Orientation', 
                                'Transmission+Retardance+Orientation',
                                'Retardance+Fluorescence', 'Retardance+Fluorescence_all']  
    _allowed_circularity_values = ['rcp', 'lcp']
    _allowed_background_correction_values = ['None', 'Input', 'Local_filter', 'Local_defocus', 'Auto']
    
    def __init__(self):
        self._output_channels = ['Transmission', 'Retardance', 'Orientation', 'Polarization']
        self._circularity = 'rcp'
        self._background_correction = 'None'
        self._flatfield_correction = False
        self._azimuth_offset = 0
        self._separate_positions = True
        
    @property
    def output_channels(self):
        return self._output_channels
    
    @property
    def circularity(self):
        return self._circularity
    
    @property
    def background_correction(self):
        return self._background_correction
    
    @property
    def flatfield_correction(self):
        return self._flatfield_correction
    
    @property
    def azimuth_offset(self):
        return self._azimuth_offset
    
    @property
    def separate_positions(self):
        return self._separate_positions
    
    @output_channels.setter
    def output_channels(self, value):     
        if not isinstance(value, list):
            value = [value]
        for val in value:
            assert val in self._allowed_output_channels, "{} is not an allowed output channel".format(val)
        self._output_channels = value
        
    @circularity.setter
    def circularity(self, value):     
        assert value in self._allowed_circularity_values, "{} is not an allowed circularity setting".format(value)
        self._circularity = value
        
    @background_correction.setter
    def background_correction(self, value):     
        assert value in self._allowed_background_correction_values, "{} is not an allowed bg_correction setting".format(value)
        self._background_correction = value
        
    @flatfield_correction.setter
    def flatfield_correction(self, value):   
        assert isinstance(value, bool), "flatfield_correction must be boolean"
        self._flatfield_correction = value
        
    @azimuth_offset.setter
    def azimuth_offset(self, value):   
        # TODO: Check that input value if right type
        self._azimuth_offset = value
        
    @separate_positions.setter
    def separate_positions(self, value):   
        assert isinstance(value, bool), "separate_positions must be boolean"
        self._separate_positions = value
        
    def __repr__(self):
        out = str(self.__class__) + '\n'
        for (key, value) in self.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        return out
    
class Plotting: 
    def __init__(self):
        self.normalize_color_images = True
        self.save_birefringence_fig = False
        self.save_stokes_fig = False
        self.save_polarization_fig = False
        self.save_micromanager_fig = False
    
    def __repr__(self):
        out = str(self.__class__) + '\n'
        for (key, value) in self.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        return out
