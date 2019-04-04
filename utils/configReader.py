#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:31:41 2019

@author: ivan.ivanov
"""
import yaml
import os.path
from imgIO import GetSubDirName

class ConfigReader:   
    def __init__(self):
        self.dataset = Dataset()
        self.processing = Processing()
        self.plotting = Plotting()
        
    def read_config(self,path):
        with open(path, 'r') as f:
            config = yaml.load(f)
            
        if 'dataset' in config:
            for (key, value) in config['dataset'].items():
                if key == 'RawDataPath':
                    self.dataset.RawDataPath = value
                if key == 'ProcessedPath':
                    self.dataset.ProcessedPath = value
                if key == 'ImgDir':
                    self.dataset.ImgDir = value
                if key == 'SmDir':
                    self.dataset.SmDir = value
                if key == 'BgDir':
                    self.dataset.BgDir = value
             
        if 'processing' in config:
            for (key, value) in config['processing'].items():
                if key == 'outputChann':
                    self.processing.outputChann = value
                if key == 'circularity':
                    self.processing.circularity = value
                if key == 'bgCorrect':
                    self.processing.bgCorrect = value
                if key == 'flatField':
                    self.processing.flatField = value
                if key == 'azimuth_offset':
                    self.processing.azimuth_offset = value
                if key == 'PosList':
                    self.processing.PosList = value
                if key == 'separate_pos':
                    self.processing.separate_pos = value
         
        if 'plotting' in config:
            for (key, value) in config['plotting'].items():
                if key == 'norm':
                    self.plotting.norm = value
                if key == 'save_fig':
                    self.plotting.save_fig = value
                if key == 'save_stokes_fig':
                    self.plotting.save_stokes_fig = value
                    
        assert self.dataset.RawDataPath, \
            'Please provde RawDataPath in config file'
        assert self.dataset.ProcessedPath, \
            'Please provde ProcessedPath in config file'
        assert self.dataset.ImgDir, \
            'Please provde ImgDir in config file'
        assert self.dataset.SmDir, \
            'Please provde SmDir in config file'
            
        if self.dataset.SmDir[0] == 'all':
            img_path = os.path.join(self.dataset.RawDataPath, self.dataset.ImgDir)
            self.dataset.SmDir = GetSubDirName(img_path)         
            
        if not any(isinstance(i, list) for i in self.processing.PosList):
            self.processing.PosList = self.processing.PosList*len(self.dataset.SmDir)
            
        if len(self.dataset.BgDir) == 1:
            self.dataset.BgDir = self.dataset.BgDir * len(self.dataset.SmDir)
                
        assert len(self.dataset.SmDir) == len(self.dataset.BgDir) == len(self.processing.PosList), \
            'Length of the background directory list must be one or same as sample directory list'
                
class Dataset:
    RawDataPath = []
    ProcessedPath = []
    ImgDir = []
    _SmDir = []
    _BgDir = []
    
    @property
    def SmDir(self):
        return self._SmDir
    
    @property
    def BgDir(self):
        return self._BgDir
    
    @SmDir.setter
    def SmDir(self, value):
        if not isinstance(value, list):
            value = [value]
        self._SmDir = value
        
    @BgDir.setter
    def BgDir(self, value):
        if not isinstance(value, list):
            value = [value]
        self._BgDir = value
    
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
    _allowed_bgCorrect_values = ['None', 'Input', 'Local_filter', 'Local_defocus', 'Auto']
    
    def __init__(self):
        self._outputChann = ['Transmission', 'Retardance', 'Orientation', 'Scattering']
        self._circularity = 'rcp'
        self._bgCorrect = 'None'
        self._flatField = False
        self._azimuth_offset = 0
        self._PosList = 'all'
        self._separate_pos = True
        
    @property
    def outputChann(self):
        return self._outputChann
    
    @property
    def circularity(self):
        return self._circularity
    
    @property
    def bgCorrect(self):
        return self._bgCorrect
    
    @property
    def flatField(self):
        return self._flatField
    
    @property
    def azimuth_offset(self):
        return self._azimuth_offset
    
    @property
    def PosList(self):
        return self._PosList
    
    @property
    def separate_pos(self):
        return self._separate_pos
    
    @outputChann.setter
    def outputChann(self, value):     
        if not isinstance(value, list):
            value = [value]
        for val in value:
            assert val in self._allowed_output_channels, "{} is not an allowed output channel".format(val)
        self._outputChann = value
        
    @circularity.setter
    def circularity(self, value):     
        assert value in self._allowed_circularity_values, "{} is not an allowed circularity setting".format(value)
        self._circularity = value
        
    @bgCorrect.setter
    def bgCorrect(self, value):     
        assert value in self._allowed_bgCorrect_values, "{} is not an allowed bgCorrect setting".format(value)
        self._bgCorrect = value
        
    @flatField.setter
    def flatField(self, value):   
        assert isinstance(value, bool), "flatField must be boolean"
        self._flatField = value
        
    @azimuth_offset.setter
    def azimuth_offset(self, value):   
        # TODO: Check that input value if right type
        self._azimuth_offset = value
        
    @separate_pos.setter
    def separate_pos(self, value):   
        assert isinstance(value, bool), "separate_pos must be boolean"
        self._separate_pos = value
        
    @PosList.setter
    def PosList(self, value):   
        if not isinstance(value, list):
            value = [value]
        self._PosList = value
    
class Plotting:
    norm = True
    save_fig = False
    save_stokes_fig = False