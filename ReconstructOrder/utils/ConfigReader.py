"""
Module to read yaml config file and return python object after input parameter consistency is first checked
"""

import yaml
import os.path
from collections.abc import Iterable
from .imgIO import GetSubDirName


class ConfigReader:
    """
    Parser of the yaml ReconstructOrder configuration file

    The reader checks that the user-provided data directories exist and that
    the provided parameters are within the allowed values. See
    config/config_example.yml for description of the parameters and expected
    input values.

    """

    def __init__(self, path=[]):
        """
        ConfigReader __init__ method.

        Initializes the ConfigReader object with default parameters.
        Optionally, if the path to the config file is provided in `path`, the
        config yaml file is read at initialization. Otherwise, the method
        `read_config` needs to be called.

        Parameters
        ----------
        path : str, optional
            Path to yaml config file.

        """

        self.dataset = Dataset()
        self.processing = Processing()
        self.plotting = Plotting()

        if path:
            self.read_config(path)
        
    def read_config(self,path):
        """
        Reads yaml config file provided in `path`

        Parameters
        ----------
        path: str
            Path to yaml config file

        """

        with open(path, 'r') as f:
            self.yaml_config = yaml.load(f)
            
        assert 'dataset' in self.yaml_config, \
            'dataset is a required field in the config yaml file'
        assert 'data_dir' in self.yaml_config['dataset'], \
            'Please provide data_dir in config file'
        assert 'processed_dir' in self.yaml_config['dataset'], \
            'Please provide processed_dir in config file'
        assert 'samples'  in self.yaml_config['dataset'], \
            'Please provide samples in config file'

        # Assign data_dir and processed_dir first to be able to check sample
        # and background directories
        self.dataset.data_dir = self.yaml_config['dataset']['data_dir']
        self.dataset.processed_dir = self.yaml_config['dataset']['processed_dir']

        for (key, value) in self.yaml_config['dataset'].items():
            if key == 'samples':
                if value == 'all':
                    self.dataset.samples = GetSubDirName(self.dataset.data_dir)
                else:
                    self.dataset.samples = value
            elif key == 'positions':
                self.dataset.positions = value
            elif key == 'z_slices':
                self.dataset.z_slices = value
            elif key == 'timepoints':
                self.dataset.timepoints = value
            elif key == 'background':
                self.dataset.background = value
            elif key not in ('data_dir', 'processed_dir'):
                raise NameError('Unrecognized configfile field:{}, key:{}'.format('dataset', key))
             
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
                elif key == 'n_slice_local_bg':
                    self.processing.n_slice_local_bg = value
                else:
                    raise NameError('Unrecognized configfile field:{}, key:{}'.format('processing', key))

        if 'plotting' in self.yaml_config:
            for (key, value) in self.yaml_config['plotting'].items():
                if key == 'normalize_color_images':
                    self.plotting.normalize_color_images = value
                elif key == 'retardance_scaling':
                    self.plotting.retardance_scaling = value
                elif key == 'transmission_scaling':
                    self.plotting.transmission_scaling = value
                elif key == 'save_birefringence_fig':
                    self.plotting.save_birefringence_fig = value
                elif key == 'save_stokes_fig':
                    self.plotting.save_stokes_fig = value
                elif key == 'save_polarization_fig':
                    self.plotting.save_polarization_fig = value
                elif key == 'save_micromanager_fig':
                    self.plotting.save_micromanager_fig = value
                else:
                    raise NameError('Unrecognized configfile field:{}, key:{}'.format('plotting', key))

        self.__check_input_consistency__()

    def __check_input_consistency__(self):
        if self.dataset.background and 'processing' not in self.yaml_config \
            or 'background_correction' not in self.yaml_config['processing']:
            self.processing.background_correction = 'Input'
            
        if not any(isinstance(i, list) for i in self.dataset.positions):
            self.dataset.positions = [self.dataset.positions]*len(self.dataset.samples)
        else:
            assert all(isinstance(i, list) for i in self.dataset.positions),\
            'Positions input must be a list of lists'

        if not any(isinstance(i, list) for i in self.dataset.z_slices):
            self.dataset.z_slices = [self.dataset.z_slices]*len(self.dataset.samples)
        else:
            assert all(isinstance(i, list) for i in self.dataset.z_slices),\
            'z_slices input must be a list of lists'

        if not any(isinstance(i, list) for i in self.dataset.timepoints):
            self.dataset.timepoints = [self.dataset.timepoints]*len(self.dataset.samples)
        else:
            assert all(isinstance(i, list) for i in self.dataset.timepoints),\
            'timepoints input must be a list of lists'
            
        if len(self.dataset.background) == 1:
            self.dataset.background = self.dataset.background * len(self.dataset.samples)
                
        assert len(self.dataset.samples) == len(self.dataset.background) == len(self.dataset.positions) == \
                len(self.dataset.z_slices) == len(self.dataset.timepoints), \
                'Please provide equal number of samples and lists with corresponding background, positions, z_slices, and timepoints'

    def write_config(self,path):
        """
        Writes the fully-parameterized config file in output location given by `path`

        Parameters
        ----------
        path: str
            Path to output file

        """

        config_out = {'dataset':{key.strip('_'):value for (key,value) in self.dataset.__dict__.items()},
                      'processing':{key.strip('_'):value for (key,value) in self.processing.__dict__.items()},
                      'plotting':{key.strip('_'):value for (key,value) in self.plotting.__dict__.items()}}
        with open(path, 'w') as f:
            yaml.dump(config_out, f, default_flow_style=False)
            
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
        self._processed_dir = []
        self._data_dir = []
        self._samples = []
        self._positions = ['all']
        self._z_slices = ['all']
        self._timepoints = ['all']
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
    def z_slices(self):
        return self._z_slices

    @property
    def timepoints(self):
        return self._timepoints

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
        self._samples = value
        
    @positions.setter
    def positions(self, value):   
        if not isinstance(value, list):
            value = [value]
        self._positions = value
        
    @z_slices.setter
    def z_slices(self, value):
        if isinstance(value, Iterable) and value != 'all':
            value = list(value)
        else:
            value = [value]
        self._z_slices = value

    @timepoints.setter
    def timepoints(self, value):
        if isinstance(value, Iterable) and value != 'all':
            value = list(value)
        else:
            value = [value]
        self._timepoints = value

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
    _allowed_output_channels = ['Brightfield', 'Brightfield_computed', 'Retardance', 'Orientation', 'Polarization',
                                'Orientation_x', 'Orientation_y',
                                'Pol_State_0', 'Pol_State_1', 'Pol_State_2', 'Pol_State_3', 'Pol_State_4',
                                'Stokes_0', 'Stokes_1', 'Stokes_2', 'Stokes_3',
                                '405', '488', '568', '640',
                                'Retardance+Orientation', 'Polarization+Orientation', 
                                'Brightfield+Retardance+Orientation',
                                'Retardance+Fluorescence', 'Retardance+Fluorescence_all']  
    _allowed_circularity_values = ['rcp', 'lcp']
    _allowed_background_correction_values = ['None', 'Input', 'Local_filter', 'Local_defocus', 'Auto']
    
    def __init__(self):
        self._output_channels = ['Brightfield', 'Retardance', 'Orientation', 'Polarization']
        self._circularity = 'rcp'
        self._background_correction = 'None'
        self._flatfield_correction = False
        self._azimuth_offset = 0
        self._separate_positions = True
        self._n_slice_local_bg = 'all'

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

    @property
    def n_slice_local_bg(self):
        return self._n_slice_local_bg

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
        assert isinstance(value, (int, float)) and 0 <= value <= 180, \
            "azimuth_offset must be a number in range [0, 180]"
        self._azimuth_offset = value
        
    @separate_positions.setter
    def separate_positions(self, value):   
        assert isinstance(value, bool), "separate_positions must be boolean"
        self._separate_positions = value

    @n_slice_local_bg.setter
    def n_slice_local_bg(self, value):
        assert isinstance(value, int) or value == 'all',\
            "n_slice_local_bg must be integer or 'all'"
        self._n_slice_local_bg = value

    def __repr__(self):
        out = str(self.__class__) + '\n'
        for (key, value) in self.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        return out


class Plotting:
    def __init__(self):
        self.normalize_color_images = True
        self.transmission_scaling = 1E4
        self.retardance_scaling = 1E3
        self.save_birefringence_fig = False
        self.save_stokes_fig = False
        self.save_polarization_fig = False
        self.save_micromanager_fig = False
    
    def __repr__(self):
        out = str(self.__class__) + '\n'
        for (key, value) in self.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        return out
