"""
Module to read yaml config file and return python object after input parameter consistency is first checked
"""

import yaml
import os.path
from collections.abc import Iterable
from .imgIO import get_sub_dirs


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

        self.dataset    = Dataset()
        self.processing = Processing()
        self.plotting   = Plotting()

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
        self.dataset.data_dir      = self.yaml_config['dataset']['data_dir']
        self.dataset.processed_dir = self.yaml_config['dataset']['processed_dir']
        # self.dataset.inst_matrix_dir = self.yaml_config['dataset']['inst_matrix_dir']

        for (key, value) in self.yaml_config['dataset'].items():
            if key == 'samples':
                if value == 'all':
                    self.dataset.samples = get_sub_dirs(self.dataset.data_dir)
                else:
                    self.dataset.samples = value
            elif key == 'positions':
                self.dataset.positions = value
            elif key == 'ROI':
                self.dataset.ROI = value
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
                    if 'Phase2D' in value or 'Phase_semi3D' in value  or 'Phase3D' in value:
                        phase_processing = True
                    else:
                        phase_processing = False
                elif key == 'circularity':
                    self.processing.circularity = value
                elif key == 'calibration_scheme':
                    self.processing.calibration_scheme = value
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
                elif key == 'local_fit_order':
                    self.processing.local_fit_order = value
                elif key == 'binning':
                    self.processing.binning = value
                elif key == 'use_gpu':
                    self.processing.use_gpu = value
                elif key == 'gpu_id':
                    self.processing.gpu_id = value
                elif key == 'pixel_size':
                    self.processing.pixel_size = value
                elif key == 'magnification':
                    self.processing.magnification = value
                elif key == 'NA_objective':
                    self.processing.NA_objective = value
                elif key == 'NA_condenser':
                    self.processing.NA_condenser = value
                elif key == 'n_objective_media':
                    self.processing.n_objective_media = value
                elif key == 'focus_zidx':
                    self.processing.focus_zidx = value
                elif key == 'phase_denoiser_2D':
                    self.processing.phase_denoiser_2D = value
                elif key == 'Tik_reg_abs_2D':
                    self.processing.Tik_reg_abs_2D = value
                elif key == 'Tik_reg_ph_2D':
                    self.processing.Tik_reg_ph_2D = value
                elif key == 'rho_2D':
                    self.processing.rho_2D = value
                elif key == 'itr_2D':
                    self.processing.itr_2D = value
                elif key == 'TV_reg_abs_2D':
                    self.processing.TV_reg_abs_2D = value
                elif key == 'TV_reg_ph_2D':
                    self.processing.TV_reg_ph_2D = value
                elif key == 'phase_denoiser_3D':
                    self.processing.phase_denoiser_3D = value
                elif key == 'rho_3D':
                    self.processing.rho_3D = value
                elif key == 'itr_3D':
                    self.processing.itr_3D = value
                elif key == 'Tik_reg_ph_3D':
                    self.processing.Tik_reg_ph_3D = value
                elif key == 'TV_reg_ph_3D':
                    self.processing.TV_reg_ph_3D = value
                elif key == 'pad_z':
                    self.processing.pad_z = value
                else:
                    raise NameError('Unrecognized configfile field:{}, key:{}'.format('processing', key))
                    
            if phase_processing:
                
                assert self.processing.pixel_size is not None, \
                "pixel_size (camera pixel size) has to be specified to run phase reconstruction"
                
                assert self.processing.magnification is not None, \
                "magnification (microscope magnification) has to be specified to run phase reconstruction"
                
                assert self.processing.NA_objective is not None, \
                "NA_objective (numerical aperture of the objective) has to be specified to run phase reconstruction"
                
                assert self.processing.NA_condenser is not None, \
                "NA_condenser (numerical aperture of the condenser) has to be specified to run phase reconstruction"
                
                assert self.processing.n_objective_media is not None, \
                "n_objective_media (refractive index of the immersing media) has to be specified to run phase reconstruction"
                
                assert self.processing.n_objective_media >= self.processing.NA_objective and self.processing.n_objective_media >= self.processing.NA_condenser, \
                "n_objective_media (refractive index of the immersing media) has to be larger than the NA of the objective and condenser"
                
                assert self.processing.n_slice_local_bg == 'all', \
                "n_slice_local_bg has to be 'all' in order to run phase reconstruction properly"
                
                assert self.dataset.z_slices[0] == 'all', \
                "z_slices has to be 'all' in order to run phase reconstruction properly"
                
            
            if 'Phase2D' in self.processing.output_channels:
                
                assert self.processing.focus_zidx is not None, \
                "focus_zidx has to be specified to run 2D phase reconstruction"
                    
                
                

        if 'plotting' in self.yaml_config:
            for (key, value) in self.yaml_config['plotting'].items():
                if key == 'normalize_color_images':
                    self.plotting.normalize_color_images = value
                elif key == 'retardance_scaling':
                    self.plotting.retardance_scaling = float(value)
                elif key == 'transmission_scaling':
                    self.plotting.transmission_scaling = float(value)
                elif key == 'phase_2D_scaling':
                    self.plotting.phase_2D_scaling = float(value)
                elif key == 'phase_3D_scaling':
                    self.plotting.phase_3D_scaling = float(value)
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
        self._data_dir      = []
        self._samples       = []
        self._positions     = ['all']
        self._ROI           = None
        self._z_slices      = ['all']
        self._timepoints    = ['all']
        self._background    = []
    
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
    def ROI(self):
        return self._ROI
    
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
    
    @ROI.setter
    def ROI(self, value):   
        assert isinstance(value, list), \
        "ROI should be a list contains [n_start_y, n_start_x, Ny, Nx]"
        self._ROI = value
        
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
                                'Phase2D', 'Phase_semi3D', 'Phase3D',
                                'Orientation_x', 'Orientation_y',
                                'Pol_State_0', 'Pol_State_1', 'Pol_State_2', 'Pol_State_3', 'Pol_State_4',
                                'Stokes_0', 'Stokes_1', 'Stokes_2', 'Stokes_3',
                                'Stokes_0_sm', 'Stokes_1_sm', 'Stokes_2_sm', 'Stokes_3_sm',
                                '405', '488', '568', '640', 'ex561em700',
                                'Retardance+Orientation', 'Polarization+Orientation', 
                                'Brightfield+Retardance+Orientation',
                                'Retardance+Fluorescence', 'Retardance+Fluorescence_all']  
    _allowed_circularity_values = ['rcp', 'lcp']
    _allowed_calibration_schemes = ['5-State', '4-State', '4-State Extinction', 'Custom Instrument Matrix']
    _allowed_background_correction_values = ['None', 'Input', 'Local_filter', 'Local_fit', 'Local_defocus', 'Auto']
    _allowed_phase_denoiser_values = ['Tikhonov', 'TV']
    
    def __init__(self):
        self._output_channels       = ['Brightfield', 'Retardance', 'Orientation', 'Polarization']
        self._circularity           = 'rcp'
        self._calibration_scheme    = None
        self._background_correction = 'None'
        self._flatfield_correction  = False
        self._azimuth_offset        = 0
        self._separate_positions    = True
        self._n_slice_local_bg      = 'all'
        self._local_fit_order       = 2
        self._binning               = 1
        
        self._use_gpu = False
        self._gpu_id  = 0
        
        self._pixel_size    = None
        self._magnification = None
        self._NA_objective  = None
        self._NA_condenser  = None
        self._n_objective_media       = None
        self._focus_zidx    = None
        
        self._phase_denoiser_2D = 'Tikhonov'

        self._Tik_reg_abs_2D = 1e-6
        self._Tik_reg_ph_2D  = 1e-6

        self._rho_2D        = 1
        self._itr_2D        = 50
        self._TV_reg_abs_2D = 1e-3
        self._TV_reg_ph_2D  = 1e-5

        self._phase_denoiser_3D = 'Tikhonov'

        self._rho_3D        = 1e-3
        self._itr_3D        = 50
        self._Tik_reg_ph_3D = 1e-4
        self._TV_reg_ph_3D  = 5e-5
        
        self._pad_z = 0
        

    @property
    def output_channels(self):
        return self._output_channels
    
    @property
    def circularity(self):
        return self._circularity

    @property
    def calibration_scheme(self):
        return self._calibration_scheme
    
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

    @property
    def local_fit_order(self):
        return self._local_fit_order

    @property
    def binning(self):
        return self._binning
    
    @property
    def use_gpu(self):
        return self._use_gpu
    
    @property
    def gpu_id(self):
        return self._gpu_id
    
    @property
    def pixel_size(self):
        return self._pixel_size
    
    @property
    def magnification(self):
        return self._magnification
    
    @property
    def NA_objective(self):
        return self._NA_objective
    
    @property
    def NA_condenser(self):
        return self._NA_condenser
    
    @property
    def n_objective_media(self):
        return self._n_objective_media
    
    @property
    def focus_zidx(self):
        return self._focus_zidx
    
    @property
    def phase_denoiser_2D(self):
        return self._phase_denoiser_2D
    
    @property
    def Tik_reg_abs_2D(self):
        return self._Tik_reg_abs_2D
    
    @property
    def Tik_reg_ph_2D(self):
        return self._Tik_reg_ph_2D
    
    @property
    def rho_2D(self):
        return self._rho_2D
    
    @property
    def itr_2D(self):
        return self._itr_2D
    
    @property
    def TV_reg_abs_2D(self):
        return self._TV_reg_abs_2D
    
    @property
    def TV_reg_ph_2D(self):
        return self._TV_reg_ph_2D
    
    @property
    def phase_denoiser_3D(self):
        return self._phase_denoiser_3D
    
    @property
    def rho_3D(self):
        return self._rho_3D
    
    @property
    def itr_3D(self):
        return self._itr_3D
    
    @property
    def Tik_reg_ph_3D(self):
        return self._Tik_reg_ph_3D
    
    @property
    def TV_reg_ph_3D(self):
        return self._TV_reg_ph_3D
    
    @property
    def pad_z(self):
        return self._pad_z
    

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

    @calibration_scheme.setter
    def calibration_scheme(self, value):
        assert value in self._allowed_calibration_schemes, "{} is not an allowed calibration scheme".format(value)
        self._calibration_scheme = value
        
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

    @local_fit_order.setter
    def local_fit_order(self, value):
        assert isinstance(value, int) and value >= 0, \
            "local_fit_order must be a non-negative integer"
        self._local_fit_order = value

    @binning.setter
    def binning(self, value):
        assert isinstance(value, int) and value > 0, \
            "binning must be a positive integer"
        self._binning = value
        
    @use_gpu.setter
    def use_gpu(self, value):   
        assert isinstance(value, bool), "use_gpu must be boolean"
        self._use_gpu = value
        
    @gpu_id.setter
    def gpu_id(self, value):
        assert isinstance(value, int) and value >= 0, \
            "gpu_id must be a non-negative integer"
        self._gpu_id = value
    
    @pixel_size.setter
    def pixel_size(self, value):
        assert isinstance(value, (int, float)) and value > 0, \
            "pixel_size must be a number > 0"
        self._pixel_size = value 
    
    @magnification.setter
    def magnification(self, value):   
        assert isinstance(value, (int, float)) and value > 0, \
            "magnification must be a number > 0"
        self._magnification = value
        
    @NA_objective.setter
    def NA_objective(self, value):   
        assert isinstance(value, (int, float)) and 0 < value < 2, \
            "NA_objective must be a number in range [0, 2]"
        self._NA_objective = value
        
    @NA_condenser.setter
    def NA_condenser(self, value):   
        assert isinstance(value, (int, float)) and 0 < value < 2, \
            "NA_condenser must be a number in range [0, 2]"
        self._NA_condenser = value
        
    @n_objective_media.setter
    def n_objective_media(self, value):
        assert isinstance(value, (int, float)) and value >=1, \
            "n_objective_media must be a number >= 1"
        self._n_objective_media = value
        
    @focus_zidx.setter
    def focus_zidx(self, value):
        assert isinstance(value, int) and value >= 0, \
            "focus_zidx must be a non-negative integer"
        self._focus_zidx = value
    
    @phase_denoiser_2D.setter
    def phase_denoiser_2D(self, value):     
        assert value in self._allowed_phase_denoiser_values, "{} is not an allowed 2D phase denoiser setting".format(value)
        self._phase_denoiser_2D = value
    
    @Tik_reg_abs_2D.setter
    def Tik_reg_abs_2D(self, value):   
        assert isinstance(value, (int, float)) and value > 0, \
            "Tik_reg_abs_2D must be a number > 0"
        self._Tik_reg_abs_2D = value
        
    @Tik_reg_ph_2D.setter
    def Tik_reg_ph_2D(self, value):   
        assert isinstance(value, (int, float)) and value > 0, \
            "Tik_reg_ph_2D must be a number > 0"
        self._Tik_reg_ph_2D = value
        
    @rho_2D.setter
    def rho_2D(self, value):   
        assert isinstance(value, (int, float)) and value > 0, \
            "rho_2D must be a number > 0"
        self._rho_2D = value
    
    @itr_2D.setter
    def itr_2D(self, value):   
        assert isinstance(value, int) and value > 0, \
            "itr_2D must be a non-negative integer"
        self._itr_2D = value
        
    @TV_reg_abs_2D.setter
    def TV_reg_abs_2D(self, value):   
        assert isinstance(value, (int, float)) and value > 0, \
            "TV_reg_abs_2D must be a number > 0"
        self._TV_reg_abs_2D = value
        
    @TV_reg_ph_2D.setter
    def TV_reg_ph_2D(self, value):   
        assert isinstance(value, (int, float)) and value > 0, \
            "TV_reg_ph_2D must be a number > 0"
        self._TV_reg_ph_2D = value
    
    @phase_denoiser_3D.setter
    def phase_denoiser_3D(self, value):     
        assert value in self._allowed_phase_denoiser_values, "{} is not an allowed 3D phase denoiser setting".format(value)
        self._phase_denoiser_3D = value
        
    @rho_3D.setter
    def rho_3D(self, value):   
        assert isinstance(value, (int, float)) and value > 0, \
            "rho_3D must be a number > 0"
        self._rho_3D = value
    
    @itr_3D.setter
    def itr_3D(self, value):   
        assert isinstance(value, int) and value > 0, \
            "itr_3D must be a non-negative integer"
        self._itr_3D = value
        
    @Tik_reg_ph_3D.setter
    def Tik_reg_ph_3D(self, value):   
        assert isinstance(value, (int, float)) and value > 0, \
            "Tik_reg_ph_3D must be a number > 0"
        self._Tik_reg_ph_3D = value
        
    @TV_reg_ph_3D.setter
    def TV_reg_ph_3D(self, value):   
        assert isinstance(value, (int, float)) and value > 0, \
            "TV_reg_ph_3D must be a number > 0"
        self._TV_reg_ph_3D = value
        
    @pad_z.setter
    def pad_z(self, value):   
        assert isinstance(value, int) and value >= 0, \
            "pad_z must be an integer >= 0"
        self._pad_z = value

    def __repr__(self):
        out = str(self.__class__) + '\n'
        for (key, value) in self.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        return out


class Plotting:
    def __init__(self):
        self.normalize_color_images = True
        self.transmission_scaling   = 1E4
        self.retardance_scaling     = 1E3
        self.phase_2D_scaling       = 1
        self.phase_3D_scaling       = 1
        self.save_birefringence_fig = False
        self.save_stokes_fig        = False
        self.save_polarization_fig  = False
        self.save_micromanager_fig  = False
    
    def __repr__(self):
        out = str(self.__class__) + '\n'
        for (key, value) in self.__dict__.items():
            out = out + '{}: {}\n'.format(key.strip('_'),value)
        return out
