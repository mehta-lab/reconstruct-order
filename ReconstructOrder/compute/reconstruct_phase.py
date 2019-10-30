import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import uniform_filter

from ..compute.reconstruct_phase_util import *
from ..datastructures import IntensityData, StokesData, PhysicalData


class phase_reconstructor:
    
    
    '''
    
    phase_reconstructor contains methods to compute weak object transfer function 
    and conduct 2D/2.5D/3D phase reconstruction with a through-focus intensity stack
    
    Parameters
    ----------
        img_dim         : tuple
                          shape of the computed 3D space with size of (N, M, N_defocus)
                  
        lambda_illu     : float
                          wavelength of the incident light
        
        ps              : float
                          xy pixel size of the image space
                          
        psz             : float
                          z step size of the image space
        
        NA_obj          : float
                          numerical aperture of the detection objective
        
        NA_illu         : float 
                          numerical aperture of the illumination condenser
        
        focus_idx       : int
                          z index of the focus layer in the stack
                          
        n_objective_media         : float
                          refractive index of the immersing media
        
        phase_deconv    : list
                          a list of string contains the deconvolution dimension on the data
                          '2D'      for 2D phase deconvolution
                          'semi-3D' for 2.5D phase deconvolution
                          '3D'      for 3D phase deconvolution
        
        ph_deconv_layer : int
                          number of layers included for each layer of semi-3D phase reconstruction
        
        pad_z           : int
                          number of z-layers to pad (padded with mean of the stack) for 3D phase reconstruction
        
        use_gpu         : bool
                          option to use gpu or not
        
        gpu_id          : int
                          number refering to which gpu will be used
                  
    
    '''
    
    def __init__(self, img_dim, lambda_illu, ps, psz, NA_obj, NA_illu, focus_idx=None, \
                 n_objective_media=1, phase_deconv=['2D'], ph_deconv_layer=5, pad_z=0, use_gpu=False, gpu_id=0):
        
        # GPU/CPU
        
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        if self.use_gpu:
            try:
                globals()['cp'] = __import__("cupy")
                cp.cuda.Device(self.gpu_id).use()
            except ModuleNotFoundError:
                print("cupy not installed, using CPU instead")
                self.use_gpu = False
            
        
        # Basic parameter 
        self.N, self.M, self.N_defocus   = img_dim
        self.n_objective_media                     = n_objective_media
        self.lambda_illu                 = lambda_illu/n_objective_media
        self.ps                          = ps
        self.psz                         = np.abs(psz)
        self.pad_z                       = pad_z
        self.phase_deconv                = phase_deconv
        self.focus_idx                   = focus_idx
        
        if psz >0 and focus_idx is not None:
            self.z_defocus               = -(np.r_[:self.N_defocus]-focus_idx)*self.psz
        elif psz <0 and focus_idx is not None:
            self.z_defocus               = (np.r_[:self.N_defocus]-focus_idx)*self.psz
        elif psz >0 and focus_idx is None:
            self.z_defocus               = -(np.r_[:self.N_defocus]-self.N_defocus//2)*self.psz
        elif psz <0 and focus_idx is None:
            self.z_defocus               = (np.r_[:self.N_defocus]-self.N_defocus//2)*self.psz
            
        
        self.NA_obj      = NA_obj/n_objective_media
        self.NA_illu     = NA_illu/n_objective_media
        
        # setup coordinate systems
        self.xx, self.yy, self.fxx, self.fyy = gen_coordinate((self.N, self.M), ps)
        
        # detection setup
        self.Pupil_obj     = gen_Pupil(self.fxx, self.fyy, self.NA_obj, self.lambda_illu)
        self.Pupil_support = self.Pupil_obj.copy()
        
        # illumination setup
        self.Source = gen_Pupil(self.fxx, self.fyy, self.NA_illu, self.lambda_illu)
                
        # select either 2D or 3D model for deconvolution
        self.phase_deconv_setup(self.phase_deconv, ph_deconv_layer)
        
        
    def phase_deconv_setup(self, phase_deconv, ph_deconv_layer):
        
        '''
    
        setup transfer functions for phase deconvolution with the corresponding dimensions

        Parameters
        ----------
            phase_deconv    : list
                              a list of string contains the deconvolution dimension on the data
                              '2D'      for 2D phase deconvolution
                              'semi-3D' for 2.5D phase deconvolution
                              '3D'      for 3D phase deconvolution
        
            ph_deconv_layer : int
                              number of layers included for each layer of semi-3D phase reconstruction
                              

        '''
        
        if '2D' in phase_deconv:
            
            Hz_det = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, self.z_defocus, type='Prop')
            self.gen_2D_WOTF(Hz_det)
            
        if 'semi-3D' in phase_deconv:
            
            self.ph_deconv_layer = ph_deconv_layer
            if self.z_defocus[0] - self.z_defocus[1] >0:
                z_deconv = -(np.r_[:self.ph_deconv_layer]-self.ph_deconv_layer//2)*self.psz
            else:
                z_deconv = (np.r_[:self.ph_deconv_layer]-self.ph_deconv_layer//2)*self.psz
            
            Hz_det  = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z_deconv, type='Prop')
            G_fun_z = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z_deconv, type='Green')
            self.gen_semi_3D_WOTF(Hz_det, G_fun_z)
            
        if '3D' in phase_deconv:
            
            self.N_defocus_3D = self.N_defocus + 2*self.pad_z
            
            if self.z_defocus[0] - self.z_defocus[1] >0:
                z = -ifftshift((np.r_[0:self.N_defocus_3D]-self.N_defocus_3D//2)*self.psz)
            else:
                z = ifftshift((np.r_[0:self.N_defocus_3D]-self.N_defocus_3D//2)*self.psz)
                
            Hz_det = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z, type='Prop')
            G_fun_z = gen_Hz_stack(self.fxx, self.fyy, self.Pupil_support, self.lambda_illu, z, type='Green')
            self.gen_3D_WOTF(Hz_det, G_fun_z)
            
    def gen_2D_WOTF(self, Hz_det):
        
        '''
    
        setup transfer functions for 2D phase deconvolution

        Parameters
        ----------
            Hz_det : numpy.ndarray
                     propagation kernel with size of (N, M, N_defocus)
                              

        '''

        self.Hu = np.zeros((self.N, self.M, self.N_defocus),complex)
        self.Hp = np.zeros((self.N, self.M, self.N_defocus),complex)
        
        for i in range(self.N_defocus):
            self.Hu[:,:,i], self.Hp[:,:,i] = WOTF_2D_compute(self.Source, self.Pupil_obj * Hz_det[:,:,i], \
                                                             use_gpu=self.use_gpu, gpu_id=self.gpu_id)
                
    def gen_semi_3D_WOTF(self, Hz_det, G_fun_z):
        
        '''
    
        setup transfer functions for semi-3D phase deconvolution

        Parameters
        ----------
            Hz_det  : numpy.ndarray
                      propagation kernel with size of (N, M, ph_deconv_layer)
                      
            G_fun_z : numpy.ndarray
                      Weyl's representation of Green's function with size of (N, M, ph_deconv_layer)                              

        '''
        
        self.Hu_semi3D = np.zeros((self.N, self.M, self.ph_deconv_layer),complex)
        self.Hp_semi3D = np.zeros((self.N, self.M, self.ph_deconv_layer),complex)
                    
        for i in range(self.ph_deconv_layer):
            self.Hu_semi3D[:,:,i], self.Hp_semi3D[:,:,i] = WOTF_semi_3D_compute(self.Source, self.Pupil_obj, Hz_det[:,:,i], \
                                                                                G_fun_z[:,:,i]*4*np.pi*1j/self.lambda_illu, \
                                                                                use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
            
    def gen_3D_WOTF(self, Hz_det, G_fun_z):
        
        '''
    
        setup transfer functions for 3D phase deconvolution

        Parameters
        ----------
            Hz_det  : numpy.ndarray
                      propagation kernel with size of (N, M, N_defocus_3D)
                      
            G_fun_z : numpy.ndarray
                      Weyl's representation of Green's function with size of (N, M, N_defocus_3D)                              

        '''
        
        self.H_re = np.zeros((self.N, self.M, self.N_defocus_3D),dtype='complex64')
        self.H_im = np.zeros((self.N, self.M, self.N_defocus_3D),dtype='complex64')
        
        
        self.H_re, self.H_im = WOTF_3D_compute(self.Source.astype('float32'), self.Pupil_obj.astype('complex64'), Hz_det.astype('complex64'), \
                                               G_fun_z.astype('complex64'), self.psz, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        
        
    def Phase_solver_para_setter(self, denoiser_2D='Tikhonov', 
                                       Tik_reg_abs_2D = 1e-6, Tik_reg_ph_2D = 1e-6, \
                                       TV_reg_abs_2D  = 1e-3, TV_reg_ph_2D  = 1e-3, 
                                       rho_2D         = 1e-5, itr_2D        = 20, \
                                       denoiser_3D    ='Tikhonov', 
                                       Tik_reg_ph_3D  = 1e-4, TV_reg_ph_3D  = 1e-3, 
                                       rho_3D         = 1e-5, itr_3D        = 20, \
                                       verbose        = True, bg_filter     = True):
        
        
        '''
    
        setup parameters for phase deconvolution with the corresponding dimensions

        Parameters
        ----------
            denoiser_2D    : str
                             denoiser for 2D and semi-3D phase reconstruction
                             'Tikhonov' for Tikhonov denoiser
                             'TV'       for TV denoiser
                             
            Tik_reg_abs_2D : float
                             Tikhonov regularization parameter for 2D and semi-3D absorption
                             
            Tik_reg_ph_2D  : float
                             Tikhonov regularization parameter for 2D and semi-3D phase
                             
            TV_reg_abs_2D  : float
                             TV regularization parameter for 2D and semi-3D absorption
                             
            TV_reg_ph_2D   : float        
                             TV regularization parameter for 2D and semi-3D absorption
                             
            rho_2D         : float
                             augmented Lagrange multiplier for 2D amd semi-3D ADMM algorithm
                             
            itr_2D         : int
                             number of iterations for 2D and semi-3D ADMM algorithm
                             
            denoiser_3D    : str
                             denoiser for 3D phase reconstruction
                             'Tikhonov' for Tikhonov denoiser
                             'TV'       for TV denoiser
                             
            Tik_reg_ph_3D  : float
                             Tikhonov regularization parameter for 3D phase
                             
            TV_reg_ph_3D   : float
                             TV regularization parameter for 3D phase
                             
            rho_3D         : float
                             augmented Lagrange multiplier for 3D ADMM algorithm
                             
            itr_3D         : int
                             number of iterations for 3D ADMM algorithm
                             
            verbose        : bool
                             option to display detailed progress of computations or not
                             
            bg_filter      : bool
                             option for slow-varying 2D background normalization with uniform filter
        '''
        
        valid_denoiser_list = ['Tikhonov','TV']
        
        
        if denoiser_2D in valid_denoiser_list:
            self.denoiser_2D = denoiser_2D
        else:
            self.denoiser_2D = 'Tikhonov'
            print("denoiser_2D must be 'Tikhonov' or 'TV, set it to 'Tikhonov' in default")
            
        
        if denoiser_3D in valid_denoiser_list:
            self.denoiser_3D = denoiser_3D
        else:
            self.denoiser_3D = 'Tikhonov'
            print("denoiser_3D must be 'Tikohonov' or 'TV, set it to 'Tikhonov' in default")
        
        self.Tik_reg_abs_2D = Tik_reg_abs_2D
        self.Tik_reg_ph_2D  = Tik_reg_ph_2D
        
        self.Tik_reg_ph_3D  = Tik_reg_ph_3D
        
        self.TV_reg_abs_2D  = TV_reg_abs_2D
        self.TV_reg_ph_2D   = TV_reg_ph_2D
        self.rho_2D         = rho_2D
        self.itr_2D         = itr_2D
        
        self.TV_reg_ph_3D   = TV_reg_ph_3D
        self.rho_3D         = rho_3D
        self.itr_3D         = itr_3D
        
        self.verbose        = verbose
        self.bg_filter      = bg_filter
        
        
    
    
    def Phase_recon_2D(self, stoke_stack: StokesData):
        
        '''
    
        conduct 2D phase reconstruction

        Parameters
        ----------
            stoke_stack : StokesData
                          StokesData of the sample with each channel having (N, M, N_defocus)-sized numpy.ndarray
                          
        Returns
        -------
            mu_sample  : numpy.ndarray
                         2D absorption reconstruction with the size of (N, M)
                  
            phi_sample : numpy.ndarray
                         2D phase reconstruction with the size of (N, M)
                      
                                          
        '''
        
        S0_stack = stoke_stack.s0
        
        
        if self.use_gpu:
            
            S0_stack = inten_normalization(cp.array(S0_stack), type='2D', bg_filter=self.bg_filter, use_gpu=True, gpu_id=self.gpu_id)
            
            Hu = cp.array(self.Hu, copy=True)
            Hp = cp.array(self.Hp, copy=True)
            
            S0_stack_f = cp.fft.fft2(S0_stack, axes=(0,1))
            
            AHA   = [cp.sum(cp.abs(Hu)**2, axis=2) + self.Tik_reg_abs_2D, cp.sum(cp.conj(Hu)*Hp, axis=2),\
                     cp.sum(cp.conj(Hp)*Hu, axis=2),                      cp.sum(cp.abs(Hp)**2, axis=2) + self.Tik_reg_ph_2D]
            
            b_vec = [cp.sum(cp.conj(Hu)*S0_stack_f, axis=2), \
                     cp.sum(cp.conj(Hp)*S0_stack_f, axis=2)]
            
        else:
            S0_stack   = inten_normalization(S0_stack, type='2D', bg_filter=self.bg_filter)
            S0_stack_f = fft2(S0_stack,axes=(0,1))
            
            AHA   = [np.sum(np.abs(self.Hu)**2, axis=2) + self.Tik_reg_abs_2D, np.sum(np.conj(self.Hu)*self.Hp, axis=2),\
                     np.sum(np.conj(self.Hp)*self.Hu, axis=2),                 np.sum(np.abs(self.Hp)**2, axis=2) + self.Tik_reg_ph_2D]
            
            b_vec = [np.sum(np.conj(self.Hu)*S0_stack_f, axis=2), \
                     np.sum(np.conj(self.Hp)*S0_stack_f, axis=2)]
        
        
        if self.denoiser_2D == 'Tikhonov':
            
            # Deconvolution with Tikhonov regularization
            
            mu_sample, phi_sample = WOTF_Tikhonov_deconv_2D(AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
            
        elif self.denoiser_2D == 'TV':
            
            # ADMM deconvolution with anisotropic TV regularization
            
            mu_sample, phi_sample = WOTF_ADMM_TV_deconv_2D(AHA, b_vec, self.rho_2D, self.TV_reg_abs_2D, self.TV_reg_ph_2D, \
                                                           self.itr_2D, self.verbose, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        else:
            raise TypeError("denoiser type must be 'Tikhonov' or 'TV'")
            
        
        phi_sample -= phi_sample.mean()
        
        return mu_sample, phi_sample
    
    def Phase_recon_semi_3D(self, stoke_stack: StokesData):
        
        '''
    
        conduct semi-3D phase reconstruction

        Parameters
        ----------
            stoke_stack : StokesData
                          StokesData of the sample with each channel having (N, M, N_defocus)-sized numpy.ndarray
                          
        Returns
        -------
            mu_sample  : numpy.ndarray
                         semi-3D absorption reconstruction with the size of (N, M, N_defocus)
                  
            phi_sample : numpy.ndarray
                         semi-3D phase reconstruction with the size of (N, M, N_defocus)
                      
                                          
        '''
        
        S0_stack   = stoke_stack.s0
        
        mu_sample  = np.zeros((self.N, self.M, self.N_defocus))
        phi_sample = np.zeros((self.N, self.M, self.N_defocus))

        for i in range(self.N_defocus):
            
            # determines which indices to slice for each semi-3D reconstruction
            if i <= self.ph_deconv_layer//2:
                tf_start_idx = self.ph_deconv_layer//2 - i
            else:
                tf_start_idx = 0

            obj_start_idx = np.maximum(0,i-self.ph_deconv_layer//2)

            if self.N_defocus -i -1 < self.ph_deconv_layer//2:
                tf_end_idx = self.ph_deconv_layer//2 + (self.N_defocus - i)
            else:
                tf_end_idx = self.ph_deconv_layer

            obj_end_idx = np.minimum(self.N_defocus,i+self.ph_deconv_layer-self.ph_deconv_layer//2)
            
            if self.verbose:
                print('TF_index = (%d,%d), obj_z_index=(%d,%d), consistency: %s'\
                      %(tf_start_idx,tf_end_idx, obj_start_idx, obj_end_idx, (obj_end_idx-obj_start_idx)==(tf_end_idx-tf_start_idx)))
            
            
            if self.use_gpu:
                S0_stack_sub = inten_normalization(cp.array(S0_stack[:,:,obj_start_idx:obj_end_idx]), type='2D', bg_filter=self.bg_filter, use_gpu=True, gpu_id=self.gpu_id)
                S0_stack_f   = cp.fft.fft2(S0_stack_sub, axes=(0,1))
                
                Hu = cp.array(self.Hu_semi3D[:,:,tf_start_idx:tf_end_idx], copy=True)
                Hp = cp.array(self.Hp_semi3D[:,:,tf_start_idx:tf_end_idx], copy=True)

                

                AHA   = [cp.sum(cp.abs(Hu)**2, axis=2) + self.Tik_reg_abs_2D, cp.sum(cp.conj(Hu)*Hp, axis=2),\
                         cp.sum(cp.conj(Hp)*Hu, axis=2),                      cp.sum(cp.abs(Hp)**2, axis=2) + self.Tik_reg_ph_2D]

                b_vec = [cp.sum(cp.conj(Hu)*S0_stack_f, axis=2), \
                         cp.sum(cp.conj(Hp)*S0_stack_f, axis=2)]

            else:
                S0_stack_sub = inten_normalization(S0_stack[:,:,obj_start_idx:obj_end_idx], type='2D', bg_filter=self.bg_filter)
                S0_stack_f   = fft2(S0_stack_sub,axes=(0,1))
                
                Hu = self.Hu_semi3D[:,:,tf_start_idx:tf_end_idx]
                Hp = self.Hp_semi3D[:,:,tf_start_idx:tf_end_idx]

                AHA   = [np.sum(np.abs(Hu)**2, axis=2) + self.Tik_reg_abs_2D, np.sum(np.conj(Hu)*Hp, axis=2),\
                         np.sum(np.conj(Hp)*Hu, axis=2),                      np.sum(np.abs(Hp)**2, axis=2) + self.Tik_reg_ph_2D]

                b_vec = [np.sum(np.conj(Hu)*S0_stack_f, axis=2), \
                         np.sum(np.conj(Hp)*S0_stack_f, axis=2)]
                
                
            if self.denoiser_2D == 'Tikhonov':

                # Deconvolution with Tikhonov regularization

                mu_sample_temp, phi_sample_temp = WOTF_Tikhonov_deconv_2D(AHA, b_vec, use_gpu=self.use_gpu, gpu_id=self.gpu_id)

            elif self.denoiser_2D == 'TV':

                # ADMM deconvolution with anisotropic TV regularization

                mu_sample_temp, phi_sample_temp = WOTF_ADMM_TV_deconv_2D(AHA, b_vec, self.rho_2D, self.TV_reg_abs_2D, self.TV_reg_ph_2D, \
                                                                         self.itr_2D, self.verbose, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
            else:
                raise TypeError("denoiser type must be 'Tikhonov' or 'TV'")

            mu_sample[:,:,i]  = mu_sample_temp.copy()
            phi_sample[:,:,i] = phi_sample_temp - phi_sample_temp.mean()
            
            
        return mu_sample, phi_sample
    
    def Phase_recon_3D(self, stoke_stack: StokesData, absorption_ratio=0.0):
        
        '''
    
        conduct semi-3D phase reconstruction

        Parameters
        ----------
            stoke_stack      : StokesData
                               StokesData of the sample with each channel having (N, M, N_defocus)-sized numpy.ndarray
            
            absorption_ratio : float
                               absorption = absorption_ratio * phase (absorption_ratio = 0 refers to pure phase approximation)
                          
        Returns
        -------
            f_real           : numpy.ndarray
                               3D phase reconstruction with the size of (N, M, N_defocus)
                      
                                          
        '''
        
        S0_stack = stoke_stack.s0
                
        
        if self.pad_z == 0:
            S0_stack = inten_normalization(S0_stack, type='3D')
        else:
            S0_stack = inten_normalization(np.pad(S0_stack,((0,0),(0,0),(self.pad_z,self.pad_z)), mode='constant',constant_values=S0_stack.mean()), type='3D')
            
            
        H_eff = self.H_re + absorption_ratio*self.H_im


        if self.denoiser_3D == 'Tikhonov':

            f_real = WOTF_Tikhonov_deconv_3D(S0_stack, H_eff, self.Tik_reg_ph_3D, use_gpu=self.use_gpu, gpu_id=self.gpu_id)

        elif self.denoiser_3D == 'TV':

            f_real = WOTF_ADMM_TV_deconv_3D(S0_stack, H_eff, self.rho_3D, self.Tik_reg_ph_3D, self.TV_reg_ph_3D, self.itr_3D,\
                                            self.verbose, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        else:
            raise TypeError("denoiser type must be 'Tikhonov' or 'TV'")

            


        if self.pad_z != 0:
            f_real = f_real[:,:,self.pad_z//2:-(self.pad_z//2)]
        
        
        return -f_real*self.psz/(4*np.pi/self.lambda_illu)
        