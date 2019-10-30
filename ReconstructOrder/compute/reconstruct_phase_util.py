import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import uniform_filter

#### utility functions ####

def gen_coordinate(img_dim, ps):
    
    '''
    
    generate spatial and spatial frequency coordinate arrays
    
    Parameters
    ----------
        img_dim : tuple
                  shape of the computed 2D space with size of (Ny, Nx) 
                  
        ps      : float
                  pixel size of the image space
        
    Returns
    -------
        xx      : numpy.ndarray
                  2D x-coordinate array with the shape of img_dim
                  
        yy      : numpy.ndarray
                  2D y-coordinate array with the shape of img_dim
                  
        fxx     : numpy.ndarray
                  2D spatial frequency array in x-dimension with the shape of img_dim
        
        fyy     : numpy.ndarray
                  2D spatial frequency array in y-dimension with the shape of img_dim
    
    '''
    
    N, M = img_dim
    
    fx = ifftshift((np.r_[:M]-M/2)/M/ps)
    fy = ifftshift((np.r_[:N]-N/2)/N/ps)
    x  = ifftshift((np.r_[:M]-M/2)*ps)
    y  = ifftshift((np.r_[:N]-N/2)*ps)


    xx, yy   = np.meshgrid(x, y)
    fxx, fyy = np.meshgrid(fx, fy)

    return (xx, yy, fxx, fyy)

def uniform_filter_2D(image, size, use_gpu=False, gpu_id=0):
    
    '''
    
    compute uniform filter operation on 2D image with gpu option
    
    Parameters
    ----------
        image          : numpy.ndarray
                         targeted image for filtering with size of (Ny, Nx) 
                  
        size           : int
                         size of the kernel for uniform filtering
        
        use_gpu        : bool
                         option to use gpu or not
        
        gpu_id         : int
                         number refering to which gpu will be used
    
    Returns
    -------
        image_filtered : numpy.ndarray
                         filtered image with size of (Ny, Nx)
                         
    '''
    
    
    N, M = image.shape
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        # filter in y direction
        
        image_cp = cp.array(image)
    
        kernel_y = cp.zeros((3*N,))
        kernel_y[3*N//2-size//2:3*N//2+size//2] = 1
        kernel_y /= cp.sum(kernel_y)
        kernel_y = cp.fft.fft(cp.fft.ifftshift(kernel_y))

        image_bound_y = cp.zeros((3*N,M))
        image_bound_y[N:2*N,:] = image_cp.copy()
        image_bound_y[0:N,:] = cp.flipud(image_cp)
        image_bound_y[2*N:3*N,:] = cp.flipud(image_cp)
        filtered_y = cp.real(cp.fft.ifft(cp.fft.fft(image_bound_y,axis=0)*kernel_y[:,cp.newaxis],axis=0))
        filtered_y = filtered_y[N:2*N,:]
        
        # filter in x direction
        
        kernel_x = cp.zeros((3*M,))
        kernel_x[3*M//2-size//2:3*M//2+size//2] = 1
        kernel_x /= cp.sum(kernel_x)
        kernel_x = cp.fft.fft(cp.fft.ifftshift(kernel_x))

        image_bound_x = cp.zeros((N,3*M))
        image_bound_x[:,M:2*M] = filtered_y.copy()
        image_bound_x[:,0:M] = cp.fliplr(filtered_y)
        image_bound_x[:,2*M:3*M] = cp.fliplr(filtered_y)

        image_filtered = cp.real(cp.fft.ifft(cp.fft.fft(image_bound_x,axis=1)*kernel_x[cp.newaxis,:],axis=1))
        image_filtered = image_filtered[:,M:2*M]
    else:
        image_filtered = uniform_filter(image, size=size)
        
        
    return image_filtered


def softTreshold(x, threshold, use_gpu=False, gpu_id=0):
    
    '''
    
    compute soft thresholding operation on numpy ndarray with gpu option
    
    Parameters
    ----------
        x          : numpy.ndarray
                     targeted array for soft thresholding operation with arbitrary size
                  
        threshold  : numpy.ndarray
                     array contains threshold value for each x array position
        
        use_gpu    : bool
                     option to use gpu or not
        
        gpu_id     : int
                     number refering to which gpu will be used
    
    Returns
    -------
        x_threshold : numpy.ndarray
                      thresholded array
                      
    '''
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        magnitude = cp.abs(x)
        ratio     = cp.maximum(0, magnitude-threshold) / magnitude
    else:
        magnitude = np.abs(x)
        ratio     = np.maximum(0, magnitude-threshold) / magnitude
        
    x_threshold = x*ratio
    
    return x_threshold


def inten_normalization(img_stack, type='2D' , bg_filter=True, use_gpu=False, gpu_id=0):
    
    '''
    
    layer-by-layer or whole-stack intensity normalization to reduce low-frequency phase artifacts
    
    Parameters
    ----------
        img_stack      : numpy.ndarray
                         image stack for normalization with size of (Ny, Nx, Nz)
                  
        type           : str
                         '2D' refers to layer-by-layer and '3D' refers to whole-stack normalization
                     
        bg_filter      : bool
                         option for slow-varying 2D background normalization with uniform filter
        
        use_gpu        : bool
                         option to use gpu or not
        
        gpu_id         : int
                         number refering to which gpu will be used
    
    Returns
    -------
        img_norm_stack : numpy.ndarray
                         normalized image stack with size of (Ny, Nx, Nz)
                         
    '''
    
    if type == '2D':
        
        # layer-by-layer normalization
        
        N, M, Nimg = img_stack.shape

        if use_gpu:
            globals()['cp'] = __import__("cupy")
            cp.cuda.Device(gpu_id).use()

            img_norm_stack = cp.zeros_like(img_stack)

            for i in range(Nimg):
                if bg_filter:
                    img_norm_stack[:,:,i] = img_stack[:,:,i]/uniform_filter_2D(img_stack[:,:,i], size=N//2, use_gpu=True, gpu_id=gpu_id)
                else:
                    img_norm_stack[:,:,i] = img_stack[:,:,i].copy()
                img_norm_stack[:,:,i] /= img_norm_stack[:,:,i].mean()
                img_norm_stack[:,:,i] -= 1

        else:
            img_norm_stack = np.zeros_like(img_stack)

            for i in range(Nimg):
                if bg_filter:
                    img_norm_stack[:,:,i] = img_stack[:,:,i]/uniform_filter(img_stack[:,:,i], size=N//2)
                else:
                    img_norm_stack[:,:,i] = img_stack[:,:,i].copy()
                img_norm_stack[:,:,i] /= img_norm_stack[:,:,i].mean()
                img_norm_stack[:,:,i] -= 1
    
    elif type == '3D':
        
        # whole-stack normalization
        
        img_norm_stack = np.zeros_like(img_stack)
        img_norm_stack = img_stack / img_stack.mean()
        img_norm_stack -= 1

    return img_norm_stack


#### Diffraction related functions ####

def gen_Pupil(fxx, fyy, NA, lambda_in):
    
    
    '''
    
    compute pupil function given spatial frequency, NA, wavelength
    
    Parameters
    ----------
        fxx       : numpy.ndarray
                    2D spatial frequency array in x-dimension with the shape of (Ny, Nx)
        
        fyy       : numpy.ndarray
                    2D spatial frequency array in y-dimension with the shape of (Ny, Nx)
                  
        NA        : float
                    numerical aperture of the pupil function
                    
        lambda_in : float
                    wavelength of the incident light
    
    Returns
    -------
        Pupil     : numpy.ndarray
                    computed pupil function with size of (Ny, Nx)
                    
    '''
    
    N, M = fxx.shape
    
    Pupil = np.zeros((N,M))
    fr = (fxx**2 + fyy**2)**(1/2)
    Pupil[ fr < NA/lambda_in] = 1
    
    return Pupil



def gen_Hz_stack(fxx, fyy, Pupil_support, lambda_in, z_stack, type='Prop'):
    
    
    '''
    
    generate wave propagation kernel
    
    Parameters
    ----------
        fxx           : numpy.ndarray
                        2D spatial frequency array in x-dimension with the shape of (Ny, Nx)
        
        fyy           : numpy.ndarray
                        2D spatial frequency array in y-dimension with the shape of (Ny, Nx)
                    
        Pupil_support : numpy.ndarray
                        support of the pupil function (within NA) with the shape of (Ny, Nx)
                    
        lambda_in     : float
                        wavelength of the incident light
        
        z_stack       : numpy.ndarray
                        a 1D z position array with size of (Nz,) for the corresponding kernels
                        
        type          : str
                        'Prop' for propagation kernel and 'Green' for Weyl's representation of Green's function
    
    Returns
    -------
        Hz_stack      : numpy.ndarray
                        computed propagation kernel with size of (Ny, Nx, Nz)
                    
    '''
    
    N, M = fxx.shape
    N_stack = len(z_stack)
    N_defocus = len(z_stack)
    
    fr = (fxx**2 + fyy**2)**(1/2)
    
    oblique_factor = ((1 - lambda_in**2 * fr**2) *Pupil_support)**(1/2) / lambda_in
    
    if type == 'Prop':
        Hz_stack = Pupil_support[:,:,np.newaxis] * np.exp(1j*2*np.pi*z_stack[np.newaxis,np.newaxis,:]*\
                                                          oblique_factor[:,:,np.newaxis])
    elif type == 'Green':
        Hz_stack = -1j/4/np.pi* Pupil_support[:,:,np.newaxis] * \
              np.exp(1j*2*np.pi*z_stack[np.newaxis,np.newaxis,:] * \
                     oblique_factor[:,:,np.newaxis]) /(oblique_factor[:,:,np.newaxis]+1e-15)
    
    return Hz_stack



def WOTF_2D_compute(Source, Pupil, use_gpu=False, gpu_id=0):
    
    
    '''
    
    compute 2D weak object transfer function (2D WOTF)
    
    Parameters
    ----------
        Source  : numpy.ndarray
                  source pattern with size of (Ny, Nx)
                 
        Pupil   : numpy.ndarray
                  pupil function with size of (Ny, Nx)
        
        use_gpu : bool
                  option to use gpu or not
        
        gpu_id  : int
                  number refering to which gpu will be used
    
    Returns
    -------
        Hu      : numpy.ndarray
                  absorption transfer function with size of (Ny, Nx)
                  
        Hp      : numpy.ndarray
                  phase transfer function with size of (Ny, Nx)

    '''
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        Source = cp.array(Source)
        Pupil  = cp.array(Pupil)
        
        H1     = cp.fft.ifft2(cp.conj(cp.fft.fft2(Source * Pupil))*cp.fft.fft2(Pupil))
        H2     = cp.fft.ifft2(cp.fft.fft2(Source * Pupil)*cp.conj(cp.fft.fft2(Pupil)))
        I_norm = cp.sum(Source * Pupil * cp.conj(Pupil))
        Hu     = (H1 + H2)/I_norm
        Hp     = 1j*(H1-H2)/I_norm
        
        Hu     = cp.asnumpy(Hu)
        Hp     = cp.asnumpy(Hp)
        
    else:
    
        H1     = ifft2(fft2(Source * Pupil).conj()*fft2(Pupil))
        H2     = ifft2(fft2(Source * Pupil)*fft2(Pupil).conj())
        I_norm = np.sum(Source * Pupil * Pupil.conj())
        Hu     = (H1 + H2)/I_norm
        Hp     = 1j*(H1-H2)/I_norm
    
    return Hu, Hp

def WOTF_semi_3D_compute(Source, Pupil, Hz_det, G_fun_z, use_gpu=False, gpu_id=0):
    
    '''
    
    compute semi-3D weak object transfer function (semi-3D WOTF)
    
    Parameters
    ----------
        Source  : numpy.ndarray
                  source pattern with size of (Ny, Nx)
                 
        Pupil   : numpy.ndarray
                  pupil function with size of (Ny, Nx)
                  
        Hz_det  : numpy.ndarray
                  one slice of propagation kernel with size of (Ny, Nx)
                  
        G_fun_z : numpy.ndarray
                  one slice of  Weyl's representation of Green's function with size of (Ny, Nx)
        
        use_gpu : bool
                  option to use gpu or not
        
        gpu_id  : int
                  number refering to which gpu will be used
    
    Returns
    -------
        Hu      : numpy.ndarray
                  absorption transfer function with size of (Ny, Nx)
                  
        Hp      : numpy.ndarray
                  phase transfer function with size of (Ny, Nx)

    '''
    
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        Source  = cp.array(Source)
        Pupil   = cp.array(Pupil)        
        Hz_det  = cp.array(Hz_det)
        G_fun_z = cp.array(G_fun_z)
        
        H1      = cp.fft.ifft2(cp.conj(cp.fft.fft2(Source * Pupil * Hz_det))*cp.fft.fft2(Pupil * G_fun_z))
        H2      = cp.fft.ifft2(cp.fft.fft2(Source * Pupil * Hz_det)*cp.conj(cp.fft.fft2(Pupil * G_fun_z)))
        I_norm  = cp.sum(Source * Pupil * cp.conj(Pupil))
        Hu      = (H1 + H2)/I_norm
        Hp      = 1j*(H1-H2)/I_norm
        
        Hu      = cp.asnumpy(Hu)
        Hp      = cp.asnumpy(Hp)
        
    else:
    
        H1     = ifft2(fft2(Source * Pupil * Hz_det).conj()*fft2(Pupil * G_fun_z))
        H2     = ifft2(fft2(Source * Pupil * Hz_det)*fft2(Pupil * G_fun_z).conj())
        I_norm = np.sum(Source * Pupil * Pupil.conj())
        Hu     = (H1 + H2)/I_norm
        Hp     = 1j*(H1-H2)/I_norm
    
    return Hu, Hp


def WOTF_3D_compute(Source, Pupil, Hz_det, G_fun_z, psz, use_gpu=False, gpu_id=0):
    
    '''
    
    compute 3D weak object transfer function (3D WOTF)
    
    Parameters
    ----------
        Source  : numpy.ndarray
                  source pattern with size of (Ny, Nx)
                 
        Pupil   : numpy.ndarray
                  pupil function with size of (Ny, Nx)
                  
        Hz_det  : numpy.ndarray
                  propagation kernel with size of (Ny, Nx, Nz)
                  
        G_fun_z : numpy.ndarray
                  Weyl's representation of Green's function with size of (Ny, Nx, Nz)
                  
        psz     : float
                  pixel size in the z-dimension
        
        use_gpu : bool
                  option to use gpu or not
        
        gpu_id  : int
                  number refering to which gpu will be used
    
    Returns
    -------
        H_re    : numpy.ndarray
                  transfer function of real scattering potential with size of (Ny, Nx, Nz)
                  
        H_im    : numpy.ndarray
                  transfer function of imaginary scattering potential with size of (Ny, Nx, Nz)

    '''
    
    
    
    _,_,Nz = Hz_det.shape
    
    window = ifftshift(np.hanning(Nz)).astype('float32')
    
    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        Source  = cp.array(Source)
        Pupil   = cp.array(Pupil)
        Hz_det  = cp.array(Hz_det)
        G_fun_z = cp.array(G_fun_z)
        window  = cp.array(window)
        
        H1      = cp.fft.ifft2(cp.conj(cp.fft.fft2(Source[:,:,cp.newaxis] * Pupil[:,:,cp.newaxis] * Hz_det, axes=(0,1)))*\
                               cp.fft.fft2(Pupil[:,:,cp.newaxis] * G_fun_z, axes=(0,1)), axes=(0,1))
        H1      = H1*window[cp.newaxis,cp.newaxis,:]
        H1      = cp.fft.fft(H1, axis=2)*psz
        H2      = cp.fft.ifft2(cp.fft.fft2(Source[:,:,cp.newaxis] * Pupil[:,:,cp.newaxis] * Hz_det, axes=(0,1))*\
                               cp.conj(cp.fft.fft2(Pupil[:,:,cp.newaxis] * G_fun_z, axes=(0,1))), axes=(0,1))
        H2      = H2*window[cp.newaxis,cp.newaxis,:]
        H2      = cp.fft.fft(H2, axis=2)*psz
    

        I_norm  = cp.sum(Source * Pupil * cp.conj(Pupil))
        H_re    = (H1 + H2)/I_norm
        H_im    = 1j*(H1-H2)/I_norm
        
        H_re    = cp.asnumpy(H_re)
        H_im    = cp.asnumpy(H_im)
        
    else:
    
        

        H1 = ifft2(fft2(Source[:,:,np.newaxis] * Pupil[:,:,np.newaxis] * Hz_det, axes=(0,1)).conj()*\
                   fft2(Pupil[:,:,np.newaxis] * G_fun_z, axes=(0,1)), axes=(0,1))
        H1 = H1*window[np.newaxis,np.newaxis,:]
        H1 = fft(H1, axis=2)*psz
        H2 = ifft2(fft2(Source[:,:,np.newaxis] * Pupil[:,:,np.newaxis] * Hz_det, axes=(0,1))*\
                   fft2(Pupil[:,:,np.newaxis] * G_fun_z, axes=(0,1)).conj(), axes=(0,1))
        H2 = H2*window[np.newaxis,np.newaxis,:]
        H2 = fft(H2, axis=2)*psz

        I_norm = np.sum(Source * Pupil * Pupil.conj())
        H_re   = (H1 + H2)/I_norm
        H_im   = 1j*(H1-H2)/I_norm
    
    
    return H_re, H_im



#### Solver related functions ####

def WOTF_Tikhonov_deconv_2D(AHA, b_vec, use_gpu=False, gpu_id=0):
    
    
    '''
    
    2D Tikhonov deconvolution to solve for phase and absorption with weak object transfer function
    
    Parameters
    ----------
        AHA        : list
                     A^H times A matrix stored with a list of 4 2D numpy array (4 diagonal matrices)
                  
        b_vec      : list
                     measured intensity stored with a list of 2 2D numpy array (2 vectors)
        
        use_gpu    : bool
                      option to use gpu or not
        
        gpu_id     : int
                     number refering to which gpu will be used
    
    Returns
    -------
        mu_sample  : numpy.ndarray
                     2D absorption reconstruction with the size of (Ny, Nx)
                  
        phi_sample : numpy.ndarray
                     2D phase reconstruction with the size of (Ny, Nx)

    '''
        
    determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]


    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        mu_sample  = cp.asnumpy(cp.real(cp.fft.ifft2((b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant)))
        phi_sample = cp.asnumpy(cp.real(cp.fft.ifft2((b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant)))
    else:
        mu_sample  = np.real(ifft2((b_vec[0]*AHA[3] - b_vec[1]*AHA[1]) / determinant))
        phi_sample = np.real(ifft2((b_vec[1]*AHA[0] - b_vec[0]*AHA[2]) / determinant))

    return mu_sample, phi_sample


def WOTF_ADMM_TV_deconv_2D(AHA, b_vec, rho, lambda_u, lambda_p, itr, verbose, use_gpu=False, gpu_id=0):
    
    '''
    
    2D TV deconvolution to solve for phase and absorption with weak object transfer function
    
    ADMM formulation:
        
        0.5 * || A*x - b ||_2^2 + lambda * || z ||_1 + 0.5 * rho * || D*x - z + u ||_2^2
    
    Parameters
    ----------
        AHA        : list
                     A^H times A matrix stored with a list of 4 2D numpy array (4 diagonal matrices)
                  
        b_vec      : list
                     measured intensity stored with a list of 2 2D numpy array (2 vectors)
                     
        rho        : float
                     ADMM rho parameter
        
        lambda_u   : float
                     TV regularization parameter for absorption
        
        lambda_p   : float
                     TV regularization parameter for phase
        
        itr        : int
                     number of iterations of ADMM algorithm
        
        verbose    : bool
                     option to display progress of the computation
        
        use_gpu    : bool
                      option to use gpu or not
        
        gpu_id     : int
                     number refering to which gpu will be used
    
    Returns
    -------
        mu_sample  : numpy.ndarray
                     2D absorption reconstruction with the size of (Ny, Nx)
                  
        phi_sample : numpy.ndarray
                     2D phase reconstruction with the size of (Ny, Nx)

    '''


    # ADMM deconvolution with anisotropic TV regularization

    N, M = b_vec[0].shape

    if use_gpu:
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()

        Dx = cp.zeros((N, M))
        Dy = cp.zeros((N, M))
        Dx[0,0] = 1; Dx[0,-1] = -1; Dx = cp.fft.fft2(Dx);
        Dy[0,0] = 1; Dy[-1,0] = -1; Dy = cp.fft.fft2(Dy);

        rho_term = rho*(cp.conj(Dx)*Dx + cp.conj(Dy)*Dy)

        z_para = cp.zeros((4, N, M))
        u_para = cp.zeros((4, N, M))
        D_vec  = cp.zeros((4, N, M))


    else:
        Dx = np.zeros((N, M))
        Dy = np.zeros((N, M))
        Dx[0,0] = 1; Dx[0,-1] = -1; Dx = fft2(Dx);
        Dy[0,0] = 1; Dy[-1,0] = -1; Dy = fft2(Dy);

        rho_term = rho*(np.conj(Dx)*Dx + np.conj(Dy)*Dy)

        z_para = np.zeros((4, N, M))
        u_para = np.zeros((4, N, M))
        D_vec  = np.zeros((4, N, M))


    AHA[0] = AHA[0] + rho_term
    AHA[3] = AHA[3] + rho_term

    determinant = AHA[0]*AHA[3] - AHA[1]*AHA[2]


    for i in range(itr):


        if use_gpu:

            v_para    = cp.fft.fft2(z_para - u_para)
            b_vec_new = [b_vec[0] + rho*(cp.conj(Dx)*v_para[0] + cp.conj(Dy)*v_para[1]),\
                         b_vec[1] + rho*(cp.conj(Dx)*v_para[2] + cp.conj(Dy)*v_para[3])]


            mu_sample  = cp.real(cp.fft.ifft2((b_vec_new[0]*AHA[3] - b_vec_new[1]*AHA[1]) / determinant))
            phi_sample = cp.real(cp.fft.ifft2((b_vec_new[1]*AHA[0] - b_vec_new[0]*AHA[2]) / determinant))

            D_vec[0] = mu_sample - cp.roll(mu_sample, -1, axis=1)
            D_vec[1] = mu_sample - cp.roll(mu_sample, -1, axis=0)
            D_vec[2] = phi_sample - cp.roll(phi_sample, -1, axis=1)
            D_vec[3] = phi_sample - cp.roll(phi_sample, -1, axis=0)


            z_para = D_vec + u_para

            z_para[:2,:,:] = softTreshold(z_para[:2,:,:], lambda_u/rho, use_gpu=True, gpu_id=gpu_id)
            z_para[2:,:,:] = softTreshold(z_para[2:,:,:], lambda_p/rho, use_gpu=True, gpu_id=gpu_id)

            u_para += D_vec - z_para

            if i == itr-1:
                mu_sample  = cp.asnumpy(mu_sample)
                phi_sample = cp.asnumpy(phi_sample)




        else:

            v_para = fft2(z_para - u_para)
            b_vec_new = [b_vec[0] + rho*(np.conj(Dx)*v_para[0] + np.conj(Dy)*v_para[1]),\
                         b_vec[1] + rho*(np.conj(Dx)*v_para[2] + np.conj(Dy)*v_para[3])]


            mu_sample  = np.real(ifft2((b_vec_new[0]*AHA[3] - b_vec_new[1]*AHA[1]) / determinant))
            phi_sample = np.real(ifft2((b_vec_new[1]*AHA[0] - b_vec_new[0]*AHA[2]) / determinant))

            D_vec[0] = mu_sample - np.roll(mu_sample, -1, axis=1)
            D_vec[1] = mu_sample - np.roll(mu_sample, -1, axis=0)
            D_vec[2] = phi_sample - np.roll(phi_sample, -1, axis=1)
            D_vec[3] = phi_sample - np.roll(phi_sample, -1, axis=0)


            z_para = D_vec + u_para

            z_para[:2,:,:] = softTreshold(z_para[:2,:,:], lambda_u/rho)
            z_para[2:,:,:] = softTreshold(z_para[2:,:,:], lambda_p/rho)

            u_para += D_vec - z_para

        if verbose:
            print('Number of iteration computed (%d / %d)'%(i+1,itr))

    return mu_sample, phi_sample
    
    
def WOTF_Tikhonov_deconv_3D(S0_stack, H_eff, reg_re, use_gpu=False, gpu_id=0):
    
    '''
    
    3D Tikhonov deconvolution to solve for phase with weak object transfer function
    
    Parameters
    ----------
        S0_stack : numpy.ndarray
                   S0 z-stack for 3D phase deconvolution with size of (Ny, Nx, Nz)
                  
        H_eff    : numpy.ndarray
                   effective transfer function with size of (Ny, Nx, Nz)
                     
        reg_re   : float
                   Tikhonov regularization parameter
        
        use_gpu  : bool
                   option to use gpu or not
        
        gpu_id   : int
                   number refering to which gpu will be used
    
    Returns
    -------
        f_real   : numpy.ndarray
                   3D unscaled phase reconstruction with the size of (Ny, Nx, Nz)

    '''
    
    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        S0_stack_f = cp.fft.fftn(cp.array(S0_stack.astype('float32')), axes=(0,1,2))
        H_eff      = cp.array(H_eff.astype('complex64'))
        
        f_real     = cp.asnumpy(cp.real(cp.fft.ifftn(S0_stack_f * cp.conj(H_eff) / (cp.abs(H_eff)**2 + reg_re),axes=(0,1,2))))
    else:
        
        S0_stack_f = fftn(S0_stack, axes=(0,1,2))
        
        f_real     = np.real(ifftn(S0_stack_f * np.conj(H_eff) / (np.abs(H_eff)**2 + reg_re),axes=(0,1,2)))
        
    return f_real
    
    
    
def WOTF_ADMM_TV_deconv_3D(S0_stack, H_eff, rho, reg_re, lambda_re, itr, verbose, use_gpu=False, gpu_id=0):
    
    
    '''
    
    3D TV deconvolution to solve for phase with weak object transfer function
    
    ADMM formulation:
        
        0.5 * || A*x - b ||_2^2 + lambda * || z ||_1 + 0.5 * rho * || D*x - z + u ||_2^2
    
    Parameters
    ----------
        S0_stack  : numpy.ndarray
                    S0 z-stack for 3D phase deconvolution with size of (Ny, Nx, Nz)
                  
        H_eff     : numpy.ndarray
                    effective transfer function with size of (Ny, Nx, Nz)
                     
        reg_re    : float
                    Tikhonov regularization parameter
                     
        rho       : float
                    ADMM rho parameter
        
        lambda_re : float
                    TV regularization parameter for phase
        
        itr       : int
                    number of iterations of ADMM algorithm
        
        verbose   : bool
                    option to display progress of the computation
        
        use_gpu   : bool
                    option to use gpu or not
        
        gpu_id    : int
                    number refering to which gpu will be used
    
    Returns
    -------
        f_real    : numpy.ndarray
                    3D unscaled phase reconstruction with the size of (Ny, Nx, Nz)

    '''
    
    N, M, N_defocus = S0_stack.shape
    
    Dx = np.zeros((N, M, N_defocus)); Dx[0,0,0] = 1; Dx[0,-1,0] = -1;
    Dy = np.zeros((N, M, N_defocus)); Dy[0,0,0] = 1; Dy[-1,0,0] = -1;
    Dz = np.zeros((N, M, N_defocus)); Dz[0,0,0] = 1; Dz[0,0,-1] = -1;
    
    if use_gpu:
        
        globals()['cp'] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        
        S0_stack_f = cp.fft.fftn(cp.array(S0_stack.astype('float32')), axes=(0,1,2))
        H_eff = cp.array(H_eff.astype('complex64'))
        
        Dx = cp.fft.fftn(cp.array(Dx),axes=(0,1,2))
        Dy = cp.fft.fftn(cp.array(Dy),axes=(0,1,2))
        Dz = cp.fft.fftn(cp.array(Dz),axes=(0,1,2))

        rho_term = rho*(cp.conj(Dx)*Dx + cp.conj(Dy)*Dy + cp.conj(Dz)*Dz)+reg_re
        AHA      = cp.abs(H_eff)**2 + rho_term
        b_vec    = S0_stack_f * cp.conj(H_eff)

        z_para = cp.zeros((3, N, M, N_defocus))
        u_para = cp.zeros((3, N, M, N_defocus))
        D_vec  = cp.zeros((3, N, M, N_defocus))




        for i in range(itr):
            v_para    = cp.fft.fftn(z_para - u_para, axes=(1,2,3))
            b_vec_new = b_vec + rho*(cp.conj(Dx)*v_para[0] + cp.conj(Dy)*v_para[1] + cp.conj(Dz)*v_para[2])


            f_real = cp.real(cp.fft.ifftn(b_vec_new / AHA, axes=(0,1,2)))

            D_vec[0] = f_real - cp.roll(f_real, -1, axis=1)
            D_vec[1] = f_real - cp.roll(f_real, -1, axis=0)
            D_vec[2] = f_real - cp.roll(f_real, -1, axis=2)


            z_para = D_vec + u_para

            z_para = softTreshold(z_para, lambda_re/rho, use_gpu=True, gpu_id=gpu_id)

            u_para += D_vec - z_para

            if verbose:
                print('Number of iteration computed (%d / %d)'%(i+1,itr))

            if i == itr-1:
                f_real = cp.asnumpy(f_real)
    
    else:
        
        S0_stack_f = fftn(S0_stack, axes=(0,1,2))
        
        Dx = fftn(Dx,axes=(0,1,2));
        Dy = fftn(Dy,axes=(0,1,2));
        Dz = fftn(Dz,axes=(0,1,2));

        rho_term = rho*(np.conj(Dx)*Dx + np.conj(Dy)*Dy + np.conj(Dz)*Dz)+reg_re
        AHA      = np.abs(H_eff)**2 + rho_term
        b_vec    = S0_stack_f * np.conj(H_eff)

        z_para = np.zeros((3, N, M, N_defocus))
        u_para = np.zeros((3, N, M, N_defocus))
        D_vec  = np.zeros((3, N, M, N_defocus))


        for i in range(itr):
            v_para    = fftn(z_para - u_para, axes=(1,2,3))
            b_vec_new = b_vec + rho*(np.conj(Dx)*v_para[0] + np.conj(Dy)*v_para[1] + np.conj(Dz)*v_para[2])


            f_real = np.real(ifftn(b_vec_new / AHA, axes=(0,1,2)))

            D_vec[0] = f_real - np.roll(f_real, -1, axis=1)
            D_vec[1] = f_real - np.roll(f_real, -1, axis=0)
            D_vec[2] = f_real - np.roll(f_real, -1, axis=2)


            z_para = D_vec + u_para

            z_para = softTreshold(z_para, lambda_re/rho)

            u_para += D_vec - z_para

            if verbose:
                print('Number of iteration computed (%d / %d)'%(i+1,itr))
                
    return f_real
        