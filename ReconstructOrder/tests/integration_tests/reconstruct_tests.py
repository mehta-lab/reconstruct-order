# bchhun, {2019-09-16}


from ReconstructOrder.compute.reconstruct import ImgReconstructor
from ..testMetrics import mse
import numpy as np



def test_complete_reconstruction(setup_gdrive_src_data):
    """
    tests compute sequence: background correction of birefringence from intensity data
    one z, one t, one p
    No config file parsing
    No plotting

    Parameters
    ----------
    setup_gdrive_src_data : pytest fixture to load data from gdrive

    Returns
    -------

    """
    bg_dat, sm_dat = setup_gdrive_src_data

    # compute initial reconstructor using background data
    img_reconstructor = ImgReconstructor(bg_dat,
                                         swing=0.03,
                                         wavelength=532)
    bg_stokes = img_reconstructor.compute_stokes(bg_dat)
    bg_stokes_normalized = img_reconstructor.stokes_normalization(bg_stokes)

    # compute sample stokes and correct with background data
    sm_stokes = img_reconstructor.compute_stokes(sm_dat)
    sm_stokes_normalized = img_reconstructor.stokes_normalization(sm_stokes)
    sm_stokes_normalized = img_reconstructor.correct_background(sm_stokes_normalized, bg_stokes_normalized)

    reconstructed_birefring = img_reconstructor.reconstruct_birefringence(sm_stokes_normalized)

    assert reconstructed_birefring.I_trans is not None
    assert reconstructed_birefring.retard is not None
    assert reconstructed_birefring.azimuth is not None
    assert reconstructed_birefring.polarization is not None

    
def test_recon_dims_shape(setup_reconstructed_data, setup_gdrive_target_data):
    """
    test dims and dtype

    Parameters
    ----------
    setup_reconstructed_data

    Returns
    -------

    """
    recon_data = setup_reconstructed_data
    target_dat = setup_gdrive_target_data

    assert recon_data.I_trans.shape == target_dat.I_trans.shape
    assert recon_data.retard.shape == target_dat.I_trans.shape
    assert recon_data.azimuth.shape == target_dat.I_trans.shape
    assert recon_data.polarization.shape == target_dat.I_trans.shape
    
    assert recon_data.I_trans.dtype == target_dat.I_trans.dtype
    assert recon_data.retard.dtype == target_dat.I_trans.dtype
    assert recon_data.azimuth.dtype == target_dat.I_trans.dtype
    assert recon_data.polarization.dtype == target_dat.I_trans.dtype


def test_recon_mse(setup_reconstructed_data, setup_gdrive_target_data):
    """
    test array by comparing MSE

    Parameters
    ----------
    setup_reconstructed_data
    setup_gdrive_target_data

    Returns
    -------

    """
    recon_data = setup_reconstructed_data
    target_dat = setup_gdrive_target_data
    
    assert mse(recon_data.I_trans, target_dat.I_trans) < np.finfo(np.float32).eps
    assert mse(recon_data.retard, target_dat.retard) < np.finfo(np.float32).eps
    assert mse(recon_data.azimuth, target_dat.azimuth) < np.finfo(np.float32).eps
    assert mse(recon_data.polarization, target_dat.polarization) < np.finfo(np.float32).eps



