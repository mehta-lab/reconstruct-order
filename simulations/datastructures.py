#%%
from ReconstructOrder.compute.reconstruct import ImgReconstructor
from ReconstructOrder.datastructures import PhysicalData
from ReconstructOrder.datastructures.intensity_data import IntensityData
from ReconstructOrder.utils.ConfigReader import ConfigReader
import numpy as np

#%%
"""
The typical reconstruction workflow is as follows:
1) load background data
2) compute stokes on background data then normalize it
3) load sample data
4) compute stokes on sample then normalize it
6) Correct Background 

"""

#%%

"""
load data into an Intensity Data object
"""
import os
import tifffile as tf

int_dat = IntensityData()
config = ConfigReader()

int_dat.channel_names = ['IExt','I90','I135','I45','I0']

raw_data_path = "/Users/bryant.chhun/Desktop/Data/ForDataStructuresTests/Raw/Sample/Pos0"
int_dat.replace_image(tf.imread(os.path.join(raw_data_path, 'img_000000010_state0_002.tif')), 'IExt')
int_dat.replace_image(tf.imread(os.path.join(raw_data_path, 'img_000000010_state1_002.tif')), 'I90')
int_dat.replace_image(tf.imread(os.path.join(raw_data_path, 'img_000000010_state2_002.tif')), 'I135')
int_dat.replace_image(tf.imread(os.path.join(raw_data_path, 'img_000000010_state3_002.tif')), 'I45')
int_dat.replace_image(tf.imread(os.path.join(raw_data_path, 'img_000000010_state4_002.tif')), 'I0')

bg_int = IntensityData()
bg_int.channel_names = ['IExt','I90','I135','I45','I0']

bg_data_path = "/Users/bryant.chhun/Desktop/Data/ForDataStructuresTests/Raw/Background/Pos0"
bg_int.replace_image(tf.imread(os.path.join(bg_data_path, 'img_000000000_State0 - Acquired Image_000.tif')), 'IExt')
bg_int.replace_image(tf.imread(os.path.join(bg_data_path, 'img_000000000_State1 - Acquired Image_000.tif')), 'I90')
bg_int.replace_image(tf.imread(os.path.join(bg_data_path, 'img_000000000_State2 - Acquired Image_000.tif')), 'I135')
bg_int.replace_image(tf.imread(os.path.join(bg_data_path, 'img_000000000_State3 - Acquired Image_000.tif')), 'I45')
bg_int.replace_image(tf.imread(os.path.join(bg_data_path, 'img_000000000_State4 - Acquired Image_000.tif')), 'I0')

#%%
"""
reconstructor is initialized based on parameters from the experiment
    usually these params are found in the config file parsed from metadata or config.yml
"""
img_reconstructor = ImgReconstructor(bg_int.data.shape,
                                     bg_method="Local_fit",
                                     n_slice_local_bg=1,
                                     poly_fit_order=2,
                                     swing=0.03,
                                     wavelength=532,
                                     azimuth_offset=0,
                                     circularity='rcp')
#%%

print('computing background')
background_stokes = img_reconstructor.compute_stokes(config, bg_int)
background_normalized = img_reconstructor.stokes_normalization(background_stokes)

#%%

print('computing stokes')
sample_stokes = img_reconstructor.compute_stokes(int_dat)
sample_normalized = img_reconstructor.stokes_normalization(sample_stokes)

#%%
print('correcting background')
# let's keep a few background correction outputs

sample_bg_corrected_local_fit = img_reconstructor.correct_background(sample_normalized, background_normalized)

img_reconstructor.bg_method = "Local_filter"
sample_bg_corrected_local_filter = img_reconstructor.correct_background(sample_normalized, background_normalized)

img_reconstructor.bg_method = None
sample_bg_corrected_global = img_reconstructor.correct_background(sample_normalized, background_normalized)

#%%
print('computing physical')

# let's compute physical using several background correction methods
sample_physical_local_fit = img_reconstructor.reconstruct_birefringence(sample_bg_corrected_local_fit)

sample_physical_local_filter = img_reconstructor.reconstruct_birefringence(sample_bg_corrected_local_filter)

sample_physical_global = img_reconstructor.reconstruct_birefringence(sample_bg_corrected_global)

#%%
from ReconstructOrder.utils.plotting import im_bit_convert

print('calculate mse')
target_retardance = tf.imread(
    '/Users/bryant.chhun/Desktop/Data/ForDataStructuresTests/Processed/img_Retardance_t010_p000_z002.tif')
target_orientation = tf.imread(
    '/Users/bryant.chhun/Desktop/Data/ForDataStructuresTests/Processed/img_Orientation_t010_p000_z002.tif')
target_polarization = tf.imread(
    '/Users/bryant.chhun/Desktop/Data/ForDataStructuresTests/Processed/img_Polarization_t010_p000_z002.tif')


def mse(x, Y):
    return np.square(Y-x).mean()


print("MSE local fit retardance = "+str(mse(
    im_bit_convert(sample_physical_local_fit.retard * 1E4, bit=16, norm=False),
    target_retardance)))

print("MSE local fit orientation = "+str(mse(
    im_bit_convert(sample_physical_local_fit.azimuth * 1E4, bit=16, norm=False),
    target_retardance)))

print("MSE local fit polarization = "+str(mse(
    im_bit_convert(sample_physical_local_fit.polarization * 1E4, bit=16, norm=False),
    target_retardance)))


#%%
print('writing data to disk')
from ReconstructOrder.utils.plotting import im_bit_convert


def write_birefring(sample_data:PhysicalData, path):

    # 'Brightfield_computed'
    tf.imsave(path +"_bf_computed.tif", im_bit_convert(sample_data.I_trans * 1E4, bit=16, norm=False)) # AU, set norm to False for tiling images

    # 'Retardance'
    tf.imsave(path +"_retardance.tif", im_bit_convert(sample_data.retard * 1E3, bit=16))  # scale to pm

    # 'Orientation'
    tf.imsave(path +"_orientation.tif", im_bit_convert(sample_data.azimuth * 100, bit=16))  # scale to [0, 18000], 100*degree

    # 'Polarization':
    tf.imsave(path +"_polarization.tif", im_bit_convert(sample_data.polarization * 50000, bit=16))


TARGET_FILE_FOLDER = '/Users/bryant.chhun/Desktop/Data/ForDataStructuresTests/Raw/untitled folder'


write_birefring(sample_physical_local_fit, TARGET_FILE_FOLDER+"/fit")
write_birefring(sample_physical_local_filter, TARGET_FILE_FOLDER+"/filter")
write_birefring(sample_physical_global, TARGET_FILE_FOLDER+"/global")
