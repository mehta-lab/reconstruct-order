Utils
==============

The Utils package contains methods useful for .yml configuration file reading, for image metadata parsing, image processing, and image plotting.


Config Reader
-------------

.. automodule:: ReconstructOrder.utils.ConfigReader
    :members: ConfigReader, Dataset, Processing, Plotting
    :undoc-members:
    :show-inheritance:

Image IO
-------------
.. automodule:: ReconstructOrder.utils.imgIO
    :members: get_sorted_names, get_sub_dirs, FindDirContain_pos, copy_files_in_sub_dirs, loadTiff, sort_pol_channels, export_img
    :undoc-members:
    :show-inheritance:

imgProcessing
-------------
.. automodule:: ReconstructOrder.utils.imgProcessing
    :members: ImgMin, ImgLimit, nanRobustBlur, histequal, im_bit_convert, mean_pooling_2d, mean_pooling_2d_stack, imadjustStack, imadjust, imClip, linScale, removeBubbles, imcrop, toggle_selector, line_select_callback
    :undoc-members:
    :show-inheritance:

mManagerIO
-------------
.. automodule:: ReconstructOrder.utils.mManagerIO
    :members: mManagerReader, PolAcquReader
    :undoc-members:
    :show-inheritance:

flat field
-------------
.. automodule:: ReconstructOrder.utils.flat_field
    :members: FlatFieldCorrector
    :undoc-members:
    :show-inheritance:

background estimator
-------------
.. automodule:: ReconstructOrder.utils.background_estimator
    :members: BackgroundEstimator2D
    :undoc-members:
    :show-inheritance:

aux utils
-------------
.. automodule:: ReconstructOrder.utils.aux_utils
    :members: loop_pt
    :undoc-members:
    :show-inheritance: