Introduction
==============

Reconstruct Order
-----------------

Analyze density (bright-field, phase), anisotropy (birefringence, slow axis), and degree of polarization of specimens from polarization-resolved and depth-resolved images. The acquisition, calibration, background correction, and reconstruction algorithms are described in the following preprint:

Guo, S.-M., Yeh, L.-H., Folkesson, J.,..., Mehta, S. B. (2019). Revealing architectural order with quantitative label-free imaging and deep learning. `BioRxiv 631101 <https://doi.org/10.1101/631101>`_.

As an illustration, following figure shows inputs and outputs of the ReconstructOrder for polarization-resolved data acquired at 21 consecutive focal planes with 2D phase reconstruction algorithm.

.. image:: Fig_Readme.png

ReconstructOrder currently supports data format acquired using `Micro-Manager 1.4.22 multi-dimension acquisition <https://micro-manager.org/>`_

and `OpenPolScope acquisition plugin <https://openpolscope.org/>`_. We will add support for Micro-Manager 2.0 format in the next release.

Dependencies
-----------------
Reconstruct Order is a Python library and requires the following packages

* numpy>=1.10.0
* opencv-python>=3.4.2.16
* pandas>=0.24.2
* pyyaml>=3.13
* matplotlib>=3.0.3
* scikit-image>=0.15
* scipy>=1.2.1
* tifffile>=0.15.1
* googledrivedownloader>=0.4
* natsort>=7