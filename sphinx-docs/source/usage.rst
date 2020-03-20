==========================
Usage
==========================

Overview
--------

The reconstruction parameters are specified in the configuration file.
Configuration file template(```config_example.yml```) can be found `here <https://github.com/mehta-lab/reconstruct-order>`_ under ```examples``` folder

which incluides detailed explanation of parameters for running ReconstructOrder in different modes

To use the configuration file template for your data, you need to at least modify ```dataset: data_dir``` and ```dataset: processed_dir``` to point to source data path and output path. See the template docstrings for the usage of other parameters.


How to run reconstruction:
-----------------------------------------
*There are two ways from command line*

If you pip installed the library, from any folder, simply type:

.. code-block:: bash

   runReconstruction --config path-and-name-to-your-config.yml

or

.. code-block:: bash

   python runReconstruction.py --config path-and-name-to-your-config.yml


*Inside Python scripts*

To call ReconstructOrder reconstruction in your own script:

.. code-block:: python

   import ReconstructOrder.workflow as wf
   wf.reconstruct_batch('path-and-name-to-your-config.yml')



Example
---------
In the following, we demonstrate how to download our example dataset (hosted [here](https://drive.google.com/drive/u/3/folders/1axmPgQVNi22ZqGLXzHGHIuP9kA93K9zH)) and run ReconstructOrder on it to get birefringence and phase images. This instruction should work for installation from both Option 1 and 2. <br>

a) In the terminal, switch to the environment with ReconstructOrder installed 

.. code-block:: bash

    conda activate <your-environment-name>

b) Navigate to the repository folder:

.. code-block:: bash

  cd reconstruct-order

c) Download example dataset:

.. code-block:: bash

  python DownloadExample.py

The example datasets will be downloaded and upzipped in the ```data_downloaded``` folder, together with the configuration files. <br>

d) Run ReconstructOrder on the downloaded dataset, e.g. MouseBrain dataset:

.. code-block:: bash

  python runReconstruction.py --config ./data_downloaded/MouseBrain/config.yml
    
e) The reconstructed images will be saved the ```data_downloaded``` folder. You can reconstruct other downloaded datasets following the above steps, or change the parameters in the configuration file and observe the changes in the output images.


Writing Python Scripts using the ReconstructOrder library
---------------------------------------------------------

(insert some examples here)