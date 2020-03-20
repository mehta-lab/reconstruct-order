.. Reconstruct Order documentation master file, created by
   sphinx-quickstart on Wed Apr 17 17:25:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Reconstruct Order
=============================================
Reconstruct birefringence, slow axis, transmission, and degree of polarization from polarization-resolved images.
The data is acquired with Micro-Manager and OpenPolScope acquisition plugin.


Quick start
-----------

ReconstructOrder is available on Python Package Index using Pip.
We highly recommend you install to a separate environment

.. code-block:: bash

   # USING venv
   python3 -m venv /path/to/new/virtual/environment
   cd /path/to/new/virtual/environment
   activate

.. code-block:: bash

   # USING anaconda
   # from anaconda command prompt, or terminal
   conda create -n name-of-new-environment
   source activate name-of-new-environment

Then install ReconstructOrder to this environment

.. code-block:: bash

   pip install ReconstructOrder

If you wish to run ReconstructOrder from command line, and not from the python interpreter,
you will need to clone this github repo, and run commands from within.

.. code-block:: bash

   git clone https://github.com/czbiohub/ReconstructOrder.git


Running Reconstruction on your data
-----------------------------------

To reconstruct birefringence images, you will need to create a configuration file that reflects your experiment's
parameters.  You can see example configurations in this github repo under examples/example_configs.

Modify paths to your data in there.  See "config_example.yml" for detailed description of the fields.  It's important
that your data is organized in a hierarchy as described.

finally, when the config file is ready, run the following:

FROM PYTHON

.. code-block:: python

   from ReconstructOrder import workflow as wf

   wf.runReconstruction('path_to_your_config_file')


FROM COMMAND LINE

.. code-block:: bash

   # first navigate to your cloned ReconstructOrder directory
   cd ReconstructOrder
   python runReconstruction.py --config path_to_your_config_file


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   introduction
   installation
   usage
   compute
   datastructures
   plotting
   utils
   workflow
   LICENSE


Thanks
------

This work is made possible by the Chan-Zuckerberg Biohub



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
