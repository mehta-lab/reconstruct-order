Installation
==============

create an environment
---------------------
Create a new conda environment (optional, but recommended)
Install conda package management system by installing anaconda or miniconda `link <https://conda.io/>`_.
Creating a conda environment dedicated to ReconstructOrder will avoid version conflicts among packages required by ReconstructOrder and packages required by other python software.

.. code-block:: bash

    conda create -n <your-environment-name> python=3.7
    conda activate <your-environment-name>

get ReconstructOrder
--------------------
*All code blocks below assume you are in the above environment*

**Option 1**: install released version via pip
ReconstructOrder is available on pip.  Running pip install will also install dependencies.
From your environment created above, type:


.. code-block:: bash

    pip install ReconstructOrder


**Option 2**: install developer version via git
Install the git version control system git : `link <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_

Use git to clone this repository to your current directory:

.. code-block:: bash

    git clone https://github.com/mehta-lab/reconstruct-order.git


You can install dependencies via pip (python index package) or run ReconstructOrder inside a docker container with the dependencies pre-installed

 * install dependencies via pip

    If you are running ReconstructOrder on your own machine,

    a) navigate to the cloned repository:

    .. code-block:: bash

        cd reconstruct-order

    b) install python library dependencies:

    .. code-block:: bash

        pip install -r requirements.txt

    c) Create a symbolic library link with setup.py:

    .. code-block:: bash

        python setup.py develop


 * Running inside a docker container

   If you are running ReconstructOrder on a compute node (e.g., fry2@czbiohub), it is recommended to run it in a Docker container.
   Docker is the virtual environment with all the required libraries pre-installed so you can run your copy of ReconstructOrder without recreating the environment.

   The docker image for ReconstructOrder has been built on fry2@czbiohub.

   If you are running ReconstructOrder on other servers, you can build the docker image after cloning the repository by doing :

    .. code-block:: bash

        docker build -t reconstruct_order:py37 -f Dockerfile.ReconstructOrder .


    Now, to start a docker container, do

    .. code-block:: bash

        docker run -it  -v /data/<your data dir>/:<data dir name inside docker>/ -v ~/ReconstructOrder:/ReconstructOrder reconstruct_order:py37 bash

ReconstructOrder supports NVIDIA GPU computation through cupy package, please follow `here <https://github.com/cupy/cupy>`_ for installation (check cupy is properly installed by ```import cupy```).
----------


To enable gpu processing, set ```processing: use_gpu: True``` in the configuration file.
