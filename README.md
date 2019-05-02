# ReconstructOrder
Reconstruct birefringence, slow axis, bright-field, and degree of polarization from polarization-resolved images. The core algorithm employs Stokes representation for reconstruction and background correction. The repository also provides utilities for batch analysis of multi-dimensional datasets acquired with Micro-Manager (https://micro-manager.org/) and OpenPolScope acquisition plugin (https://openpolscope.org/).

## Installation

### Create a new conda environment (optional, but recommended)
>Install conda package management system by installing anaconda or miniconda (https://conda.io/). 
>Creating a conda environment dedicated to ReconstructOrder will avoid version conflicts among packages required by ReconstructOrder and packages required by other python software.
>
>```buildoutcfg
>conda create -n <your-environment-name> python=3.7
>conda activate <your-environment-name>
>```

#### All code blocks below assume you are in the above environment

### Option 1: install released version via pip
>ReconstructOrder is available on pip.  Running pip install will also install dependencies.
>From your environment created above, type:
>```buildoutcfg
>pip install ReconstructOrder
>```

### Option 2: install developer version via git
>Install the git version control system git : https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
>
>Use git to clone this repository to your current directory:
>```buildoutcfg
>git clone https://github.com/czbiohub/ReconstructOrder.git
>```

> * #### install dependencies
>  You have two options to install dependencies: via pip (python index package) or via docker
>
>>  * ##### install dependencies via pip
>>    If you are running ReconstructOrder on your own machine, navigate to the cloned repository 
>>  and install python library dependencies:
>>
>>    ```buildoutcfg
>>    pip install -r requirements.txt
>>    ```

>>  * ##### install dependencies via docker
>>
>>    If you are running ReconstructOrder on a compute node (e.g., fry2@czbiohub), it is recommended to run it in 
a Docker container. 
Docker is the virtual environment with all the required libraries pre-installed so you can run your copy of 
ReconstructOrder without recreating the environment.
The docker image for ReconstructOrder has been built on fry2@czbiohub. 
If you are running ReconstructOrder on other servers, you can build the docker image after cloning the repository 
by doing :    

>>    ```buildoutcfg
>>    docker build -t reconstruct_order:py37 -f Dockerfile.ReconstructOrder .
>>    ```

>>    Now, to start a docker container, do 
>>    ```buildoutcfg
>>    docker run -it  -v /data/<your data dir>/:<data dir name inside docker>/ -v ~/ReconstructOrder:/ReconstructOrder reconstruct_order:py37 bash
>>    ```



## Usage
>To run reconstruction, you will need to create a configuration file.  The configuration file is a .yml file and specifies parameters for:
> * 'dataset'
> * 'processing'
> * 'plotting'
>
> Examples can be found https://github.com/czbiohub/ReconstructOrder under "examples/example_configs" folder
> 
> See /ReconstructOrder/config/config_example.yml for an example config file with detailed explanation of parameters. 
>
> Before running, you should modify the dataset:data_dir and dataset:processed_dir paths to point to source data path and output path, respectively.  Example data is located at examples/example_data 
>
> #### There are two ways to run reconstruction:
>>* #### from command line
>>   If you pip installed the library, from any folder, simply type:
>>   ```buildoutcfg
>>   runReconstruction --config path-and-name-to-your-config.yml
>>   ```
>>
>>   If you cloned the developer repo, navigate to the repo and call the script:
>>   ```buildoutcfg
>>   (C:\ReconstructOrder\) python runReconstruction.py --config path-and-name-to-your-config.yml
>>   ```
>
>>* #### from IPython
>>   If you are writing your own code and want to use the ReconstructOrder library, you can reconstruct as follows:
>>   ```buildoutcfg
>>   import ReconstructOrder.workflow as wf
>>   wf.reconstructBatch('path-and-name-to-your-config.yml')
>>   ```


## License
Chan Zuckerberg Biohub Software License

This software license is the 2-clause BSD license plus clause a third clause
that prohibits redistribution and use for commercial purposes without further
permission.

Copyright © 2019. Chan Zuckerberg Biohub.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1.	Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2.	Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3.	Redistributions and use for commercial purposes are not permitted without
the Chan Zuckerberg Biohub's written permission. For purposes of this license,
commercial purposes are the incorporation of the Chan Zuckerberg Biohub's
software into anything for which you will charge fees or other compensation or
use of the software to perform a commercial service for a third party.
Contact ip@czbiohub.org for commercial licensing opportunities.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
