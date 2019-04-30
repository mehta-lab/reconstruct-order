# ReconstructOrder
Reconstruct birefringence, slow axis, bright-field, and degree of polarization from polarization images acquired with 
Micro-Manager and OpenPolScope acquisition plugin.

## Getting Started

### Install through pip install 

### Install through git clone 
First, git clone the repository to your home directory by doing:

```buildoutcfg
git clone https://github.com/czbiohub/ReconstructOrder.git
```
 
If you are running ReconstructOrder on your own machine, install the python library dependency by:

```buildoutcfg
pip install -r requirements.txt
```

If you are running ReconstructOrder on a compute node (e.g., fry2@czbiohub), it is recommended to run it in 
a Docker container. 
Docker is the virtual environment with all the required libraries pre-installed so you can run your copy of 
ReconstructOrder without recreating the environment.
The docker image for ReconstructOrder has been built on fry2@czbiohub. 
If you are running ReconstructOrder on other servers, you can build the docker image after cloning the repository 
by doing :    

```buildoutcfg
docker build -t reconstruct_order:py37 -f Dockerfile.ReconstructOrder .
```

Now, to start a docker container, do 
```buildoutcfg
docker run -it  -v /data/<your data dir>/:<data dir name inside docker>/ -v ~/ReconstructOrder:/ReconstructOrder reconstruct_order:py37 bash
```
### Run reconstruction
To run reconstruction, go to ReconstructOrder repository directory (e.g. /ReconstructOrder) and do
```buildoutcfg
python runReconstruction.py --config <your config file>.yml
```

See /ReconstructOrder/config/config_example.yml for an example config file and explanation of parameters. 

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
