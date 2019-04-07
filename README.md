# ReconstructOrder
Reconstruct birefringence, slow axis, transmission, and degree of polarization from polarization-resolved images.
The data is acquired with Micro-Manager and OpenPolScope acquisition plugin.

## Getting Started
### Installation 
First, git clone the repository to your home directory. 
If you are running ReconstructOrder on your own machine, you need install the following python libraries:

matplotlib
numpy
cv2

If you are running ReconstructOrder on a compute node (e.g., fry2@czbiohub), it is recommended to run it in a Docker container. 
Docker is the virtual environment with all the required libraries pre-installed so you can run your copy of 
ReconstructOrder without recreating the environment. 
ReconstructOrder can be run inside the imaging Docker container that has been built on fry2. 

To start a docker container, do 
```buildoutcfg
nvidia-docker run -it  -v /data/<your data dir>/:<data dir name inside docker>/ -v ~/ReconstructOrder:/ReconstructOrder imaging_docker:gpu_py36_cu90 bash
```
### Run reconstruction
To run reconstruction, do
```buildoutcfg
python /ReconstructOrder/workflow/runReconstruction.py --config <your config file>.yml
```

See /ReconstructOrder/config/config_example.yml file for an example and explanation of parameters. 
