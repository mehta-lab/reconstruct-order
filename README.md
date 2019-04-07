# ReconstructOrder
Reconstruct retardance, orientation, scattering images from polarized images output by Open PolScope

## Getting Started
### Installation 
First, git clone the repository to your home directory. 
If you are running ReconstructOrder on your own machine, you need install the following python libraries:

matplotlib
numpy
cv2

If you are running ReconstructOrder on fry2, it is recommended to run it in a Docker container. 
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

See config/config.yml file for example and explanation. 
