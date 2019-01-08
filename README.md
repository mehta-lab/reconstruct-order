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
Docker is the virtual environment with all the required libraries pre-installed so you can run your copy of ReconstructOrder without recreating the environment. 
ReconstructOrder can be run inside the imaging Docker container that has been built on fry2. 

To start a docker container, do 
```buildoutcfg
nvidia-docker run -it  -v /data/<your data dir>/:<data dir name inside docker>/ -v ~/ReconstructOrder:/ReconstructOrder imaging_docker:gpu_py36_cu90 bash
```
### Run reconstruction
To run reconstruction, do
```buildoutcfg
python /ReconstructOrder/birefringence_python/run_reconstruction.py --config <your config file>.yml
```
The following options are available in run_reconstruction.py
 
* outputChann: (list of str) output channel names
    Current available output channel names:
        'Transmission'
        'Retardance'
        'Orientation' 
        'Retardance+Orientation'
        'Transmission+Retardance+Orientation'
        'Retardance+Fluorescence'
        '405'
        '488'
        '568'
        '640'
        
* flipPol: (bool) flip the slow axis horizontally. Set "True" for Dragonfly and "False" for ASI 
* bgCorrect: (str) 
    'Auto' (default) to correct the background using background from the metadata if available, otherwise use input background folder;
    'None' for no background correction; 
    'Input' to always use input background folder
    'Local' apply additional background correction using local background estimated from sample images  
* flatField: (bool) perform flat-field correction on fluorescence images if True
* norm: (bool) scale fluorescence images for each image for optimal contrast. Set False to use the same scale for all images
* batchProc: (bool) batch process all the folders in ImgDir if True. 
