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
python /ReconstructOrder/run_reconstruction.py --config <your config file>.yml
```
The config file contains the following parameters:
* RawDataPath: (str) The path of the parent folder that holds raw data 
* ProcessedPath: (str) The path of the parent folder where the output will be saved
* ImgDir: (str) The experiment folder within the RawDataPath to process  
* SmDir: (str or list) Acquisition folder(s) within the experiment folder to process
* BgDir: (str or list) Background folder(s) within the experiment folder
* BgDir_local: (str or None) Only used for 'Local_defocus'. Set None otherwise

The full path to the data should be RawDataPath/ImgDir/SmDir/Pos$
  
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
        
* circularity: ('lcp' or 'rcp') flip the slow axis horizontally. 
* bgCorrect: (str) 
    'Auto' (default) to correct the background using background from the metadata if available, otherwise use input background folder;
    'None' for no background correction; 
    'Input' to always use input background folder
    'Local_filter' apply additional background correction using local background estimated from Gaussian-blurred sample images
    'Local_defocus' use local defocused images from 'BgDir_local' folder. The background images must have exactly same position indices as sample images.    
* flatField: (bool) perform flat-field correction on fluorescence images if True
* norm: (bool) scale fluorescence images for each image for optimal contrast. Set False to use the same scale for all images
* batchProc: (bool) batch process all the folders in ImgDir if True. 
