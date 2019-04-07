from workflow.runReconstruction import runReconstruction

configfiles = ['config/config_mousebrainslice.yml', 'config/config_glassbeads.yml','config_mousebrainslice.yml','config_kidneytissue.yml']

if __name__ == '__main__':
    """
    Reconstruct all data supplied with ReconstructOrder repository.
    The total size of dataset is approximately 4GB and will be available for downloaded from the release on github.
    The reconstruction parameters are specified in the configuration files stored in config folder.

    Parameters
    ----------
    Path of config files relative to the root of the repository.
    
    Returns
    --------
    Outputs data to disk.
    """

    for configfile in configfiles:
        print(configfile + '\n--------------')
        runReconstruction(configfile)
        print('\n--------------')


