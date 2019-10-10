Usage
-----
There are two ways to run reconstruction:

**from command line**

    If you pip installed the library, from any folder, simply type:

    .. code-block:: bash

        runReconstruction --config path-and-name-to-your-config.yml

    If you cloned the developer repo, navigate to the repo and call the script:

    .. code-block:: bash

        (C:\ReconstructOrder\) python runReconstruction.py --config path-and-name-to-your-config.yml


**from IPython**

    If you are writing your own code and want to use the ReconstructOrder library, you can reconstruct as follows:

    .. code-block:: python

        import ReconstructOrder.workflow as wf
        wf.reconstructBatch('path-and-name-to-your-config.yml')

