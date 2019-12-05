# bchhun, {2019-11-19}

import pytest
import yaml
import os

from ReconstructOrder.utils.ConfigReader import ConfigReader


def test_basic_config_file(setup_temp_yaml):
    path = setup_temp_yaml

    print(path)

    config_reader = ConfigReader(path)

    # dataset
    assert(config_reader.dataset.positions, 'all')
    assert(config_reader.dataset.z_slices, 'all')
    assert(config_reader.dataset.timepoints, 'all')
    assert(config_reader.dataset.ROI, 'all')
    assert(config_reader.dataset.data_dir, os.getcwd()+"/temp_data")

    # processing

    # plotting
