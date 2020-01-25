# bchhun, {2019-12-12}

"""
these tests check that metadata from different micromanager versions is parsed properly
these tests will also run a full reconstruction using those metadata values


"""
from ReconstructOrder.workflow import reconstruct_batch
import pytest


# ==== mm1.4.22 metadata tests ===============

def test_reconstruct_mm1422_kazansky_grid(setup_mm1422_background_src, setup_mm1422_kazansky_grid_src):
    _ = setup_mm1422_background_src
    config = setup_mm1422_kazansky_grid_src
    try:
        reconstruct_batch(config)
    except Exception as ex:
        pytest.fail("Exception thrown during reconstruction = "+str(ex))


def test_reconstruct_mm1422_kazansky_HCS_snake(setup_mm1422_background_src, setup_mm1422_kazansky_HCS_snake_src):
    _ = setup_mm1422_background_src
    config = setup_mm1422_kazansky_HCS_snake_src
    try:
        reconstruct_batch(config)
    except Exception as ex:
        pytest.fail("Exception thrown during reconstruction = "+str(ex))


def test_reconstruct_mm1422_kazansky_one_position(setup_mm1422_background_src, setup_mm1422_kazansky_one_position_src):
    _ = setup_mm1422_background_src
    config = setup_mm1422_kazansky_one_position_src
    try:
        reconstruct_batch(config)
    except Exception as ex:
        pytest.fail("Exception thrown during reconstruction = "+str(ex))


# ==== mm2.0 gamma metadata tests =============

def test_reconstruct_mm2_gamma_kazansky_grid(setup_mm2_gamma_background_src, setup_mm2_gamma_kazansky_grid_src):
    _ = setup_mm2_gamma_background_src
    config = setup_mm2_gamma_kazansky_grid_src
    try:
        reconstruct_batch(config)
    except Exception as ex:
        pytest.fail("Exception thrown during reconstruction = "+str(ex))


def test_reconstruct_mm2_gamma_kazansky_HCS_one_position(setup_mm2_gamma_background_src, setup_mm2_gamma_kazansky_HCS_one_position_src):
    _ = setup_mm2_gamma_background_src
    config = setup_mm2_gamma_kazansky_HCS_one_position_src
    try:
        reconstruct_batch(config)
    except Exception as ex:
        pytest.fail("Exception thrown during reconstruction = "+str(ex))


def test_reconstruct_mm2_gamma_kazansky_HCS_snake(setup_mm2_gamma_background_src, setup_mm2_gamma_kazansky_HCS_snake_src):
    _ = setup_mm2_gamma_background_src
    config = setup_mm2_gamma_kazansky_HCS_snake_src
    try:
        reconstruct_batch(config)
    except Exception as ex:
        pytest.fail("Exception thrown during reconstruction = "+str(ex))


def test_reconstruct_mm2_gamma_kazansky_HCS_typewriter(setup_mm2_gamma_background_src, setup_mm2_gamma_kazansky_HCS_typewriter_src):
    _ = setup_mm2_gamma_background_src
    config = setup_mm2_gamma_kazansky_HCS_typewriter_src
    try:
        reconstruct_batch(config)
    except Exception as ex:
        pytest.fail("Exception thrown during reconstruction = "+str(ex))