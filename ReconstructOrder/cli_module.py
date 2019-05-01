# bchhun, {5/1/19}


import argparse

from ReconstructOrder.workflow import reconstructBatch


def parse_args():
    """Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                       help='path to yaml configuration file')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reconstructBatch(args.config)
