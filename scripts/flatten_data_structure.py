"""Covert data structure from hierarchical to flat"""
from ReconstructOrder.utils import copy_files_in_sub_dirs
import argparse
import os
from shutil import copy2

def parse_args():
    """Parse command line arguments
    :return: namespace containing the arguments passed.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str,
                       help='path to input directory')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input_path = args.input
    output_path = ''.join([input_path, '_flat'])
    copy_files_in_sub_dirs(input_path, output_path)
    copy2(os.path.join(input_path, 'meta'), output_path)
