import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def load(filename=None):
    if not filename:
        args = make_parser().parse_args()
        filename = args.filename

    with open(filename, 'r') as stream:
        cfg = yaml.safe_load(stream)
        
    return cfg


def make_parser():
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="experiment definition file",
        metavar="FILE",
    )

    return parser