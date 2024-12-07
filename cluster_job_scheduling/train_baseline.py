import logging
import warnings
from gymnasium import logger

from cfg_loader import load
from trainers import make_trainer


if __name__ == '__main__':
    # filter the annoying warning messages
    warnings.filterwarnings("ignore", module="gymnasium")
    # logging.disable('WARN')
    logger.set_level(logging.INFO)
    # gymnasium.warning = False
    cfg = load()
    make_trainer(cfg).train()
