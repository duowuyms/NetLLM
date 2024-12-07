from .vpg import VPG
from .ppo import PPO

def make_trainer(cfg):
    glob = globals()
    trainer_cls = cfg['trainer']['trainer_cls']
    assert trainer_cls in glob, \
        f"'{trainer_cls}' is not a valid trainer."
    return glob[trainer_cls](agent_cfg=cfg['agent'],
                             env_cfg=cfg['env'],
                             train_cfg=cfg['trainer'])