from abc import ABC, abstractmethod

from gymnasium.core import ObsType, ActType



class Scheduler(ABC):
    '''Base class for all schedulers'''

    def __init__(self, name: str):
        self.name = name


    def __call__(self, obs: ObsType) -> ActType:
        return self.schedule(obs)
        

    @abstractmethod
    def schedule(self, obs: ObsType) -> ActType:
        pass