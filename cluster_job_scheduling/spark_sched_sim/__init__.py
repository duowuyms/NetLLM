from gymnasium.envs.registration import register
from .spark_sched_sim import SparkSchedSimEnv

register(
     id="SparkSchedSimEnv-v0",
     entry_point="spark_sched_sim:SparkSchedSimEnv"
)