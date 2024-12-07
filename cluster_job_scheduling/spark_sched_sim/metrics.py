import numpy as np


def job_durations(env):
    durations = []
    while hasattr(env, 'env'):
        env = env.env
    for job_id in env.active_job_ids + list(env.completed_job_ids):
        job = env.jobs[job_id]
        t_end = min(job.t_completed, env.wall_time)
        durations += [t_end - job.t_arrival]
    return durations


def avg_job_duration(env=None, jd=None):
    if jd is None:
        assert env is not None, 'env and jd cannot both be None.'
        return np.mean(job_durations(env))
    return np.mean(jd)


def avg_num_jobs(env):
    return sum(job_durations(env)) / env.wall_time


def job_duration_percentiles(env=None, jd=None):
    if jd is None:
        assert env is not None, 'env and jd cannot both be None.'
        jd = job_durations(env)
        return np.percentile(jd, [25, 50, 75, 100])
    return np.percentile(jd, [25, 50, 75, 100])