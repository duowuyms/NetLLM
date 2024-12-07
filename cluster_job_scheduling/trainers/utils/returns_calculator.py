from itertools import chain
import numpy as np

class CircularArray:
    def __init__(self, cap, num_cols):
        self.cap = cap
        self.data = np.zeros((cap, num_cols))

    def extend(self, new_data):
        num_new = new_data.shape[0]
        if num_new > self.cap:
            new_data = new_data[-self.cap:]
            num_new = self.cap

        num_keep = self.cap - num_new
        if num_keep > 0:
            self.data[:num_keep] = self.data[-num_keep:]

        self.data[num_keep:] = new_data



class ReturnsCalculator:
    def __init__(self, buff_cap=None, beta=None):
        assert bool(buff_cap) ^ bool(beta), \
            'exactly one of `buff_cap` and `beta` must be specified'
        
        self.buff_cap = buff_cap
        self.beta = beta

        # estimate of the long-run average number of concurrent jobs under 
        # the current policy
        self.avg_num_jobs = None
        
        if buff_cap:
            # circular buffer used for computing the moving average. each row 
            # corresponds to a time-step; the first column is the duration of 
            # that step in ms, and the second column is the reward from that 
            # step
            self.buff = CircularArray(buff_cap, num_cols=2)

        
    def __call__(self, rewards_list, times_list, resets_list):
        dt_list = [np.array(ts[1:]) - np.array(ts[:-1]) for ts in times_list]

        if self.beta:
            return self._calc_discounted_returns(dt_list, rewards_list)
        else:
            return self._calc_differential_returns(dt_list, rewards_list)
    

    def _calc_differential_returns(self, dt_list, rewards_list):
        self._update_avg_num_jobs(dt_list, rewards_list)

        diff_returns_list = []
        for dts, rs in zip(dt_list, rewards_list):
            diff_returns = np.zeros(len(rs))
            R = 0
            for k, (dt, r) in reversed(list(enumerate(zip(dts, rs)))):
                job_time = -r
                expected_job_time = dt * self.avg_num_jobs
                R = -(job_time - expected_job_time) + R
                diff_returns[k] = R
            diff_returns_list += [diff_returns]
        return diff_returns_list
    

    def _calc_discounted_returns(self, dt_list, rewards_list):
        disc_returns_list = []
        for dts, rs in zip(dt_list, rewards_list):
            disc_returns = np.zeros(len(rs))
            R = 0
            for k, (dt, r) in reversed(list(enumerate(zip(dts, rs)))):
                R = r + np.exp(-self.beta * 1e-3 * dt) * R
                disc_returns[k] = R
            disc_returns_list += [disc_returns]
        return disc_returns_list


    def _update_avg_num_jobs(self, deltas_list, rewards_list):
        new_data = np.array(list(zip(chain(*deltas_list), 
                                     chain(*rewards_list))))

        # filter out timesteps that have a duration of 0ms
        new_data = new_data[new_data[:,0] > 0]

        # add new data to circular buffer, discarding some of the old data
        self.buff.extend(new_data)

        total_time, rew_sum = self.buff.data.sum(0)
        total_job_time = -rew_sum
        self.avg_num_jobs = total_job_time / total_time