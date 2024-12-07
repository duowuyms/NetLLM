import numpy as np

class Baseline:
    def __init__(self, num_sequences, num_rollouts):
        self.num_sequences = num_sequences
        self.num_rollouts = num_rollouts


    def __call__(self, ts_list, ys_list):
        return self.average(ts_list, ys_list)


    def average(self, ts_list, ys_list):
        baseline_list = []
        for j in range(self.num_sequences):
            start = j * self.num_rollouts
            end = start + self.num_rollouts
            baseline_list += self._average(
                ts_list[start:end], ys_list[start:end])
        return baseline_list


    def _average(self, ts_list, ys_list):
        ts_unique = np.unique(np.hstack(ts_list))

        # shape: (num envs, len(ts_unique))
        # y_hats[i, t] is the linear interpolation of (ts_list[i], ys_list[i]) 
        # at time t
        y_hats = np.vstack([
            np.interp(ts_unique, ts, ys) 
            for ts, ys in zip(ts_list, ys_list)
        ])

        # find baseline at each unique time point
        baseline = {}
        for t, y_hat in zip(ts_unique, y_hats.T):
            baseline[t] = y_hat.mean()

        baseline_list = [
            np.array([baseline[t] for t in ts]) 
            for ts in ts_list]

        return baseline_list