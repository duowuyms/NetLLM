"""
Implementation of regression model.
Currently only implement linear regression.
"""
import numpy as np
import torch
import sklearn
from torch import nn
from sklearn.linear_model import LinearRegression as LR


class LinearRegression(nn.Module):
    def __init__(self, in_channels, his_window, fut_window, seed=152, 
                 device=torch.device('cuda') if torch.cuda.is_available() else 'cpu'):
        """
        :param in_channels: number of input channels (3 for 360 video and 6 for volumetric video)
        :param his_window: historical window
        :param fut_window: future (prediction) window
        :param device: cuda or cpu
        """
        super().__init__()

        self.in_channels = in_channels
        self.history_window = his_window
        self.future_window = fut_window
        self.seed = seed
        self.device = device
        sklearn.random.seed(self.seed)

    def forward(self, history, future, tearch_forcing=False):
        """
        :param history: historical viewport trajectory
        :param future: future (ground truth) viewport trajectory
        :param teacher_forcing: no usage, just to align to other models
        :return: prediction, ground truth 
        """
        batch_size = history.shape[0]
        prediction = torch.zeros_like(future).cpu()
        ground_truth = future.cpu()
        for i in range(batch_size):
            pred_all_channels = []
            for c in range(self.in_channels):  # independently predict each channel
                data = history[i, :, c].cpu().numpy()
                pred_channel = []
                for j in range(self.future_window):  # predict in auto-regressive mode (like deep learning models)
                    history_horizon = np.arange(self.history_window + j).reshape(-1, 1)
                    future_horizon = np.arange(self.history_window + j, self.history_window + j + 1).reshape(-1, 1)
                    regressor = LR(fit_intercept=True).fit(history_horizon, data.reshape(-1, 1))
                    pred = regressor.predict(future_horizon)
                    data = np.concatenate([data, pred.reshape(-1)], axis=0)
                    pred = torch.from_numpy(pred).reshape(-1, 1)
                    pred_channel.append(pred[0])
                pred_channel = torch.tensor(pred_channel).unsqueeze(-1)
                pred_all_channels.append(pred_channel)
            prediction[i] = torch.cat(pred_all_channels, dim=-1)
        return prediction, ground_truth
                
    def inference(self, history, future):
        """
        Inference function. Use it for testing.
        """
        return self.forward(history, future, tearch_forcing=False)


def create_linear_regression(his_window, fut_window, device, seed):
    model = LinearRegression(in_channels=3, his_window=his_window, fut_window=fut_window, device=device, seed=seed)
    return model

