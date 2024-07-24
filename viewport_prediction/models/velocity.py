import torch
import torch.nn as nn


class VelocityMethod(nn.Module):
    r'''
    Velocity-based method for viewport prediction.
    '''
    def __init__(self,
                 fut_window_length,
                ):
        super().__init__()
        self.fut_window_length = fut_window_length
        self.outputlist = []

    def forward(self, x) -> torch.Tensor:
        """
        :param x: history viewport trajectory
        :return: the predict viewport trajectory
        """
        m = x.shape[1]
        v = (x[0][m - 1] - x[0][0]) / (m-1)
        self.outputlist = []
        for i in range(self.fut_window_length):
            if len(self.outputlist):
                outputs = self.outputlist[-1] + v
                self.outputlist.append(outputs)
            else:
                outputs = x[0][-1] + v
                self.outputlist.append(outputs.unsqueeze(0).unsqueeze(0))
        return self.outputlist
    
    def inference(self, batch, future) -> torch.Tensor:
        """
        Inference function. Use it for testing.
        """
        outputs = self.forward(batch)
        pred = torch.cat(outputs, dim=1)
        gt = future.to(pred.device)
        return pred, gt
    
    
def create_velocity(fut_window):
    return VelocityMethod(fut_window)