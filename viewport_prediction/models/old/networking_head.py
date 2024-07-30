import torch
import torch.nn as nn


class SimpleLinearTaskHead(nn.Module):
    """
    A simple linear layer as task head for NetLLM.

    Note: Task head is the networking head in our paper. It is the early name of our networking head.
    """
    def __init__(self, input_dim, output_dim, fut_window):
        super().__init__()
        self.input_dim = input_dim
        self.fut_window = fut_window
        self.task_head = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=True),  
            nn.Tanh()
        )
    
    def forward(self, input_logits, input_real_lengths):
        last_one = input_logits.shape[1]
        needed_logits = input_logits[:, last_one-1, :]
        needed_logits = torch.unsqueeze(needed_logits, dim=1)
        prediction = self.task_head(needed_logits)
        return prediction
    
    def teacher_forcing(self, input_logits, input_real_lengths):
        size = input_logits.shape[1]
        needed_logits = input_logits[:, size-self.fut_window-1:size-1, :]
        prediction = self.task_head(needed_logits)
        return prediction
    