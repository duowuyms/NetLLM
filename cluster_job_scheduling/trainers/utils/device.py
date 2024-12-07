import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')