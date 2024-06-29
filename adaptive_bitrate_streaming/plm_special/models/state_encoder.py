"""
Customized state encoder based on Pensieve's encoder.
"""
import torch.nn as nn


class EncoderNetwork(nn.Module):
    """
    The encoder network for encoding each piece of information of the state.
    This design of the network is from Pensieve/Genet.
    """
    def __init__(self, conv_size=4, bitrate_levels=6, embed_dim=128):
        super().__init__()
        self.past_k = conv_size
        self.bitrate_levels = 6
        self.embed_dim = embed_dim
        self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # last bitrate
        self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # current buffer size
        self.conv3 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k throughput
        self.conv4 = nn.Sequential(nn.Conv1d(1, embed_dim, conv_size), nn.LeakyReLU(), nn.Flatten())  # past k download time
        self.conv5 = nn.Sequential(nn.Conv1d(1, embed_dim, bitrate_levels), nn.LeakyReLU(), nn.Flatten())  # next chunk sizes
        self.fc6 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # remain chunks        


    def forward(self, state):
        # state.shape: (batch_size, seq_len, 6, 6) -> (batch_size x seq_len, 6, 6)
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = state.reshape(batch_size * seq_len, 6, 6)
        
        last_bitrate = state[..., 0:1, -1]
        current_buffer_size = state[..., 1:2, -1]
        throughputs = state[..., 2:3, :]
        download_time = state[..., 3:4, :]
        next_chunk_size = state[..., 4:5, :self.bitrate_levels]
        remain_chunks = state[..., 5:6, -1]
        
        features1 = self.fc1(last_bitrate).reshape(batch_size, seq_len, -1)
        features2 = self.fc2(current_buffer_size).reshape(batch_size, seq_len, -1)
        features3 = self.conv3(throughputs).reshape(batch_size, seq_len, -1)
        features4 = self.conv4(download_time).reshape(batch_size, seq_len, -1)
        features5 = self.conv5(next_chunk_size).reshape(batch_size, seq_len, -1)
        features6 = self.fc6(remain_chunks).reshape(batch_size, seq_len, -1)
        return features1, features2, features3, features4, features5, features6
