"""
Based on code from https://gitlab.com/miguelfromeror/head-motion-prediction/-/blob/master/TRACK_model.py
"""

import torch
import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.func = function
    
    def forward(self, x):
        return self.func(x)
    

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        bt = x.shape[:2]
        bt_as_b = bt[0] * bt[1]
        x_reshape = x.contiguous().view(bt_as_b, *x.shape[2:])  # (samples * timesteps, input_size)
        
        y = self.module(x_reshape)

        # We have to reshape Y
        y = y.contiguous().view(bt[0], bt[1], *y.shape[1:])  # (samples, timesteps, output_size)
        
        return y


def toPosition(values):
    orientation = values[0]
    # The network returns values between 0 and 1, we force it to be between -1/2 and 1/2
    motion = values[1]
    return (orientation + motion)


class Track(nn.Module):
    def __init__(
            self, 
            M_WINDOW: int,           # M_WINDOW=history window
            H_WINDOW: int,           # H_WINDOW=future window
            NUM_TILES_HEIGHT: int,   # NUM_TILES_HEIGHT
            NUM_TILES_WIDTH: int     # NUM_TILES_WIDTH
            ):
        
        super().__init__()

        self.M_WINDOW = M_WINDOW
        self.H_WINDOW = H_WINDOW
        self.nt_height = NUM_TILES_HEIGHT
        self.nt_width = NUM_TILES_WIDTH
        self.sal_size = NUM_TILES_HEIGHT * NUM_TILES_WIDTH

        self.sense_pos_enc = nn.LSTM(3, 256, batch_first=True)
        self.sense_sal_enc = nn.LSTM(self.sal_size, 256, batch_first=True)
        self.fuse_1_enc = nn.LSTM(512, 256, batch_first=True)
        self.sense_pos_dec = nn.LSTM(3, 256, batch_first=True)
        self.sense_sal_dec = nn.LSTM(self.sal_size, 256, batch_first=True)
        self.fuse_1_dec = nn.LSTM(512, 256, batch_first=True)
        self.fuse_2 = nn.Linear(256, 256)
        self.fc_layer_out = nn.Linear(256, 3)
        
        self.To_Position = Lambda(toPosition)

    def forward(
            self, 
            encoder_position_inputs: torch.Tensor,  # [B, M_WINDOW, 3]
            encoder_saliency_inputs: torch.Tensor,  # [B, M_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, 1]
            decoder_position_inputs: torch.Tensor,  # [B, 1, 3]
            decoder_saliency_inputs: torch.Tensor,  # [B, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, 1]
            ) -> torch.Tensor:

        # Encoding
        out_enc_pos, states_1 = self.sense_pos_enc(encoder_position_inputs)
        out_flat_enc = TimeDistributed(nn.Flatten())(encoder_saliency_inputs).float()

        out_enc_sal, states_2 = self.sense_sal_enc(out_flat_enc)
        
        conc_out_enc = torch.cat([out_enc_sal, out_enc_pos], dim=-1)

        fuse_out_enc, states_fuse = self.fuse_1_enc(conc_out_enc)

        # Decoding
        all_pos_outputs = []
        inputs = decoder_position_inputs
        for curr_idx in range(self.H_WINDOW):
            out_enc_pos, states_1 = self.sense_pos_dec(inputs, states_1)

            flatten_timestep_saliency = decoder_saliency_inputs[:, curr_idx:curr_idx+1]\
                                            .contiguous().view(decoder_saliency_inputs.shape[0], 1, self.sal_size).float()

            out_enc_sal, states_2 = self.sense_sal_dec(flatten_timestep_saliency, states_2)

            conc_out_dec = torch.cat([out_enc_sal, out_enc_pos], axis=-1)

            fuse_out_dec_1, _ = self.fuse_1_dec(conc_out_dec, states_fuse)
            fuse_out_dec_2 = TimeDistributed(self.fuse_2)(fuse_out_dec_1)

            outputs_delta = self.fc_layer_out(fuse_out_dec_2)
            decoder_pred = self.To_Position([inputs, outputs_delta])

            all_pos_outputs.append(decoder_pred)
            
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = decoder_pred

        # Concatenate all predictions
        decoder_outputs_pos = torch.cat(all_pos_outputs, axis=1)
        return decoder_outputs_pos


def create_track(M_WINDOW, H_WINDOW, device, HEIGHT, WIDTH):
    device = torch.device(device)
    model = Track(M_WINDOW, H_WINDOW, HEIGHT, WIDTH)
    model.to(device)
    return model
