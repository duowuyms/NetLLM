import os
import torch
import torch.nn as nn
from typing import *
from transformers.utils.dummy_pt_objects import PreTrainedModel
from config import cfg


class Pipeline(nn.Module):
    '''
    Pipeline for viewport prediction.
    '''
    def __init__(self,
                plm: PreTrainedModel,
                loss_func = None,
                fut_window = None,
                device = 'cuda',
                embed_size = 1024,
                frequency = 5,
                using_multimodal = False,
                dataset = None
                ):
        """
        :param plm: the pretrained llm
        :param embed_size: the embed size of llm
        :param frequency: the frequency of dataset
        :param fut_window: future (prediction) window
        :param dataset: the dataset
        :param using_multimodal: adding multimodal image features (True/False)
        :param device: cuda or cpu
        """
        super().__init__()
        self.plm = plm
        self.using_multimodal = using_multimodal
        self.dataset = dataset
        self.device = device
        self.frequency = frequency
        self.embed_size = embed_size
        self.fut_window_length = fut_window

        self.conv1d = nn.Sequential(nn.Conv1d(1, 256, 3), nn.LeakyReLU(), nn.Flatten()).to(device)
        self.embed_vp = nn.Linear(256, self.embed_size).to(device)
        self.embed_multimodal = nn.Linear(768, embed_size).to(device)  # 768 = ViT output feature size
        self.embed_ln = nn.LayerNorm(self.embed_size).to(device)

        self.loaded_tensor_cache = {}
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.embed_vp, self.embed_multimodal, self.embed_ln, self.conv1d, self.plm.networking_head
        ])

        if loss_func is None:
            loss_func = nn.MSELoss()
        self.loss_fct = loss_func
        self.fut_window = fut_window
    
    def forward(self, batch, future, video_user_position, teacher_forcing=True) -> torch.Tensor:
        """
        :param batch: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the loss value for training
        """
        if teacher_forcing:
            pred = self.teaching_forcing(batch, future, video_user_position)
        else:
            pred = self.auto_regressive(batch, future, video_user_position)
        gt = future.to(pred.device)
        loss = self.loss_fct(pred, gt)
        return loss
    
    def auto_regressive(self, x, future, video_user_position) -> torch.Tensor:
        """
        auto-regressive generation
        
        :return: the loss value for training
        """
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :]).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:  # we make using multimodal image features as an option, as not all datasets provide video information.
            mapped_tensor = self.get_multimodal_information(video_user_position)
            x = torch.cat([mapped_tensor, x], dim=1)

        x = self.embed_ln(x)

        outputlist = []
        for _ in range(self.fut_window_length):
            outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device))
            outputlist.append(outputs.logits)
            x = torch.cat((x, self.embed_vp(self.conv1d(outputs.logits)).unsqueeze(1)), dim=1)

        pred = torch.cat(outputlist, dim=1)
        return pred
    
    def teaching_forcing(self, x, future, video_user_position) -> torch.Tensor:
        """
        teaching-forcing generation

        :param x: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the return value by llm
        """

        x = torch.cat((x, future), dim=1)
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :]).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:
            mapped_tensor = self.get_multimodal_information(video_user_position)
            x = torch.cat([mapped_tensor, x], dim=1)
        
        x = self.embed_ln(x)

        outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device), teacher_forcing=True)
        return outputs.logits
    
    def inference(self, batch, future, video_user_info) -> torch.Tensor:
        """
        Inference function. Use it for testing.
        """
        pred = self.auto_regressive(batch, future, video_user_info)
        gt = future.to(pred.device)
        return pred, gt
    
    def get_multimodal_information(self, video_user_position):
        """
        Get the corresponding image content features.
        Note that we use ViT to extract image features (the first output features of ViT that contains the overall information of the image).
        Since we use the frozen ViT for image feature extraction, we can actually use ViT to extract features first,
        then store all features into disk, and fetch them when needed.
        This way, we can avoid repeatedly using ViT to extract features for the same images.
        As a result, we can speed up the training process.
        
        TODO: Support on-the-fly image feature extraction with ViT.

        :param video_user_position: details information for current trajectory
        return: the embedding of the image
        """
        video_index = video_user_position[0].item()
        position_index = video_user_position[2].item()
        image_index = (position_index - 1) * (cfg.video_frame[self.dataset][video_index-1]//self.frequency)
        # add cache_key
        if image_index % 100 == 0:
            cache_key = f'{video_index}_{image_index//100}'
        else:
            cache_key = f'{video_index}_{(image_index//100)+1}'
        if cache_key in self.loaded_tensor_cache:
            loaded_tensor_dict = self.loaded_tensor_cache[cache_key]
        else:
            if image_index % 100 == 0:
                loaded_tensor_dict = torch.load(os.path.join(cfg.dataset_image_features[self.dataset], f'video{video_index}_images/feature_dict{(image_index//100)}.pth'))
            else:
                loaded_tensor_dict = torch.load(os.path.join(cfg.dataset_image_features[self.dataset], f'video{video_index}_images/feature_dict{(image_index//100) + 1}.pth'))
        
        self.loaded_tensor_cache[cache_key] = loaded_tensor_dict  # add to loaded_tensor_dict
        load_tensor = loaded_tensor_dict[f'{image_index}'].to(self.device)
        mapped_tensor = self.embed_multimodal(load_tensor)
        mapped_tensor = mapped_tensor.unsqueeze(1)
        return mapped_tensor