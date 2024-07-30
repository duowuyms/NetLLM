import os
import torch
import torch.nn as nn
from typing import *
from transformers.utils.dummy_pt_objects import PreTrainedModel
from config import cfg

class EmbeddingModelViewportPrediction(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel,
                 device = 'cuda',
                 embed_size = 1024,
                 frequency = 5,
                 fut_window_length = 10,
                 using_teaching_forcing = False,
                 using_multimodal = False,
                 dataset = None
                ):
        super().__init__()
        self.plm = plm
        self.device = device
        self.frequency = frequency
        self.embed_size = embed_size
        self.fut_window_length = fut_window_length

        self.embed_ln = nn.LayerNorm(self.embed_size).to(device)
        self.conv1d1 = nn.Sequential(nn.Conv1d(1, 256, 3), nn.LeakyReLU(), nn.Flatten()).to(device)
        self.linear_layer = nn.Linear(256, self.embed_size).to(device)
        self.linear_layer_for_multimodal = nn.Linear(768, embed_size).to(device)

        self.outputlist = []
        self.using_teaching_forcing = using_teaching_forcing
        self.using_multimodal = using_multimodal
        self.dataset = dataset
        self.loaded_tensor_cache = {}
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.linear_layer, self.linear_layer_for_multimodal, self.embed_ln, self.conv1d1, self.plm.task_head
        ])

    def forward(self, x, video_user_position) -> torch.Tensor:
        """
        using auto regressive

        :param x: history viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the return value by llm
        """

        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.linear_layer(self.conv1d1(x[:, i, :]).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:
            mapped_tensor = self.get_multimodal_information(video_user_position)
            x = torch.cat([mapped_tensor, x], dim=1)

        x = self.embed_ln(x)

        self.outputlist = []
        for _ in range(self.fut_window_length):
            outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device))
            self.outputlist.append(outputs.logits)
            x = torch.cat((x, self.linear_layer(self.conv1d1(outputs.logits)).unsqueeze(1)), dim=1)
        return self.outputlist

    def teaching_forcing(self, x, future, video_user_position) -> torch.Tensor:
        """
        using teaching-forcing

        :param x: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the return value by llm
        """

        x = torch.cat((x, future), dim=1)
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.linear_layer(self.conv1d1(x[:, i, :]).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:
            mapped_tensor = self.get_multimodal_information(video_user_position)
            x = torch.cat([mapped_tensor, x], dim=1)
        
        x = self.embed_ln(x)

        outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device), teacher_forcing = self.using_teaching_forcing)
        return outputs
    
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
        
        self.loaded_tensor_cache[cache_key] = loaded_tensor_dict  #add to loaded_tensor_dict
        load_tensor = loaded_tensor_dict[f'{image_index}'].to(self.device)
        mapped_tensor = self.linear_layer_for_multimodal(load_tensor)
        mapped_tensor = mapped_tensor.unsqueeze(1)
        return mapped_tensor

class EmbeddingForViewportPrediction(nn.Module):
    r'''
    Embedding pipeline for viewport prediction.
    '''
    def __init__(self,
                plm: PreTrainedModel,
                loss_func = None,
                fut_window = None,
                device = 'cuda',
                embed_size = 1024,
                frequency = 5,
                using_teaching_forcing = False,
                using_multimodal = False,
                dataset = None
                ):
        """
        :param plm: the pretrained model
        :param embed_size: the embed size of plm
        :param frequency: the frequency of dataset
        :param fut_window: future (prediction) window
        :param dataset: the dataset
        :param using_teaching_forcing: using teacher forcing(True/False)
        :param using_multimodal: adding multimodal(True/False)
        :param device: cuda or cpu
        """
        super().__init__()
        self.plm = plm
        self.using_teaching_forcing = using_teaching_forcing
        self.using_multimodal = using_multimodal
        self.dataset = dataset
        self.embedding_model = EmbeddingModelViewportPrediction(plm, device, embed_size, frequency, fut_window, using_teaching_forcing, using_multimodal, dataset)

        if loss_func is None:
            loss_func = nn.MSELoss()
        self.loss_fct = loss_func
        self.fut_window = fut_window

    @property
    def device(self):
        return self.plm.device
    
    def forward(self, batch, future, video_user_position) -> torch.Tensor:
        """
        :param batch: history viewport trajectory
        :param future: future viewport trajectory
        :param video_user_position: details information for current trajectory
        :return: the loss value for training
        """
        if self.using_teaching_forcing:
            outputs = self.embedding_model.teaching_forcing(batch, future, video_user_position)
            pred = outputs.logits
        else:
            outputs = self.embedding_model(batch, video_user_position)
            pred = torch.cat(outputs, dim=1)
        # pred = outputs.logits
        gt = future.to(pred.device)
        loss = self.loss_fct(pred, gt)
        return loss
    
    def autoregressive(self, batch, future, video_user_position) -> torch.Tensor:
        """
        auto regressive function
        
        :return: the loss value for training
        """
        outputs = self.embedding_model(batch, video_user_position)
        pred = torch.cat(outputs, dim=1)
        gt = future.to(pred.device)
        loss = self.loss_fct(pred, gt)
        return loss
    
    def inference(self, batch, future, video_user_info) -> torch.Tensor:
        """
        Inference function. Use it for testing.
        """
        outputs = self.embedding_model(batch, video_user_info)
        pred = torch.cat(outputs, dim=1)
        gt = future.to(pred.device)
        return pred, gt