from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class GPT2NetworkingHeadModel(GPT2LMHeadModel):
    """
    GPT2 with networking head.
    This class is implemented based on GPT2LMHeadModel.
    """
    _tied_weights_keys = ["networking_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.n_embd
        self.vocab_size = config.vocab_size
        self.networking_head = None

    def get_networking_head(self):
        return self.networking_head

    def set_networking_head(self, networking_head):
        self.networking_head = networking_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_ids_len: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher_forcing: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Reimplement forward method for viewport prediction.
        """
        
        assert self.networking_head is not None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        if teacher_forcing:
            prediction = self.networking_head.teacher_forcing(hidden_states)
        else:
            prediction = self.networking_head(hidden_states)

        loss = None
        if not return_dict:
            output = (prediction,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            # logits=lm_logits,
            logits=prediction,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
