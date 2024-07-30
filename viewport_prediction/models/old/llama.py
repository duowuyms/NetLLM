import torch
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union


class LlamaTaskHeadModel2(LlamaForCausalLM):
    """
    Llama with task head for viewport prediction.
    This class is implemented based on LlamaForCausalLM.

    Note: Task head is the networking head in our paper. It is the early name of our networking head.
    """
    _tied_weights_keys = ["task_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.task_head = None

    def get_task_head(self):  # task head is the networking head
        return self.task_head

    def set_task_head(self, task_head):
        self.task_head = task_head

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_len: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher_forcing: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Reimplement forward method for viewport prediction.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # if self.pretraining_tp > 1:
        #     lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
        #     logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
        #     logits = torch.cat(logits, dim=-1)
        # else:
        #     logits = self.lm_head(hidden_states)

        # logits = logits.float()

        if teacher_forcing:
            prediction = self.task_head.teacher_forcing(hidden_states, input_ids_len)
        else:
            prediction = self.task_head(hidden_states, input_ids_len)
        # prediction = self.task_head(hidden_states, input_ids_len)

        loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            # output = (logits,) + outputs[1:]
            output = (prediction,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=logits,
            logits=prediction,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
