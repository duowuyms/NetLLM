import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict


TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],
    'llava': ["q_proj", "v_proj"],
    'mistral': ["q_proj", "v_proj"],
    'opt': ["q_proj", "v_proj"],
    'gpt2': ["q_proj", "v_proj"],
    't5-lm': ["q", "v"]
}


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def peft_model(plm, plm_type, rank, print_trainable=False, task_type=TaskType.FEATURE_EXTRACTION):
    for param in plm.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    plm.gradient_checkpointing_enable()
    plm.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    config = LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=TARGET_MODULES[plm_type],
        lora_dropout=0.05,
        bias="none",
        task_type=task_type
    )

    model = get_peft_model(plm, config)
    model.from_pretrained
    if print_trainable:
        print_trainable_parameters(model)
    return model
