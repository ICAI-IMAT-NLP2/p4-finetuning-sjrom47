import math

import torch
import torch.nn as nn

try:
    from utils import download_and_load_model
except:
    from src.utils import download_and_load_model


class LoRA(nn.Module):
    def __init__(self, original_layer, r=4, alpha=32):
        """
        Low-Rank Adaptation (LoRA) module.

        Args:
            original_layer (nn.Module): The original layer to which LoRA is applied.
            r (int): Rank of the low-rank approximation.
            alpha (int): Scaling factor for the LoRA module.
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.original_layer = original_layer

        out_feat, in_feat = original_layer.weight.shape
        self.A = torch.nn.Parameter(torch.empty(in_feat, r))
        self.B = torch.nn.Parameter(torch.empty(r, out_feat))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.scaling = alpha / r

        for param in original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original_layer(x) + self.scaling * (x @ self.A @ self.B)


def inject_lora_into_model(model, r=4, alpha=32, device="cpu"):
    """
    Inject LoRA layers into the linear layers of the attention modules of the model.

    Args:
        model (PreTrainedModel): The pre-trained model.
        r (int): Rank of the low-rank approximation.
        alpha (int): Scaling factor for LoRA.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').

    Returns:
        model (PreTrainedModel): The model with LoRA injected into attention layers.
    """
    for child_name, child_module in model.named_children():
        if isinstance(child_module, nn.Linear) and any(
            key == child_name for key in ["q", "k", "v", "o"]
        ):
            lora_layer = LoRA(child_module, r, alpha)
            setattr(model, child_name, lora_layer)
        else:
            inject_lora_into_model(child_module, r, alpha, device)
    return model.to(device)


class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length, model_hidden_size):
        """
        Creates trainable soft prompts to prepend to input embeddings.

        Args:
            prompt_length (int): Number of virtual tokens in the soft prompt.
            model_hidden_size (int): The hidden size of the pre-trained model.
        """
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, model_hidden_size))

    def forward(self, input_embeddings):
        """
        Forward pass to prepend soft prompts to input embeddings.

        Args:
            input_embeddings (torch.Tensor): The original input embeddings from the tokenizer.

        Returns:
            torch.Tensor: The concatenated soft prompts and original embeddings.
        """

        batch_size = input_embeddings.shape[0]
        soft_prompt_expanded = self.soft_prompt.expand(
            batch_size, *self.soft_prompt.shape
        )

        return torch.cat([soft_prompt_expanded, input_embeddings], dim=1)
