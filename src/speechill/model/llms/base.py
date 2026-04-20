from typing import Optional, List, Dict
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import TaskType, LoraConfig as PeftLoraConfig, get_peft_model

class BaseLLM(nn.Module):
    """Base class for LLM modules."""

    def __init__(self,
             model_path: str,
             trust_remote_code: bool = True,
             torch_dtype = torch.bfloat16,       # fix 4
             max_length: int = 100,
             do_sample: bool = False,
             temperature: float = 1.0,
             repetition_penalty: float = 1.05,
             use_lora: bool = False,
             lora_rank: int = 8,
             lora_alpha: int = 32,
             lora_dropout: float = 0.1,
             lora_target_modules: Optional[List[str]] = None):  # fix 2
        super().__init__()
        if lora_target_modules is None:                              # fix 2
            lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj']

        self.llm_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=self.llm_config,          # fix 1: added comma
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(  # fix 3
            model_path, trust_remote_code=trust_remote_code
        )
        if use_lora:
            lora_config = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,  # fix 5
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        self.max_length = max_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return {'loss': outputs.loss, 'logits': outputs.logits}

    def generate(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> torch.Tensor:
        max_new_tokens = max_new_tokens or self.max_length

        outputs = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            use_cache=True,
            **generation_kwargs
        )

        return outputs

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(
            token_ids,
            add_special_tokens=False,
            skip_special_tokens=True
        )

    @property
    def embed_tokens(self):
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens
        return self.model.embed_tokens

    @property
    def lm_head(self):
        if hasattr(self.model, 'model'):
            return self.model.model.lm_head
        return self.model.lm_head
