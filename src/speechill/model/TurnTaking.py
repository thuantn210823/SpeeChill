"""
Turn-Taking Model: Composed Architecture

Main model that integrates Prompt, Adapter, Encoder, and LLM modules
through configuration-based initialization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

from src.speechill.data.processor import lengths_to_mask

class MyTurnTakingModel(nn.Module):
    def __init__(self,
                 encoder,
                 adapter,
                 llm,
                 prompt,
                 ignore_id: int = -100):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.adapter = adapter
        self.prompt = prompt
        self.ignore_id = ignore_id

        self.speech_token_embed = nn.Embedding(2, self.llm.model.config.hidden_size)

    def get_embedding_from_wav(
        self,
        wavs: torch.Tensor,
        wavs_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process audio through encoder and adapter."""
        encoder_out, encoder_lens = self.encoder(wavs, wavs_len) ## Check again because Enformer output vaid lengths not masks
        encoder_mask = lengths_to_mask(encoder_lens)
        adapter_out, adapter_mask = self.adapter(encoder_out, encoder_mask)
        return adapter_out, adapter_mask

    def add_speech_tokens(
        self,
        embeds: torch.Tensor,
        masks: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Add speech BOS/EOS tokens to embeddings."""
        B = len(embeds)
        bos_embed = self.speech_token_embed(
            torch.full((B, 1), 0, device=embeds.device)
        )
        eos_embed = self.speech_token_embed(
            torch.full((B, 1), 1, device=embeds.device)
        )
        bos_eos_target = torch.full((B, 1), self.ignore_id, device = embeds.device)
        bos_eos_mask = torch.full((B, 1), True, device=embeds.device)

        embeds = torch.cat([bos_embed, embeds, eos_embed], dim=1)
        masks = torch.cat([bos_eos_mask, masks.squeeze(1), bos_eos_mask], dim=1)

        if target is not None:
            # ignore_target = torch.full((B, 2), self.ignore_id, device=embeds.device)
            target = torch.cat([bos_eos_target, target.squeeze(1), bos_eos_target], dim=1)

        return embeds, masks, target

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        wavs = batch['feats'].to(device)
        wavs_len = batch['feat_lengths'].to(device)
        labels = batch['target']

        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        speech_target = torch.full(speech_masks.shape, self.ignore_id, device=speech_embeds.device)

        speech_embeds, speech_masks, speech_target = self.add_speech_tokens(
            speech_embeds, speech_masks, speech_target
        )

        prompt_list = [self.prompt.get_prompt(prompt) for prompt in batch['task']]
        prompt_out = self.prompt.embed_prompt(prompt_list, self.llm.tokenizer)
        prompt_embeds = self.llm.model.get_input_embeddings()(prompt_out['input_ids'].to(device))
        prompt_mask = prompt_out['attention_mask'].to(device)
        prompt_target = torch.full(prompt_out['input_ids'].shape, self.ignore_id, device=device)

        _labels = [label + self.llm.tokenizer.eos_token for label in labels]
        label_outs = self.llm.tokenizer(_labels, return_tensors='pt', padding = True)
        labels_embeds = self.llm.embed_tokens(label_outs['input_ids'].to(device))
        labels_mask = label_outs['attention_mask'].to(device)

        inputs_list = []
        masks_list = []
        targets_list = []

        if prompt_embeds is not None:
            inputs_list.append(prompt_embeds)
            masks_list.append(prompt_mask)
            targets_list.append(prompt_target)

        inputs_list.extend([speech_embeds, labels_embeds])
        masks_list.extend([speech_masks, labels_mask])
        targets_list.extend([speech_target, label_outs['input_ids'].to(device)])

        inputs_embeds = torch.cat(inputs_list, dim=1).to(self.llm.model.dtype)
        attention_mask = torch.cat(masks_list, dim=1).to(self.llm.model.dtype)
        target = torch.cat(targets_list, dim=1).to(torch.long)

        # position_ids = attention_mask.long().cumsum(-1) - 1
        # position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self.llm(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=target
        )

        return {'loss': outputs['loss']}

    @torch.no_grad()
    def generate(
        self,
        wavs: torch.Tensor,
        wavs_len: torch.Tensor,
        task: str = "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>"
    ) -> List[str]:
        """Generate turn-taking prediction from audio."""
        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)

        speech_embeds, speech_masks, _ = self.add_speech_tokens(speech_embeds, speech_masks)

        prompt_text = self.prompt.get_prompt(task)
        prompt_out = self.prompt.embed_prompt(prompt_text, self.llm.tokenizer)
        prompt_embeds = self.llm.model.get_input_embeddings()(prompt_out['input_ids'].to(wavs.device))
        prompt_mask = prompt_out['attention_mask'].to(wavs.device)

        inputs_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        # attention_mask = torch.ones(
        #     inputs_embeds.size()[:-1],
        #     dtype=torch.long,
        #     device=inputs_embeds.device
        # )
        attention_mask = torch.cat([prompt_mask, speech_masks], dim = 1)

        inputs_embeds = inputs_embeds.to(self.llm.model.dtype)
        attention_mask = attention_mask.to(self.llm.model.dtype)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_token_id=self.llm.model.config.eos_token_id,
            pad_token_id=-100
        )

        return self.llm.tokenier.decode(outputs, skip_special_tokens = True)
