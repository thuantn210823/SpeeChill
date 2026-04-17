"""
Turn-Taking Model: Composed Architecture

Main model that integrates Prompt, Adapter, Encoder, and LLM modules
through configuration-based initialization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List

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

        self.speech_token_embed = nn.Embedding(2, self.llm.hidden_size)

    def get_embedding_from_wav(
        self,
        wavs: torch.Tensor,
        wavs_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process audio through encoder and adapter."""
        encoder_out, encoder_mask = self.encoder(wavs, wavs_len) ## Check again because Enformer output vaid lengths not masks
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
        bos_eos_target = torch.full((B, 1), self.ignore_id, deivce = embeds.device)
        bos_eos_mask = torch.full((B, 1), True, device=embeds.device)

        embeds = torch.cat([bos_embed, embeds, eos_embed], dim=1)
        masks = torch.cat([bos_eos_mask, masks, bos_eos_mask], dim=1)

        if target is not None:
            # ignore_target = torch.full((B, 2), self.ignore_id, device=embeds.device)
            target = torch.cat([bos_eos_target, target, bos_eos_target], dim=1)

        return embeds, masks, target

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        wavs = batch['feats'].to(device)
        wavs_len = batch['feats_lengths'].to(device)
        labels = batch['target'].to(device)

        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        speech_target = torch.full(speech_masks.shape, self.ignore_id, device=speech_embeds.device)

        speech_embeds, speech_masks, speech_target = self.add_speech_tokens(
            speech_embeds, speech_masks, speech_target
        )

        if 'prompt' in batch:
            prompt = batch['prompt'].to(device)
            prompt_embeds = self.llm.embed_tokens(prompt)
            prompt_target = torch.full(prompt.shape, self.ignore_id, device=device)
            prompt_mask = torch.ones_like(prompt, dtype=torch.bool, device=device)
        else:
            prompt_embeds = None
            prompt_target = None
            prompt_mask = None

        labels_embeds = self.llm.embed_tokens(labels)
        labels_mask = torch.ones_like(labels, dtype=torch.bool, device=labels.device)

        inputs_list = []
        masks_list = []
        targets_list = []

        if prompt_embeds is not None:
            inputs_list.append(prompt_embeds)
            masks_list.append(prompt_mask)
            targets_list.append(prompt_target)

        inputs_list.extend([speech_embeds, labels_embeds])
        masks_list.extend([speech_masks, labels_mask])
        targets_list.extend([speech_target, labels])

        inputs_embeds = torch.cat(inputs_list, dim=1)
        attention_mask = torch.cat(masks_list, dim=1)
        target = torch.cat(targets_list, dim=1)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

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
        prompt_ids = self.prompt.embed_prompt(prompt_text, self.llm.tokenizer)
        prompt_embeds = self.llm.embed_tokens(prompt_ids.to(speech_embeds.device))

        inputs_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.size()[:-1],
            dtype=torch.long,
            device=inputs_embeds.device
        )

        if inputs_embeds.dtype == torch.float16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_token_id=151643,
            pad_token_id=-100
        )

        return self.llm.decode(outputs)
