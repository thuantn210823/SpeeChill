"""
Turn-Taking Model: Composed Architecture

Main model that integrates Prompt, Adapter, Encoder, and LLM modules
through configuration-based initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from src.speechill.data.processor import lengths_to_mask, extract_filterbank

class MyTurnTakingModel(nn.Module):
    def __init__(self,
                 encoder,
                 adapter,
                 llm,
                 speech_tokenizer,
                 prompt,
                 ignore_id: int = -100):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.adapter = adapter
        self.speech_tokenizer = speech_tokenizer
        self.linear = nn.Linear(speech_tokenizer.embedding_dim, adapter.llm_dim)
        self.prompt = prompt
        self.ignore_id = ignore_id

        self.speech_token_embed = nn.Embedding(2, self.llm.model.config.hidden_size)

    def get_features(self, wavs, srs):
        feats = []
        stokens_list = []
        feat_lens = []
        for wav, sr in zip(wavs, srs):
            feat = extract_filterbank(wav, sr, device = wav.device)
            stoken = self.speech_tokenizer.encode(wav.cpu().numpy().squeeze(), sr)['last_hidden_state']
            feats.append(feat.transpose(0, 1))
            stokens_list.append(stoken.squeeze(0))
            feat_lens.append(feat.transpose(0, 1).shape[-1])
        feats = pad_sequence(feats, batch_first = True)
        # stokens = torch.stack(stokens_list, dim=0)
        stokens = pad_sequence(stokens_list, batch_first = True)
        feat_lens = torch.tensor(feat_lens, dtype = torch.long, device = wav.device)
        return feats, feat_lens, stokens

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
        wavs = batch['waveforms'].to(device)
        srs = batch['srs']

        wavs, wavs_len, stokens = self.get_features(wavs, srs)
        labels = batch['target']

        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        stokens = self.linear(stokens)

        # Get dimensions
        B, S, D = speech_embeds.shape
        B, T, D = stokens.shape
        len_ = S+T

        # Pad to same length
        max_len = max(S, T)
        if S < max_len:
            speech_embeds = F.pad(speech_embeds, (0, 0, 0, max_len - S))
            speech_masks = F.pad(speech_masks, (0, max_len - speech_masks.shape[-1]))
        if T < max_len:
            stokens = F.pad(stokens, (0, 0, 0, max_len - T))

        # Interleave embeddings: [s[0], t[0], s[1], t[1], ...]
        combined = torch.stack([speech_embeds, stokens], dim=2)  # (B, max_len, 2, D)
        new_speech_embeds = combined.view(B, max_len * 2, D)  # (B, max_len*2, D)
        new_speech_embeds = new_speech_embeds[:, :len_, :]

        # Interleave masks (duplicate each mask value)
        mask_for_interleave = speech_masks.squeeze(1) if speech_masks.dim() == 3 else speech_masks
        new_speech_masks = mask_for_interleave.repeat_interleave(2, dim=1)  # (B, max_len*2)
        new_speech_masks = new_speech_masks.unsqueeze(1)  # (B, 1, max_len*2)
        new_speech_masks = new_speech_masks[:, :, :len_]

        # Update variables
        speech_embeds = new_speech_embeds
        speech_masks = new_speech_masks

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
        srs: List[int],
        task: str = "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>"
    ) -> List[str]:
        """Generate turn-taking prediction from audio."""
        # Get features and speech tokenizer tokens
        feats, feat_lens, stokens = self.get_features(wavs, srs)
        speech_embeds, speech_masks = self.get_embedding_from_wav(feats, feat_lens)
        stokens = self.linear(stokens)

        # Interleave embeddings
        B, S, D = speech_embeds.shape
        B, T, D = stokens.shape
        max_len = max(S, T)
        
        if S < max_len:
            speech_embeds = F.pad(speech_embeds, (0,0,0, max_len - S))
            speech_masks = F.pad(speech_masks, (0, max_len - speech_masks.shape[-1]))
        if T < max_len:
            stokens = F.pad(stokens, (0,0,0, max_len - T))
        
        # Interleave: [s[0], t[0], s[1], t[1], ...]
        combined = torch.stack([speech_embeds, stokens], dim=2)
        speech_embeds = combined.view(B, max_len * 2, D)
        
        # Interleave masks
        mask_for_interleave = speech_masks.squeeze(1) if speech_masks.dim() == 3 else speech_masks
        speech_masks = mask_for_interleave.repeat_interleave(2, dim=1).unsqueeze(1)

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
            pad_token_id=-100,
            do_sample = False,
            max_length = self.llm.max_length
        )

        return self.llm.tokenizer.decode(outputs, skip_special_tokens = True)
