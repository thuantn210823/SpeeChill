"""
Dataset module for data loading.

This module provides data loading utilities for the turn-taking model.
"""

import json
import random
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
import librosa
import torch.nn.functional as F
from omegaconf import DictConfig


class TurnTakingDataset(Dataset):
    """Dataset for turn-taking model training."""

    def __init__(self, data_list: str, data_cfg: DictConfig,
                 tokenizer=None, prompt_cfg=None):
        """
        Initialize dataset.

        Args:
            data_list: Path to manifest JSON file
            data_cfg: Dataset configuration with processing parameters
            tokenizer: Tokenizer for text processing
            prompt_cfg: Prompt configuration for training
        """
        self.data_list = data_list
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer
        self.prompt_cfg = prompt_cfg

        with open(data_list, 'r', encoding='utf-8') as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(sample['audio_filepath'])
        waveform = waveform.squeeze(0)

        # Extract log mel spectrogram
        n_fft = self.data_cfg.get('n_fft', 400)
        hop_length = self.data_cfg.get('hop_length', 160)
        num_mel_bins = self.data_cfg.get('num_mel_bins', 80)

        window = torch.hann_window(n_fft)
        stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs()**2

        filters = torch.from_numpy(
            librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mel_bins)
        )
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        feat = log_spec.transpose(0, 1)

        # Tokenize text
        txt = sample['txt']
        if self.tokenizer:
            tokens, label = self.tokenizer.tokenize(txt)
            label = label + [self.tokenizer.eod_id]
        else:
            tokens = []
            label = []

        # Prepare prompt if configured
        prompt = None
        if self.prompt_cfg and 'task' in sample:
            task_name = sample['task']
            if task_name in self.prompt_cfg:
                prompt_list = self.prompt_cfg[task_name]
                prompt_text = random.choice(prompt_list) if self.prompt_cfg.get('random_selection', True) else prompt_list[0]
                prompt_tokens = self.tokenizer.tokenize(prompt_text)
                prompt = prompt_tokens[0]  # Get token IDs

        result = {
            'key': sample.get('audio_filepath', f'sample_{idx}'),
            'wav': waveform.unsqueeze(0),
            'sample_rate': sample_rate,
            'feat': feat,
            'txt': txt,
            'tokens': tokens,
            'label': label,
        }

        if prompt is not None:
            result['prompt'] = prompt
            result['task'] = sample.get('task', '')

        return result