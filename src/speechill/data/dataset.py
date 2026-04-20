"""
Dataset module for data loading.

This module provides data loading utilities for the turn-taking model.
"""
from typing import Dict, Any, Optional

import json
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import librosa
import soundfile as sf
from omegaconf import DictConfig
from .processor import extract_filterbank


class TurnTakingDataset(Dataset):
    """Dataset for turn-taking model training."""

    def __init__(self, data_list: str):
        """
        Initialize dataset.

        Args:
            data_list: Path to manifest JSON file
        """
        self.data_list = data_list

        with open(data_list, 'r', encoding='utf-8') as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        waveform, sample_rate = sf.read(sample['audio_filepath'])
        if waveform.ndim != 1:
            waveform = waveform.mean(axis = 0)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        if waveform.shape[-1] < 2400:
            pad_amount = 2400 - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        
        feat = extract_filterbank(waveform, sample_rate)
        return {
            "feat": feat,
            "task": sample['task'],
            "text": sample['text']
        }