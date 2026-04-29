from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import MimiModel, AutoFeatureExtractor
from transformers import WhisperFeatureExtractor
from modeling_whisper import WhisperVQEncoder
import torchaudio

class SpeechTokenizer(nn.Module):
    """
    """
    def __init__(self,
                 model_name: str,
                 cache: str):
        super().__init__()
        if model_name not in ['mimi', 'glm-4']:
            raise "Currently only supporting 'mimi' and 'glm-4'"
        self.model_name = model_name
        self._setup(cache)
    
    def _setup(self, cache: str):
        if self.model_name == 'glm-4':
            self.model = WhisperVQEncoder.from_pretrained(cache)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(cache)
            self.sr = self.feature_extractor.sampling_rate
            self.model.eval()
            self.num_embeddings = self.model.codebook.num_embeddings
            self.embedding_dim = self.model.codebook.embedding_dim
        
        elif self.model_name == 'mimi':
            self.model = MimiModel.from_pretrained(cache)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(cache)
            self.sr = self.feature_extractor.sampling_rate
            self.model.eval()
    
    def encode(self, inputs: np.array, sr):
        inputs = torch.from_numpy(inputs.copy()).float()
        inputs = torchaudio.functional.resample(inputs, sr, self.sr).numpy()

        if self.model_name == 'glm-4':
            inputs = self.feature_extractor(
                raw_speech = inputs, 
                sampling_rate = self.sr,
                return_tensors = "pt",
                return_attention_mask = True
            ).to(self.model.device)
        elif self.model_name == 'mimi':
            inputs = self.feature_extractor(
                raw_audio = inputs,
                sampling_rate = self.sr,
                return_tensors = 'pt'
            ).to(self.model.device)
            with torch.no_grad():
                quantized_token_ids = self.model.encode(inputs['input_values']).audio_codes
                last_hidden_state = self.quantizer.decode(quantized_token_ids)
                outputs = {'last_hidden_state': last_hidden_state,
                           'quantized_token_ids': quantized_token_ids}
        return outputs