# Speech-Language Model (SLM): ASR & Turn-Taking

A modular spoken language understanding system that supports both **Automatic Speech Recognition (ASR)** and **Turn-Taking Detection** through a unified architecture. The model automatically switches behavior based on the `task` field in the data manifest.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Task Switching Mechanism](#task-switching-mechanism)
- [Supported Tasks](#supported-tasks)
- [Components](#components)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)

## Overview

This project implements a Speech-Language Model (SLM) that combines:

1. **ASR (Automatic Speech Recognition)** - Transcribes audio to text
2. **Turn-Taking Detection** - Predicts turn-taking behavior in dialogues:
   - `<BACKCHANNEL>` - Short acknowledgment/feedback (e.g., "yeah", "uh-huh")
   - `<COMPLETE>` - Semantic completeness (speaker finished thought)
   - `<INCOMPLETE>` - Semantic incompleteness (speaker interrupted)
   - `<WAIT>` - Request to pause or end conversation

The system uses a **Encoder + Adapter + LLM** pipeline where the same model handles both tasks by switching prompts via the `task` field in `manifest.json`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Audio Waveform                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Feature Extraction (MelSpectrogram)              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                              Encoder                                 │
│                  (Conformer/Transformer Whisper)                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                              Adapter                                 │
│                   (Lyz-Conv: Linear Projection)                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Speech Token Embedding (BOS/EOS)                 │
│                         nn.Embedding(2, hidden_size)                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                             Prompt                                   │
│                  (Task-specific instruction templates)              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM (Qwen2.5-0.5B-Instruct)                      │
│                 Embedding → Causal LM → Output Tokens               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT: Text + Task Label                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Audio** → Mel-spectrogram features (80-dimensional)
2. **Encoder** → Processes features, outputs contextual representations
3. **Adapter** → Projects encoder dimensions to LLM hidden size
4. **Speech Tokens** → Adds BOS (beginning-of-speech) and EOS (end-of-speech) embeddings
5. **Prompt** → Prepends task-specific instruction (ASR or Turn-Taking)
6. **LLM** → Generates text output based on combined input

## Task Switching Mechanism

The model switches behavior based on the `task` field in `manifest.json`:

```json
{"task": "<TRANSCRIBE>", "audio_filepath": "audio.wav", "txt": "Hello world"}
{"task": "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>", "audio_filepath": "audio.wav", "txt": "Hello<BACKCHANNEL>"}
```

### How It Works

1. **manifest.json** contains a `task` field for each audio sample
2. **Dataset** passes the `task` to the model during training
3. **Prompt Module** looks up the corresponding instruction template
4. **LLM** generates output based on the task instruction

### Prompt Templates

Prompts are defined in YAML files (`configs/prompts/`):

**ASR (English):**
```yaml
<TRANSCRIBE>:
  - Transcribe the following audio.
  - Transcribe the speech content in this audio into text.
```

**Turn-Taking (English):**
```yaml
<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>:
  - Please transcribe the audio content and append one of the following four tags at the end for interruption judgment：<complete> for complete semantics, <incomplete> for incomplete semantics, <backchannel> for short acknowledgments, and <wait> for requests to pause or terminate the conversation.
  - Please transcribe the audio into text and mark the interruption type at the end：<complete> (semantically complete), <incomplete> (semantically incomplete), <backchannel> (backchanneling), and <wait> (requesting to pause or end the dialogue).
  
```

## Supported Tasks

| Task | Description | Output Example |
|------|-------------|----------------|
| `<TRANSCRIBE>` | ASR only | "yeb" |
| `<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>` | ASR + Turn-Taking | "yeb <BACKCHANNEL>" |

## Components

### 1. Dataset (`src/speechill/data/dataset.py`)

- Loads audio files (WAV, FLAC, MP3, etc.)
- Extracts Mel-spectrogram features
- Returns: `feat`, `task`, `text`

### 2. Feature Extraction (`src/speechill/data/processor.py`)

- **MelSpectrogram**: 80 mel bins, 25ms window, 10ms hop
- Kaldi-style fbank/mfcc extraction supported

### 3. Encoder (`src/speechill/model/encoders/`)

- Transformer-based encoder
- Input: Mel-spectrogram (B, T, 80)
- Output: Hidden representations (B, T, 1024)

### 4. Adapter (`src/speechill/model/adapters/`)

- **Lyz-Conv Adapter**: 1D Convolutional adapter
- Projects encoder dimensions to LLM hidden size
- Configurable kernel size, activation, normalization

### 5. LLM (`src/speechill/model/llms/base.py`)

- **Qwen2.5-0.5B-Instruct** (configurable)
- Optional LoRA fine-tuning support
- Methods: `forward()`, `generate()`, `decode()`

### 6. Prompts (`src/speechill/model/prompts/base.py`)

- Loads prompt templates from YAML
- Random selection for data augmentation
- Tokenizes prompts for LLM input

### 7. Turn-Taking Model (`src/speechill/model/TurnTaking.py`)

Main model class that orchestrates all components:

```python
class MyTurnTakingModel(nn.Module):
    def __init__(self, encoder, adapter, llm, prompt, ignore_id=-100):
        # Initialize encoder, adapter, LLM, prompt
        self.speech_token_embed = nn.Embedding(2, llm.hidden_size)

    def forward(self, batch, device):
        # 1. Extract audio features
        # 2. Add speech tokens (BOS/EOS)
        # 3. Embed prompt
        # 4. Concatenate [prompt, speech, labels]
        # 5. Forward through LLM
        # 6. Return loss
```

## Configuration

### Main Config (`configs/train/turn-taking.yaml`)

```yaml
model:
  encoder:
    _target_: src.speechill.model.encoders.create_encoder
    name: transformer_whisper
    input_dim: 80
    output_dim: 1024

  adapter:
    _target_: src.speechill.model.adapters.create_adapter
    name: lyz_conv
    encoder_dim: 1024
    llm_dim: 896

  llm:
    _target_: src.speechill.model.llms.create_llm
    name: qwen
    model_path: src/Qwen2.5-0.5B-Instruct
    use_lora: false

  prompt:
    _target_: src.speechill.model.prompts.create_prompt_module
    prompts_file: configs/prompts/prompt_cn.yaml
    random_selection: true
```

### Data Manifest (`data/manifest.json`)

```json
{"task": "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>", "audio_filepath": "samples/audio.wav", "txt": "yeb<BACKCHANNEL>", "duration": 3.5}
```

### Prompt Files (`configs/prompts/`)

- `prompt_cn.yaml` - Chinese prompts
- `prompt_en.yaml` - English prompts
- `prompt_vn.yaml` - Vietnamese prompts

## Training

### Quick Start

```bash
# Install dependencies
pip install -e .

# Train with config
python train.py --config configs/train/turn-taking.yaml
```

### Training Script (`train.py`)

The training script uses Hydra for configuration management and Lightning for training loop.

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 2 | Training batch size |
| learning_rate | 5e-5 | AdamW learning rate |
| max_epochs | 100 | Maximum training epochs |
| gradient_clip | 5 | Gradient clipping norm |
| accumulate_grad | 4 | Gradient accumulation steps |
| use_lora | false | Enable LoRA fine-tuning |

## Inference

### Supported Generation Tasks

```python
# ASR only
task = "<TRANSCRIBE>"

# ASR + Turn-Taking
task = "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>"
```

## Dependencies

- Python 3.10+
- PyTorch 2.10+
- Transformers 4.44+
- Lightning
- Hydra
- Torchaudio
- Librosa

See `requirements.txt` for full list.

## License

Apache License 2.0
