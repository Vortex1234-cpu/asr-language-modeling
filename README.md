# Speech-to-Text System with HMM and Language Model Integration

## Overview

This project implements a speech-to-text transcription system using Hidden Markov Models (HMM) combined with an n-gram language model to improve transcription accuracy. The system extracts MFCC features from audio, trains an HMM for phoneme or character recognition, and uses a trigram language model to refine output predictions.

## Features

- MFCC feature extraction using Librosa  
- Hidden Markov Model training with `hmmlearn`  
- Trigram language model integration using NLTK  
- Evaluation with Word Error Rate (WER) and Character Error Rate (CER) using `jiwer`  
- Transcription of new audio files

## Installation

1. Clone the repository or download the files.  
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download the necessary NLTK data:

```python
import nltk
nltk.download('punkt')
```

## Usage

1. Train the model on your dataset (follow notebook steps or scripts).  
2. Save the trained HMM model and language model counts.  
3. Transcribe audio files:

```python
transcript = transcribe_audio('path_to_audio.wav')
print(transcript)
```

4. Optionally, evaluate transcription accuracy if ground truth is available:

```python
from jiwer import wer, cer

print("WER:", wer(true_text, transcript))
print("CER:", cer(true_text, transcript))
```

## Files

- `hmm_model.pkl` — Saved HMM model  
- `char_encoder.pkl` — Character encoder for decoding states  
- `context_counts.pkl` — Language model trigram counts  
- `requirements.txt` — Python dependencies  
- `README.md` — Project documentation

## References

- [Librosa](https://librosa.org/) for audio processing  
- [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/) for HMM training  
- [NLTK](https://www.nltk.org/) for NLP utilities  
- [JIWER](https://github.com/jitsi/jiwer) for transcription evaluation  
