# DeepTrace Model Card

## Overview

DeepTrace is a multimodal deepfake detection system served through a Flask backend. It accepts either video or audio input and returns a binary verdict of `REAL` or `FAKE` together with branch-level scores and a human-readable confidence label.

This model card describes what can be verified from the current inference repository. It also explicitly calls out metadata that is not present in the repo and therefore should not be claimed without separate evidence.

## Intended Use

- Detect potential deepfakes in uploaded media.
- Support triage and human review.
- Provide a model-assisted signal, not a final forensic judgment.

## Non-Goals

- This system is not a legally authoritative authenticity detector.
- This system is not guaranteed to generalize to all deepfake generation methods.
- This system is not a replacement for human moderation, forensic review, or source verification.

## System Architecture

DeepTrace uses three model components:

1. Video detector:
   An Xception-based image classifier processes sampled video frames and produces frame-level fake probabilities and a pooled video embedding.

2. Audio detector:
   A wav2vec2-based audio classifier processes a normalized 3-second waveform and produces an audio fake probability and an audio embedding.

3. Fusion detector:
   A cross-modal transformer combines the video and audio embeddings when the audio branch is considered reliable.

## Runtime Pipeline

1. The backend receives a `POST /analyze` request with a `file_url`.
2. The file is downloaded to a temporary local path.
3. The file type is inferred from its extension.
4. If the input is video:
   - 20 frames are sampled across the clip.
   - Each frame is converted to RGB.
   - A face is detected with MTCNN.
   - If no face is detected, the full frame is resized and used instead.
   - Each processed frame is passed through the Xception branch.
   - Frame probabilities are averaged to form the video score.
5. If the input is video or audio:
   - Audio is decoded with `soundfile`.
   - For video input, audio is first extracted with FFmpeg.
   - Multi-channel audio is converted to mono.
   - Audio is resampled to 16 kHz when necessary.
   - The waveform is truncated or padded to 3 seconds.
   - The waveform is amplitude-normalized.
   - The waveform is passed through the wav2vec branch.
6. If video and reliable audio are both available:
   - Video and audio embeddings are fused by a transformer.
7. The final score is aggregated from the active branches.
8. The backend returns a JSON response with verdict and confidence fields.

## Model Components

### Video Branch

- Backbone: `timm` Xception
- Pretrained at construction: `False`
- Output classes: `2`
- Intermediate embedding dimension: `512`
- Checkpoint file: `backend/models/xception_best.pt`

### Audio Branch

- Backbone: `facebook/wav2vec2-base`
- Feature extractor frozen: `Yes`
- Intermediate embedding dimension: `512`
- Output classes: `2`
- Checkpoint file: `backend/models/wav2vec_best.pt`

### Fusion Branch

- Type: Cross-modal transformer
- Embedding dimension: `512`
- Attention heads: `8`
- Transformer layers: `2`
- Dropout: `0.1`
- Checkpoint file: `backend/models/fusion_best.pt`

## Input Specification

### Supported File Types

- Video: `.mp4`, `.avi`, `.mov`, `.mkv`
- Audio: `.wav`, `.mp3`, `.flac`, `.m4a`

### Video Preprocessing

- Samples `20` frames per video
- Face detector: MTCNN
- Fallback when no face is found: use resized full frame
- Input image size: `299 x 299`
- Normalization: mean `[0.5, 0.5, 0.5]`, std `[0.5, 0.5, 0.5]`

### Audio Preprocessing

- Target sample rate: `16000 Hz`
- Target clip length: `3 seconds`
- Multi-channel handling: average to mono
- Resampling method: `scipy.signal.resample_poly`
- Waveform normalization: divide by absolute max plus epsilon

## Output Specification

The inference API returns:

- `verdict`: `REAL`, `FAKE`, or `UNKNOWN`
- `confidence_score`: final score on a 0-100 scale
- `confidence_label`
- `video_score`
- `audio_score`
- `fusion_score`

### Verdict Logic

- If final score is greater than `50`, verdict is `FAKE`
- Otherwise verdict is `REAL`

### Confidence Label Logic

- Distance from 50 greater than `30`: `High confidence`
- Distance from 50 greater than `15`: `Medium confidence`
- Otherwise: `Low confidence - manual review recommended`

## Branch Weighting

When all branches are active:

- Video: `0.3`
- Audio: `0.3`
- Fusion: `0.4`

When only video and reliable audio are available:

- Video: `0.5`
- Audio: `0.5`

When only one usable branch is available:

- That branch gets weight `1.0`

## Audio Reliability Guard

The system contains an explicit safeguard for the audio branch:

- If the audio score is within `15` points of `50`, the audio branch is treated as unreliable.
- Unreliable audio is excluded from video-audio fusion.
- In video analysis, unreliable audio causes the system to fall back to video-only scoring.

This is important because the current code comments acknowledge that the audio model may require retraining on real speech data.

## Operational Dependencies

- Python backend with Flask
- PyTorch
- torchvision
- torchaudio
- timm
- transformers
- facenet-pytorch
- OpenCV
- soundfile
- scipy
- FFmpeg for video audio extraction and some fallback decoding paths

## Known Limitations

- The system depends heavily on correct file decoding and FFmpeg availability.
- The audio branch is explicitly treated as potentially unreliable.
- The system uses filename extension to decide whether an input is video or audio.
- The system samples only 20 frames, so localized manipulation outside those frames may be missed.
- If face detection fails, the full frame is used, which may reduce sensitivity to face-level artifacts.
- The confidence score is a heuristic model score, not a calibrated probability of truth.
- Low-confidence outputs should be manually reviewed.

## Risks And Responsible Use

- False positives may wrongly flag authentic content.
- False negatives may miss manipulated content.
- Performance may vary across languages, accents, recording conditions, compression levels, editing pipelines, and generation methods.
- The tool should not be used as the sole basis for punitive, legal, academic, or employment decisions.

## Proven Metadata Present In This Repo

The following facts are directly supported by the current codebase:

- The backend loads three checkpoints:
  - `xception_best.pt`
  - `wav2vec_best.pt`
  - `fusion_best.pt`
- The system performs binary classification.
- The system uses a multimodal pipeline with conditional fusion.
- The system excludes audio when it behaves too close to chance.

## Metadata Missing From This Repo

The following information is not currently documented in the repository and should not be claimed without separate evidence:

- Training dataset names
- Dataset sizes
- Data licenses
- Train/validation/test split policy
- Exact training code
- Hyperparameters such as optimizer, learning rate, batch size, and epoch count
- Data augmentation policy during training
- Validation and test metrics
- Calibration metrics
- Fairness or subgroup analysis
- Robustness evaluation
- Model version dates and authorship history
- Known benchmark comparisons

## Recommended Accountability Statement

Use this short description when presenting the system:

"DeepTrace is a multimodal deepfake detection pipeline that analyzes sampled video frames, short-form audio, and a fused video-audio representation to estimate whether media is likely real or fake. It is designed as a decision-support tool rather than a definitive forensic authority. The current repository documents the inference pipeline and runtime thresholds, but it does not yet contain full training and evaluation metadata needed for strong performance claims."

## Recommended Next Documentation

To make this system more accountable, add:

- A dataset card
- A training report
- Evaluation metrics by split
- Failure case examples
- A model version history
- A deployment and monitoring note
