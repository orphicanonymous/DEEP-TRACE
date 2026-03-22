import os
import subprocess
import tempfile
from math import gcd

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import timm
from scipy.signal import resample_poly
from transformers import Wav2Vec2Model
from facenet_pytorch import MTCNN

# ── FFmpeg path (hardcoded for Windows — change if yours differs) ─────────────
FFMPEG_PATH = r"C:\Users\HP\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

# ── Config (must match exactly what was used during training) ─────────────────
CFG = {
    'img_size':          299,
    'frames_per_video':  20,        # increased from 10 → better coverage
    'audio_sr':          16000,
    'audio_clip_sec':    3,
    'embed_dim':         512,
    'n_heads':           8,
    'n_layers':          2,
    'dropout':           0.1,

    # Audio reliability gate:
    # If the audio score is within this many points of 50 (pure chance),
    # we treat the audio branch as unreliable and exclude it from scoring.
    'audio_uncertainty_band': 15,
}

# ── Image transform (validation — no augmentation) ───────────────────────────
val_tfm = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — XceptionNet video detector
# ─────────────────────────────────────────────────────────────────────────────
class XceptionDetector(nn.Module):
    def __init__(self, freeze_backbone=False, embed_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            'xception', pretrained=False, num_classes=0, global_pool='avg')
        backbone_feat = self.backbone.num_features

        self.embed_proj = nn.Sequential(
            nn.Linear(backbone_feat, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def get_embedding(self, x):
        return self.embed_proj(self.backbone(x))

    def forward(self, x):
        emb    = self.get_embedding(x)
        logits = self.classifier(emb)
        return logits, emb


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — wav2vec audio detector
# ─────────────────────────────────────────────────────────────────────────────
class Wav2VecDetector(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(self._resolve_model_source())
        self.wav2vec.feature_extractor._freeze_parameters()
        w2v_hidden = self.wav2vec.config.hidden_size

        self.embed_proj = nn.Sequential(
            nn.Linear(w2v_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def get_embedding(self, waveform):
        out    = self.wav2vec(input_values=waveform)
        hidden = out.last_hidden_state.mean(dim=1)
        return self.embed_proj(hidden)

    def forward(self, waveform):
        emb    = self.get_embedding(waveform)
        logits = self.classifier(emb)
        return logits, emb

    @staticmethod
    def _resolve_model_source():
        cache_root = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))
        snapshot_root = cache_root / 'hub' / 'models--facebook--wav2vec2-base' / 'snapshots'
        if snapshot_root.exists():
            snapshots = sorted(
                [path for path in snapshot_root.iterdir() if (path / 'config.json').exists()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return str(snapshots[0])
        return 'facebook/wav2vec2-base'


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — Cross-Modal Fusion Transformer
# ─────────────────────────────────────────────────────────────────────────────
class CrossModalTransformer(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim    = embed_dim
        self.cls_token    = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.vid_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.aud_type_emb = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.vid_to_aud_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.aud_to_vid_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, vid_emb, aud_emb):
        B = vid_emb.shape[0]
        v = vid_emb.unsqueeze(1) + self.vid_type_emb
        a = aud_emb.unsqueeze(1) + self.aud_type_emb

        v_attended, _ = self.vid_to_aud_attn(v, a, a)
        a_attended, _ = self.aud_to_vid_attn(a, v, v)

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, v_attended, a_attended], dim=1)
        fused  = self.transformer(tokens)

        cls_out    = fused[:, 0, :]
        logits     = self.head(cls_out)
        scaled     = logits / self.temperature.clamp(min=0.1)
        confidence = torch.softmax(scaled, dim=1)[:, 1] * 100
        return logits, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Helper — extract audio from video using FFmpeg
# ─────────────────────────────────────────────────────────────────────────────
def extract_audio_from_video(video_path: str, target_sr: int) -> str:
    """
    Uses FFmpeg to extract the audio track from a video file into a
    temporary WAV file. Returns the path to the temp WAV file.
    Caller is responsible for deleting the temp file after use.
    """
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Prefer the configured path, but fall back safely if it's inaccessible.
    ffmpeg_exe = _get_ffmpeg_executable()

    subprocess.run([
        ffmpeg_exe,
        '-y',                       # overwrite output without asking
        '-i', video_path,           # input file
        '-ac', '1',                 # mono
        '-ar', str(target_sr),      # resample to target sample rate
        '-vn',                      # no video
        tmp_path                    # output WAV
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return tmp_path


def _get_ffmpeg_executable() -> str:
    """
    Return a usable FFmpeg executable path.
    Falls back to `ffmpeg` if the configured path is missing or inaccessible.
    """
    try:
        if Path(FFMPEG_PATH).exists():
            return FFMPEG_PATH
    except (OSError, PermissionError):
        pass
    return 'ffmpeg'


def _waveform_from_audio_data(audio_data: np.ndarray) -> torch.Tensor:
    """
    Convert SoundFile output into torch waveform shaped as [channels, samples].
    """
    arr = np.asarray(audio_data, dtype=np.float32)
    if arr.ndim == 1:
        return torch.from_numpy(arr).unsqueeze(0)
    if arr.ndim == 2:
        # SoundFile returns [samples, channels] for multi-channel audio.
        return torch.from_numpy(arr.T)
    raise ValueError(f'Unsupported audio array shape: {arr.shape}')


def _resample_waveform(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    """
    Resample waveform [channels, samples] without torchaudio/TorchCodec dependency.
    """
    src_sr = int(src_sr)
    dst_sr = int(dst_sr)
    if src_sr == dst_sr:
        return waveform

    factor = gcd(src_sr, dst_sr)
    up = dst_sr // factor
    down = src_sr // factor

    channels = waveform.detach().cpu().numpy()
    resampled = [resample_poly(ch, up, down).astype(np.float32, copy=False) for ch in channels]
    return torch.from_numpy(np.stack(resampled, axis=0))


# ─────────────────────────────────────────────────────────────────────────────
# Main inference class
# ─────────────────────────────────────────────────────────────────────────────
class DeepTraceInference:
    def __init__(self, models_dir: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models_dir  = Path(models_dir)

        print(f'Loading models on {self.device}...')

        # Load XceptionNet
        self.xception = XceptionDetector(
            freeze_backbone=False, embed_dim=CFG['embed_dim']).to(self.device)
        self.xception.load_state_dict(
            torch.load(models_dir / 'xception_best.pt',
                       map_location=self.device,
                       weights_only=False)['model_state'])
        self.xception.eval()
        print('  XceptionNet loaded')

        # Load wav2vec
        self.wav2vec = Wav2VecDetector(embed_dim=CFG['embed_dim']).to(self.device)
        self.wav2vec.load_state_dict(
            torch.load(models_dir / 'wav2vec_best.pt',
                       map_location=self.device,
                       weights_only=False)['model_state'])
        self.wav2vec.eval()
        print('  wav2vec loaded')

        # Load Fusion Transformer
        self.fusion = CrossModalTransformer(
            embed_dim=CFG['embed_dim'],
            n_heads=CFG['n_heads'],
            n_layers=CFG['n_layers'],
            dropout=CFG['dropout']
        ).to(self.device)
        self.fusion.load_state_dict(
            torch.load(models_dir / 'fusion_best.pt',
                       map_location=self.device,
                       weights_only=False)['model_state'])
        self.fusion.eval()
        print('  Fusion Transformer loaded')

        self.mtcnn   = MTCNN(image_size=299, margin=40, device=self.device)
        self.img_tfm = val_tfm
        print('All models ready!')

        # Confirm FFmpeg is reachable at startup
        ffmpeg_exe = _get_ffmpeg_executable()
        try:
            subprocess.run([ffmpeg_exe, '-version'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=True)
            print(f'  FFmpeg found: {ffmpeg_exe}')
        except Exception:
            print(f'  WARNING: FFmpeg not found at "{ffmpeg_exe}". '
                  'Audio extraction from video will fail. '
                  'Update FFMPEG_PATH at the top of inference.py.')

    def _audio_is_reliable(self, aud_score: float) -> bool:
        """
        Returns False if the audio score is too close to 50% (random chance).
        This happens when the audio model was trained on synthetic data and
        cannot distinguish real vs fake speech.
        """
        return abs(aud_score - 50.0) > CFG['audio_uncertainty_band']

    @torch.no_grad()
    def predict(self, file_path: str) -> dict:
        file_path = str(file_path)
        is_video  = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        is_audio  = file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))

        vid_score, aud_score, fusion_score = None, None, None
        vid_emb_t, aud_emb_t              = None, None
        audio_failed                       = False
        audio_reliable                     = False

        # ── Video branch ──────────────────────────────────────────────────
        if is_video:
            cap          = cv2.VideoCapture(file_path)
            total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_scores = []
            vid_embs     = []

            print(f'[DeepTrace] Video: {total} total frames, sampling {CFG["frames_per_video"]}')

            if total <= 0:
                print('[DeepTrace] WARNING: Invalid frame count reported by video file.')
                sample_indices = []
            else:
                sample_indices = np.linspace(0, total - 1, CFG['frames_per_video'], dtype=int)

            for fi in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(rgb)
                face = self.mtcnn(pil)

                if face is None:
                    # No face detected — use full resized frame
                    face_img = pil.resize((299, 299))
                else:
                    # MTCNN returns float tensor in [0, 1]; convert to valid 8-bit image.
                    face_uint8 = (
                        face.detach()
                        .permute(1, 2, 0)
                        .mul(255.0)
                        .clamp(0, 255)
                        .byte()
                        .cpu()
                        .numpy()
                    )
                    face_img = Image.fromarray(
                        face_uint8)

                tensor      = self.img_tfm(face_img).unsqueeze(0).to(self.device)
                logits, emb = self.xception(tensor)
                prob_fake   = torch.softmax(logits, dim=1)[0, 1].item()
                frame_scores.append(prob_fake)
                vid_embs.append(emb)

            cap.release()
            if frame_scores:
                vid_score = np.mean(frame_scores) * 100
                vid_emb_t = torch.stack(vid_embs).mean(0)
                print(f'[DeepTrace] Video score: {vid_score:.1f}% '
                      f'(from {len(frame_scores)} frames, '
                      f'min={min(frame_scores)*100:.1f}%, max={max(frame_scores)*100:.1f}%)')
            else:
                print('[DeepTrace] WARNING: No frames could be read from video.')

        # ── Audio branch ──────────────────────────────────────────────────
        if is_video or is_audio:
            tmp_path = None
            try:
                if is_video:
                    # Extract audio track from video via FFmpeg
                    print('[DeepTrace] Extracting audio from video via FFmpeg...')
                    tmp_path = extract_audio_from_video(file_path, CFG['audio_sr'])
                    audio_data, sr = sf.read(tmp_path, dtype='float32')
                    waveform = _waveform_from_audio_data(audio_data)
                else:
                    # Avoid torchaudio.load() to prevent TorchCodec dependency issues.
                    try:
                        audio_data, sr = sf.read(file_path, dtype='float32')
                        waveform = _waveform_from_audio_data(audio_data)
                    except Exception:
                        print('[DeepTrace] Direct audio decode failed; retrying via FFmpeg...')
                        tmp_path = extract_audio_from_video(file_path, CFG['audio_sr'])
                        audio_data, sr = sf.read(tmp_path, dtype='float32')
                        waveform = _waveform_from_audio_data(audio_data)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)
                if sr != CFG['audio_sr']:
                    waveform = _resample_waveform(waveform, sr, CFG['audio_sr'])

                clip_len = CFG['audio_sr'] * CFG['audio_clip_sec']
                waveform = waveform.squeeze(0)[:clip_len]
                if waveform.shape[0] < clip_len:
                    waveform = F.pad(waveform, (0, clip_len - waveform.shape[0]))

                waveform  = waveform / (waveform.abs().max() + 1e-8)
                waveform  = waveform.unsqueeze(0).to(self.device)
                logits_a, aud_emb_t = self.wav2vec(waveform)
                aud_score = torch.softmax(logits_a, dim=1)[0, 1].item() * 100

                audio_reliable = self._audio_is_reliable(aud_score)
                print(f'[DeepTrace] Audio score: {aud_score:.1f}% '
                      f'(reliable={audio_reliable}, '
                      f'distance from 50: {abs(aud_score - 50):.1f})')

                if not audio_reliable:
                    print('[DeepTrace] Audio score too close to 50% — '
                          'excluded from final score. '
                          'Retrain wav2vec on real speech data to fix this.')

            except Exception as e:
                print(f'[DeepTrace] Audio processing FAILED: {e}')
                audio_failed   = True
                aud_emb_t      = torch.zeros(1, CFG['embed_dim']).to(self.device)
                aud_score      = None
                audio_reliable = False

            finally:
                # Always clean up the temp WAV file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        # ── Fusion branch ─────────────────────────────────────────────────
        # Only run fusion if audio is reliable — garbage audio embeddings
        # will corrupt the fusion score too.
        if is_video and vid_emb_t is not None and aud_emb_t is not None and audio_reliable:
            _, fusion_conf = self.fusion(vid_emb_t, aud_emb_t)
            fusion_score   = fusion_conf.item()
            print(f'[DeepTrace] Fusion score: {fusion_score:.1f}%')
        elif is_video and not audio_reliable:
            print('[DeepTrace] Fusion skipped — audio branch unreliable.')

        # ── Aggregate final score ─────────────────────────────────────────
        active_scores  = []
        active_weights = []

        if vid_score is not None and audio_reliable and fusion_score is not None:
            # Best case: all three branches working
            active_scores  = [vid_score, aud_score, fusion_score]
            active_weights = [0.3, 0.3, 0.4]
            print('[DeepTrace] Scoring mode: video + audio + fusion')

        elif vid_score is not None and audio_reliable:
            # Video + reliable audio, no fusion
            active_scores  = [vid_score, aud_score]
            active_weights = [0.5, 0.5]
            print('[DeepTrace] Scoring mode: video + audio (no fusion)')

        elif vid_score is not None:
            # Video only — audio unreliable or failed
            active_scores  = [vid_score]
            active_weights = [1.0]
            print('[DeepTrace] Scoring mode: video only (audio excluded)')

        elif aud_score is not None and audio_reliable:
            # Audio-only file, reliable
            active_scores  = [aud_score]
            active_weights = [1.0]
            print('[DeepTrace] Scoring mode: audio only')

        elif aud_score is not None:
            # Audio-only file, unreliable
            active_scores  = [aud_score]
            active_weights = [1.0]
            print('[DeepTrace] Scoring mode: audio only (UNRELIABLE — retrain needed)')

        else:
            print('[DeepTrace] ERROR: No valid scores produced.')
            return {
                'verdict':          'UNKNOWN',
                'confidence_score': 50.0,
                'confidence_label': 'Analysis failed — could not read file',
                'video_score':      None,
                'audio_score':      None,
                'fusion_score':     None,
            }

        final_score = sum(s * w for s, w in zip(active_scores, active_weights))
        print(f'[DeepTrace] Final score: {final_score:.1f}%')

        verdict  = 'FAKE' if final_score > 50 else 'REAL'
        distance = abs(final_score - 50)

        if distance > 30:
            confidence_label = 'High confidence'
        elif distance > 15:
            confidence_label = 'Medium confidence'
        else:
            confidence_label = 'Low confidence — manual review recommended'

        # Append a note if audio was excluded
        if is_video and not audio_reliable and not audio_failed:
            confidence_label += ' (audio model needs retraining)'
        elif is_video and audio_failed:
            confidence_label += ' (audio extraction failed)'

        return {
            'verdict':          verdict,
            'confidence_score': round(final_score, 1),
            'confidence_label': confidence_label,
            'video_score':      round(vid_score,    1) if vid_score    is not None else None,
            # Only show audio score if reliable or audio-only file
            'audio_score':      round(aud_score,    1) if (aud_score is not None and (audio_reliable or is_audio)) else None,
            'fusion_score':     round(fusion_score, 1) if fusion_score is not None else None,
        }
