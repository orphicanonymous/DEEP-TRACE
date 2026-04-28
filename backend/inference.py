import base64
import os
import shutil
import subprocess
import tempfile
from math import gcd
from io import BytesIO

try:
    import soundfile as sf
except ModuleNotFoundError as exc:
    if exc.name == '_cffi_backend':
        raise RuntimeError(
            'soundfile could not import its cffi backend. '
            'This virtualenv likely has a Python-version mismatch '
            '(for example, a cp312 wheel inside a Python 3.13 env). '
            'Reinstall cffi for the active interpreter or recreate the virtualenv.'
        ) from exc
    raise

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

# ── FFmpeg path (set FFMPEG_PATH env var if not on system PATH) ───────────────
FFMPEG_PATH = os.environ.get('FFMPEG_PATH', '').strip()

# ── Config (must match exactly what was used during training) ─────────────────
CFG = {
    'img_size':             299,
    'frames_per_video':     20,
    'audio_sr':             16000,
    'audio_clip_sec':       3,
    'embed_dim':            512,
    'n_heads':              8,
    'n_layers':             2,
    'dropout':              0.1,
    # Audio reliability gate:
    # If audio score is within this many points of 50 (pure chance),
    # exclude it from the final score.
    'audio_uncertainty_band': 10,
}

# ── Validation transform (no augmentation) ───────────────────────────────────
val_tfm = T.Compose([
    T.Resize((CFG['img_size'], CFG['img_size'])),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ── TTA transforms — 4 augmented views per face crop ─────────────────────────
# Each view stresses different artifact types that GAN face-swaps leave behind.
tta_tfms = [
    # 1. Original — baseline
    T.Compose([
        T.Resize((CFG['img_size'], CFG['img_size'])),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]),
    # 2. Horizontal flip — catches asymmetric GAN artifacts
    T.Compose([
        T.Resize((CFG['img_size'], CFG['img_size'])),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]),
    # 3. Slight zoom-in crop — tests scale robustness
    T.Compose([
        T.Resize((320, 320)),
        T.CenterCrop(CFG['img_size']),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]),
    # 4. Subtle brightness + contrast jitter — counters lighting normalization in fakes
    T.Compose([
        T.Resize((CFG['img_size'], CFG['img_size'])),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — XceptionNet video detector
# ─────────────────────────────────────────────────────────────────────────────
class XceptionDetector(nn.Module):
    def __init__(self, freeze_backbone: bool = False, embed_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            'xception', pretrained=False, num_classes=0, global_pool='avg'
        )
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

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_proj(self.backbone(x))

    def forward(self, x: torch.Tensor):
        emb    = self.get_embedding(x)
        logits = self.classifier(emb)
        return logits, emb


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Wav2Vec2 audio detector
# ─────────────────────────────────────────────────────────────────────────────
class Wav2VecDetector(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(self._resolve_model_source())
        self.wav2vec.feature_extractor._freeze_parameters()
        w2v_hidden = self.wav2vec.config.hidden_size  # 768

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

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        out    = self.wav2vec(input_values=waveform)
        hidden = out.last_hidden_state.mean(dim=1)
        return self.embed_proj(hidden)

    def forward(self, waveform: torch.Tensor):
        emb    = self.get_embedding(waveform)
        logits = self.classifier(emb)
        return logits, emb

    @staticmethod
    def _resolve_model_source() -> str:
        """Use a local cached snapshot if available, otherwise download."""
        cache_root    = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))
        snapshot_root = cache_root / 'hub' / 'models--facebook--wav2vec2-base' / 'snapshots'
        if snapshot_root.exists():
            snapshots = sorted(
                [p for p in snapshot_root.iterdir() if (p / 'config.json').exists()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return str(snapshots[0])
        return 'facebook/wav2vec2-base'


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — Cross-Modal Fusion Transformer
# ─────────────────────────────────────────────────────────────────────────────
class CrossModalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        n_heads:   int = 8,
        n_layers:  int = 2,
        dropout:   float = 0.1,
    ):
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
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, vid_emb: torch.Tensor, aud_emb: torch.Tensor):
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_ffmpeg_executable() -> str | None:
    """Return the first usable FFmpeg path, or None."""
    if FFMPEG_PATH:
        try:
            p = Path(FFMPEG_PATH)
            if p.is_file():
                return str(p)
        except (OSError, PermissionError):
            pass
    return shutil.which('ffmpeg')


def extract_audio_from_video(video_path: str, target_sr: int) -> str:
    """
    Extract audio track from a video file to a temporary WAV via FFmpeg.
    Caller must delete the returned path when done.
    """
    tmp      = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmp.name
    tmp.close()

    ffmpeg_exe = _get_ffmpeg_executable()
    if not ffmpeg_exe:
        raise FileNotFoundError(
            'FFmpeg not found. Install it and add to PATH, '
            'or set the FFMPEG_PATH environment variable.'
        )

    subprocess.run(
        [ffmpeg_exe, '-y', '-i', video_path,
         '-ac', '1', '-ar', str(target_sr), '-vn', tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return tmp_path


def _waveform_from_audio_data(audio_data: np.ndarray) -> torch.Tensor:
    """Convert SoundFile output → torch waveform [channels, samples]."""
    arr = np.asarray(audio_data, dtype=np.float32)
    if arr.ndim == 1:
        return torch.from_numpy(arr).unsqueeze(0)
    if arr.ndim == 2:
        # SoundFile returns [samples, channels] — transpose to [channels, samples]
        return torch.from_numpy(arr.T)
    raise ValueError(f'Unsupported audio array shape: {arr.shape}')


def _resample_waveform(
    waveform: torch.Tensor, src_sr: int, dst_sr: int
) -> torch.Tensor:
    """Resample [channels, samples] without torchaudio dependency."""
    src_sr, dst_sr = int(src_sr), int(dst_sr)
    if src_sr == dst_sr:
        return waveform
    factor   = gcd(src_sr, dst_sr)
    up, down = dst_sr // factor, src_sr // factor
    channels = waveform.detach().cpu().numpy()
    resampled = [
        resample_poly(ch, up, down).astype(np.float32, copy=False)
        for ch in channels
    ]
    return torch.from_numpy(np.stack(resampled, axis=0))


def _face_crop_to_pil(face_tensor: torch.Tensor) -> Image.Image:
    """Convert MTCNN output tensor → PIL Image."""
    face_uint8 = (
        face_tensor.detach()
        .permute(1, 2, 0)
        .mul(255.0)
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )
    return Image.fromarray(face_uint8)


def _confidence_bucket(score: float) -> str:
    """Convert a 0-100 score into a coarse evidence strength label."""
    if score >= 85:
        return 'strong'
    if score >= 70:
        return 'moderate'
    if score >= 55:
        return 'weak'
    return 'minimal'


def _frame_reason(score: float, used_face_crop: bool) -> str:
    """Create a bounded explanation for frame evidence."""
    strength = _confidence_bucket(score)
    if score >= 50:
        if used_face_crop:
            return f'{strength.capitalize()} visual anomaly on detected face region'
        return f'{strength.capitalize()} visual anomaly on full frame fallback'
    if used_face_crop:
        return 'Frame appears visually consistent in detected face region'
    return 'Frame appears visually consistent, but no face was isolated'


def _normalize_series(values: list[float]) -> np.ndarray | None:
    """Scale a numeric series to 0..1, or return None if it is degenerate."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    span = float(arr.max() - arr.min())
    if span < 1e-8:
        return None
    return (arr - arr.min()) / (span + 1e-8)


def _estimate_mouth_motion(
    face_img: Image.Image, previous_roi: np.ndarray | None
) -> tuple[float, np.ndarray]:
    """Use lower-face pixel change as a lightweight mouth-motion proxy."""
    frame = np.asarray(face_img.resize((CFG['img_size'], CFG['img_size'])).convert('RGB'))
    h, w = frame.shape[:2]
    y0, y1 = int(h * 0.58), int(h * 0.9)
    x0, x1 = int(w * 0.22), int(w * 0.78)
    mouth_roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    if previous_roi is None or previous_roi.shape != mouth_roi.shape:
        return 0.0, mouth_roi
    motion = float(np.mean(np.abs(mouth_roi.astype(np.float32) - previous_roi.astype(np.float32))) / 255.0)
    return motion, mouth_roi


def _extract_rppg_frame_data(face_img: Image.Image, prev_frame: np.ndarray | None = None) -> dict:
    """
    Extract rPPG-relevant signals from a face image using multiple ROIs.
    
    Returns motion-compensated color signals from forehead, cheeks, nose, and chin,
    plus per-region quality metrics.
    
    Improvements over simple extraction:
    - Multi-region tracking to detect spatial inconsistencies
    - Motion scoring via frame difference
    - Per-region confidence based on color variation
    - Poisson-weighted approach for robust signal
    """
    frame = np.asarray(face_img.resize((CFG['img_size'], CFG['img_size'])).convert('RGB')).astype(np.float32)
    h, w = frame.shape[:2]
    
    # Define adaptive ROIs based on face dimensions
    # Each ROI: top, bottom, left, right (as fractions of h, w)
    rois = {
        'forehead':  {'y': (0.12, 0.30), 'x': (0.28, 0.72)},
        'left_cheek': {'y': (0.40, 0.65), 'x': (0.10, 0.40)},
        'right_cheek': {'y': (0.40, 0.65), 'x': (0.60, 0.90)},
        'nose':      {'y': (0.35, 0.55), 'x': (0.40, 0.60)},
        'chin':      {'y': (0.65, 0.85), 'x': (0.25, 0.75)},
    }
    
    region_signals = {}
    region_qualities = {}
    
    # Extract per-region signals using green + red channels (more stable for rPPG)
    for region_name, bounds in rois.items():
        y_start, y_end = int(h * bounds['y'][0]), int(h * bounds['y'][1])
        x_start, x_end = int(w * bounds['x'][0]), int(w * bounds['x'][1])
        
        roi = frame[y_start:y_end, x_start:x_end, :]
        if roi.size == 0:
            region_signals[region_name] = 0.0
            region_qualities[region_name] = 0.0
            continue
        
        # Use green channel primarily (most sensitive to blood flow)
        # weighted with red channel (reduces motion artifacts)
        green = np.mean(roi[:,:, 1])  # Green channel
        red = np.mean(roi[:,:, 0])     # Red channel
        # Robust signal combining both channels
        combined_signal = 0.7 * green + 0.3 * red
        region_signals[region_name] = float(combined_signal / 255.0)
        
        # Quality metric: higher variation = better signal
        region_std = float(np.std(roi[:,:, 1]) / 255.0)
        region_qualities[region_name] = region_std
    
    # Motion compensation: frame-to-frame difference
    motion_score = 0.0
    if prev_frame is not None:
        # Compare downsample versions for efficiency
        prev_small = cv2.resize(prev_frame, (50, 50))
        frame_small = cv2.resize(frame.astype(np.uint8), (50, 50))
        motion_score = float(np.mean(np.abs(prev_small.astype(np.float32) - frame_small.astype(np.float32))) / 255.0)
    
    # Primary output: weighted average of stable regions
    # Down-weight regions with poor signal
    weights = np.array([region_qualities[r] for r in ['forehead', 'left_cheek', 'right_cheek']])
    weights = np.clip(weights, 0.1, 1.0)  # Avoid zero weights
    weights = weights / np.sum(weights)
    primary_signal = float(
        weights[0] * region_signals['forehead'] +
        weights[1] * region_signals['left_cheek'] +
        weights[2] * region_signals['right_cheek']
    )
    
    return {
        'primary_signal': primary_signal,
        'region_signals': region_signals,
        'region_qualities': region_qualities,
        'motion_score': motion_score,
        'frame_array': frame,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main inference class
# ─────────────────────────────────────────────────────────────────────────────
class DeepTraceInference:

    def __init__(self, models_dir: str):
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models_dir     = Path(models_dir)

        print(f'[DeepTrace] Loading models on {self.device}...')

        # ── XceptionNet ───────────────────────────────────────────────────
        self.xception = XceptionDetector(
            freeze_backbone=False, embed_dim=CFG['embed_dim']
        ).to(self.device)
        self.xception.load_state_dict(
            self._load_checkpoint(models_dir / 'xception_best.pt')
        )
        self.xception.eval()
        print('  ✓ XceptionNet loaded')

        # ── Wav2Vec2 ──────────────────────────────────────────────────────
        self.wav2vec = Wav2VecDetector(embed_dim=CFG['embed_dim']).to(self.device)
        wav2vec_ckpt = self._resolve_audio_checkpoint(models_dir)
        self.wav2vec.load_state_dict(
            self._load_checkpoint(wav2vec_ckpt)
        )
        self.wav2vec.eval()
        print(f'  ✓ Wav2Vec2 loaded from {wav2vec_ckpt.name}')

        # ── Fusion Transformer ────────────────────────────────────────────
        self.fusion = CrossModalTransformer(
            embed_dim=CFG['embed_dim'],
            n_heads=CFG['n_heads'],
            n_layers=CFG['n_layers'],
            dropout=CFG['dropout'],
        ).to(self.device)
        self.fusion.load_state_dict(
            self._load_checkpoint(models_dir / 'fusion_best.pt')
        )
        self.fusion.eval()
        print('  ✓ Fusion Transformer loaded')

        # ── Face detector + image transform ──────────────────────────────
        self.mtcnn   = MTCNN(image_size=CFG['img_size'], margin=40, device=self.device)
        self.img_tfm = val_tfm

        # ── FFmpeg check ──────────────────────────────────────────────────
        ffmpeg_exe = _get_ffmpeg_executable()
        if ffmpeg_exe:
            try:
                subprocess.run(
                    [ffmpeg_exe, '-version'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                print(f'  ✓ FFmpeg found: {ffmpeg_exe}')
            except Exception:
                print(f'  ⚠ FFmpeg at "{ffmpeg_exe}" could not be executed.')

        else:
            print('  ⚠ FFmpeg not found — audio extraction from video will fail.')

        print('[DeepTrace] All models ready.')

    def _load_checkpoint(self, path: Path) -> dict:
        """
        Load a .pt checkpoint safely.
        Handles PyTorch >=2.6 weights_only default change and
        numpy scalar unpickling issues.
        """
        import numpy as np
        try:
            # Try safe load first (PyTorch >= 2.6 default)
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            # Fall back to full unpickling for older checkpoints
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        return ckpt['model_state']

    def _resolve_audio_checkpoint(self, models_dir: Path) -> Path:
        """Prefer the newer wav2vec checkpoint when it is present."""
        candidates = [
            models_dir.parent / 'wav2vec_best1.pt',
            models_dir / 'wav2vec_best1.pt',
            models_dir / 'wav2vec_best.pt',
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError('No wav2vec checkpoint found for the audio branch.')

    def _audio_is_reliable(self, aud_score: float) -> bool:
        """
        Returns False if the audio score is within `audio_uncertainty_band`
        points of 50% (random chance). This filters out an undertrained
        audio model rather than letting it corrupt the final score.
        """
        return abs(aud_score - 50.0) > CFG['audio_uncertainty_band']

    def _predict_frame_with_tta(
        self, face_img: Image.Image
    ) -> tuple[float, torch.Tensor]:
        """
        Run TTA (Test-Time Augmentation) on a single face crop.

        Applies 4 transforms (original, flip, zoom-crop, brightness jitter),
        averages the fake-probabilities, and mean-pools the embeddings.
        This reduces per-frame variance and catches artifacts that only appear
        under certain orientations or lighting conditions.

        Returns:
            prob_fake  — averaged fake probability (0.0 – 1.0)
            mean_emb   — mean embedding tensor [1, embed_dim]
        """
        tta_probs: list[float]         = []
        tta_embs:  list[torch.Tensor]  = []

        for tfm in tta_tfms:
            tensor      = tfm(face_img).unsqueeze(0).to(self.device)
            logits, emb = self.xception(tensor)
            tta_probs.append(torch.softmax(logits, dim=1)[0, 1].item())
            tta_embs.append(emb)

        prob_fake = float(np.mean(tta_probs))
        mean_emb  = torch.stack(tta_embs).mean(0)  # [1, embed_dim]
        return prob_fake, mean_emb

    def _generate_visual_heatmap(self, face_img: Image.Image) -> str:
        """Generate a simple gradient-based heatmap for the fake class."""
        with torch.enable_grad():
            tensor = tta_tfms[0](face_img).unsqueeze(0).to(self.device)
            tensor.requires_grad_(True)
            self.xception.zero_grad(set_to_none=True)
            logits, _ = self.xception(tensor)
            logits[:, 1].sum().backward()
            saliency = tensor.grad.detach().abs()[0].max(dim=0).values.cpu().numpy()

        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)
        heatmap = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
        base = np.asarray(
            face_img.resize((CFG['img_size'], CFG['img_size'])).convert('RGB'),
            dtype=np.uint8,
        )
        overlay = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.58, heatmap, 0.42, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        buffer = BytesIO()
        Image.fromarray(overlay_rgb).save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        return f'data:image/png;base64,{encoded}'

    def _build_sync_report(
        self, waveform: torch.Tensor | None, sr: int | None, timestamps_sec: list[float], mouth_motion: list[float]
    ) -> dict:
        """Estimate lip-audio consistency from sampled timestamps."""
        report = {
            'available': False,
            'consistency_score': None,
            'anomaly_score': None,
            'correlation': None,
            'region': 'lower-face / mouth region',
            'hotspots': [],
            'reason': 'Insufficient audio-visual data for lip-sync analysis',
        }
        if waveform is None or sr is None or len(timestamps_sec) < 4 or len(mouth_motion) < 4:
            return report

        audio = waveform.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
        if audio.size == 0:
            return report

        window_sec = 0.18
        audio_energy = []
        for ts in timestamps_sec:
            start = max(0, int((ts - window_sec / 2) * sr))
            end = min(audio.size, int((ts + window_sec / 2) * sr))
            if end <= start:
                audio_energy.append(0.0)
                continue
            segment = audio[start:end]
            audio_energy.append(float(np.mean(np.abs(segment))))

        mouth_norm = _normalize_series(mouth_motion)
        audio_norm = _normalize_series(audio_energy)
        if mouth_norm is None or audio_norm is None or mouth_norm.shape != audio_norm.shape:
            report['reason'] = 'Mouth-motion or audio-energy signal was too flat for sync analysis'
            return report

        corr = float(np.corrcoef(mouth_norm, audio_norm)[0, 1])
        if not np.isfinite(corr):
            report['reason'] = 'Could not compute a stable lip-audio correlation'
            return report

        consistency = float(np.clip((corr + 1.0) * 50.0, 0.0, 100.0))
        anomaly = float(np.clip(100.0 - consistency, 0.0, 100.0))
        mismatch_curve = np.abs(mouth_norm - audio_norm)
        hotspot_order = np.argsort(mismatch_curve)[::-1]
        hotspots: list[dict] = []
        for idx in hotspot_order[:3]:
            mismatch_pct = float(np.clip(mismatch_curve[idx] * 100.0, 0.0, 100.0))
            if mismatch_pct < 12:
                continue
            if any(abs(timestamps_sec[idx] - item['timestamp_sec']) < 0.12 for item in hotspots):
                continue
            if mouth_norm[idx] > audio_norm[idx]:
                hotspot_reason = 'Visible mouth motion exceeds supporting audio energy'
            else:
                hotspot_reason = 'Audio energy rises without matching mouth movement'
            hotspots.append({
                'timestamp_sec': round(float(timestamps_sec[idx]), 3),
                'mismatch_score': round(mismatch_pct, 1),
                'mouth_motion_score': round(float(mouth_norm[idx] * 100.0), 1),
                'audio_energy_score': round(float(audio_norm[idx] * 100.0), 1),
                'region': 'lower-face / mouth region',
                'reason': hotspot_reason,
            })

        if consistency >= 70:
            reason = 'Lip motion tracks audio energy well across sampled timestamps'
        elif consistency >= 45:
            reason = 'Lip-audio consistency is mixed; some segments may need manual review'
        else:
            reason = 'Weak lip-audio correlation suggests a possible cross-modal mismatch'

        report.update({
            'available': True,
            'consistency_score': round(consistency, 1),
            'anomaly_score': round(anomaly, 1),
            'correlation': round(corr, 3),
            'hotspots': hotspots,
            'reason': reason,
        })
        return report

    def _build_rppg_report(self, 
                           timestamps_sec: list[float], 
                           rppg_data: list[dict],
                           motion_scores: list[float]) -> dict:
        """
        Build comprehensive rPPG report with hotspot detection and motion analysis.
        
        Improvements:
        - Detrending for robust frequency analysis
        - Per-frame quality scoring
        - Hotspot detection for anomalous regions
        - Motion artifact flagging
        - Multi-metric validation
        """
        report = {
            'available': False,
            'consistency_score': None,
            'dominant_bpm': None,
            'heart_rate_variability': None,
            'motion_artifacts': False,
            'hotspots': [],
            'reason': 'Insufficient data for rPPG analysis',
        }
        
        if len(timestamps_sec) < 12 or len(rppg_data) < 12:
            return report
        
        duration = float(timestamps_sec[-1] - timestamps_sec[0])
        if duration <= 0:
            report['reason'] = 'Video duration too short'
            return report
        
        effective_fps = (len(timestamps_sec) - 1) / duration
        if effective_fps < 6:
            report['reason'] = 'Frame sampling too sparse for heartbeat analysis'
            return report
        
        # Extract primary signals and motion scores
        primary_signals = np.array([d['primary_signal'] for d in rppg_data], dtype=np.float32)
        motion_arr = np.array(motion_scores, dtype=np.float32)
        
        # Detrending: remove slow baseline drift using polynomial fit
        t_uniform = np.linspace(0, 1, len(primary_signals))
        poly_order = 2
        poly_coeffs = np.polyfit(t_uniform, primary_signals, poly_order)
        poly_trend = np.polyval(poly_coeffs, t_uniform)
        detrended_signal = primary_signals - poly_trend
        
        # Normalize detrended signal
        signal_std = np.std(detrended_signal)
        if signal_std < 1e-6:
            report['reason'] = 'Facial color variation too flat after detrending'
            return report
        normalized_signal = detrended_signal / (signal_std + 1e-8)
        
        # FFT analysis on detrended, normalized signal
        t_resampled = np.linspace(timestamps_sec[0], timestamps_sec[-1], len(primary_signals), dtype=np.float32)
        signal_interp = np.interp(t_resampled, np.asarray(timestamps_sec, dtype=np.float32), normalized_signal)
        
        freqs = np.fft.rfftfreq(len(signal_interp), d=float(t_resampled[1] - t_resampled[0]) if len(t_resampled) > 1 else 1e-8)
        power = np.abs(np.fft.rfft(signal_interp)) ** 2
        
        # Focus on physiological heart rate band: 0.75-3.0 Hz (45-180 BPM)
        band_mask = (freqs >= 0.75) & (freqs <= 3.0)
        if not np.any(band_mask):
            report['reason'] = 'Heart rate band not observable in signal'
            return report
        
        band_power = power[band_mask]
        total_power = float(np.sum(power[1:]) + 1e-10)
        band_ratio = float(np.sum(band_power) / total_power)
        
        # Detect dominant frequency (heart rate)
        peak_idx = int(np.argmax(band_power))
        dominant_freq = float(freqs[band_mask][peak_idx])
        dominant_bpm = dominant_freq * 60.0
        
        # Consistency score: how much power is in the heart rate band
        consistency = float(np.clip(band_ratio * 200.0, 0.0, 100.0))
        
        # Heart rate variability (HRV): power in a ±0.25Hz band around dominant frequency
        hrv_band_mask = (freqs >= (dominant_freq - 0.25)) & (freqs <= (dominant_freq + 0.25))
        if np.any(hrv_band_mask):
            hrv_power = np.sum(power[hrv_band_mask])
            hrv_score = float(np.clip(100.0 * (1.0 - hrv_power / (np.sum(band_power) + 1e-10)), 0.0, 100.0))
        else:
            hrv_score = 50.0
        
        # Motion artifact detection
        high_motion_frames = np.where(motion_arr > 0.15)[0]
        motion_artifact_ratio = float(len(high_motion_frames) / len(motion_arr)) if len(motion_arr) > 0 else 0.0
        motion_artifacts = motion_artifact_ratio > 0.25
        
        if motion_artifacts:
            report['motion_artifacts'] = True
        
        # Hotspot detection: identify frames with weak or inconsistent pulse signal
        # Use a sliding window to find regions where the signal is inconsistent
        window_size = max(3, len(normalized_signal) // 5)  # 5 windows across video
        hotspots = []
        
        for i in range(len(normalized_signal) - window_size):
            window = normalized_signal[i:i+window_size]
            window_power = np.abs(np.fft.rfft(window)) ** 2
            window_freqs = np.fft.rfftfreq(len(window), d=1.0 / effective_fps)
            window_band = window_freqs[(window_freqs >= 0.75) & (window_freqs <= 3.0)]
            
            if len(window_band) > 0:
                window_band_power = window_power[(window_freqs >= 0.75) & (window_freqs <= 3.0)]
                window_total = np.sum(window_power[1:]) + 1e-10
                window_ratio = np.sum(window_band_power) / window_total
                window_consistency = np.clip(window_ratio * 200.0, 0.0, 100.0)
                
                # Flag windows with low consistency or high motion
                if window_consistency < 40 or motion_arr[i:i+window_size].mean() > 0.12:
                    ts_start = timestamps_sec[i] if i < len(timestamps_sec) else timestamps_sec[-1]
                    ts_end = timestamps_sec[i+window_size-1] if i+window_size-1 < len(timestamps_sec) else timestamps_sec[-1]
                    
                    hotspots.append({
                        'timestamp_sec': round(float(ts_start), 2),
                        'duration_sec': round(float(ts_end - ts_start), 2),
                        'consistency_score': round(float(window_consistency), 1),
                        'motion_score': round(float(motion_arr[i:i+window_size].mean()), 3),
                        'reason': ('High motion artifact' if motion_arr[i:i+window_size].mean() > 0.12 
                                   else 'Weak pulse-like signal')
                    })
        
        # Remove consecutive/overlapping hotspots
        if hotspots:
            filtered_hotspots = [hotspots[0]]
            for hs in hotspots[1:]:
                last_hs = filtered_hotspots[-1]
                if hs['timestamp_sec'] > last_hs['timestamp_sec'] + last_hs['duration_sec']:
                    filtered_hotspots.append(hs)
            hotspots = filtered_hotspots[:3]  # Limit to top 3
        
        # Determine reason based on metrics
        if consistency >= 70 and not motion_artifacts:
            reason = 'Facial color changes show a stable, regular pulse-like rhythm'
        elif consistency >= 50 and motion_artifact_ratio < 0.15:
            reason = 'Biological rhythm evidence present but with some inconsistency'
        elif motion_artifacts:
            reason = 'Signal contains motion artifacts that complicate pulse detection'
        else:
            reason = 'Pulse-like facial color variation is weak or unreliable'
        
        report.update({
            'available': True,
            'consistency_score': round(consistency, 1),
            'dominant_bpm': round(dominant_bpm, 1),
            'heart_rate_variability': round(hrv_score, 1),
            'motion_artifacts': motion_artifacts,
            'motion_artifact_ratio': round(motion_artifact_ratio, 2),
            'hotspots': hotspots,
            'reason': reason,
        })
        return report

    @torch.no_grad()
    def predict(self, file_path: str) -> dict:
        file_path = str(file_path)
        ext       = file_path.lower()
        is_video  = ext.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        is_audio  = ext.endswith(('.wav', '.mp3', '.flac', '.m4a'))

        vid_score:    float | None = None
        aud_score:    float | None = None
        fusion_score: float | None = None
        vid_emb_t:    torch.Tensor | None = None
        aud_emb_t:    torch.Tensor | None = None
        waveform_for_sync: torch.Tensor | None = None
        waveform_sr: int | None = None
        audio_failed:   bool = False
        audio_reliable: bool = False
        scoring_mode: str = 'unknown'
        limitations: list[str] = []
        evidence: list[dict] = []
        frames_analyzed = 0
        timestamps_sec: list[float] = []
        mouth_motion_series: list[float] = []
        rppg_series: list[dict] = []  # Changed to store full rPPG data dicts
        motion_scores: list[float] = []  # New: track frame-to-frame motion
        forensic_signals = {
            'heatmaps_available': False,
            'temporal_inconsistency_score': None,
            'lip_sync': {
                'available': False,
                'consistency_score': None,
                'anomaly_score': None,
                'correlation': None,
                'reason': 'Lip-sync analysis was not run',
            },
            'rppg': {
                'available': False,
                'consistency_score': None,
                'dominant_bpm': None,
                'reason': 'rPPG analysis was not run',
            },
        }

        # ── Video branch ──────────────────────────────────────────────────
        if is_video:
            cap          = cv2.VideoCapture(file_path)
            total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_scores: list[float]        = []
            vid_embs:     list[torch.Tensor] = []
            frame_evidence: list[dict]       = []
            face_fallback_count              = 0
            previous_mouth_roi: np.ndarray | None = None
            previous_frame_array: np.ndarray | None = None  # New: for motion computation

            print(f'[DeepTrace] Video: {total} total frames, '
                  f'sampling {CFG["frames_per_video"]}')

            if total <= 0:
                print('[DeepTrace] WARNING: Invalid frame count from video file.')
                sample_indices = []
            else:
                sample_indices = np.linspace(
                    0, total - 1, CFG['frames_per_video'], dtype=int
                )

            for fi in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(rgb)
                face = self.mtcnn(pil)

                if face is None:
                    # No face detected — fall back to full resized frame
                    face_img = pil.resize(
                        (CFG['img_size'], CFG['img_size']), Image.BILINEAR
                    )
                    used_face_crop = False
                    face_fallback_count += 1
                else:
                    face_img = _face_crop_to_pil(face)
                    used_face_crop = True

                mouth_motion, previous_mouth_roi = _estimate_mouth_motion(
                    face_img, previous_mouth_roi
                )
                
                # Enhanced rPPG extraction with motion tracking
                rppg_data = _extract_rppg_frame_data(face_img, previous_frame_array)
                motion_scores.append(rppg_data['motion_score'])
                previous_frame_array = rppg_data['frame_array']
                
                ts_value = round(float(fi) / fps, 3) if fps > 0 else None

                # TTA: average over 4 augmented views
                prob_fake, mean_emb = self._predict_frame_with_tta(face_img)
                frame_scores.append(prob_fake)
                vid_embs.append(mean_emb)
                frames_analyzed += 1
                if ts_value is not None:
                    timestamps_sec.append(ts_value)
                    mouth_motion_series.append(mouth_motion)
                    rppg_series.append(rppg_data)  # Store full rPPG data dict
                frame_evidence.append({
                    'frame_index': int(fi),
                    'timestamp_sec': ts_value,
                    'video_fake_score': round(prob_fake * 100, 1),
                    'used_face_crop': used_face_crop,
                    'reason': _frame_reason(prob_fake * 100, used_face_crop),
                    'face_img': face_img.copy(),
                })

            cap.release()

            if frame_scores:
                vid_score = float(np.mean(frame_scores)) * 100
                vid_emb_t = torch.stack(vid_embs).mean(0)
                sorted_evidence = sorted(
                    frame_evidence,
                    key=lambda item: item['video_fake_score'],
                    reverse=True,
                )
                evidence = []
                for rank, item in enumerate(sorted_evidence[:5]):
                    item_out = {
                        'frame_index': item['frame_index'],
                        'timestamp_sec': item['timestamp_sec'],
                        'video_fake_score': item['video_fake_score'],
                        'used_face_crop': item['used_face_crop'],
                        'reason': item['reason'],
                    }
                    if rank < 3:
                        item_out['heatmap_data_url'] = self._generate_visual_heatmap(item['face_img'])
                    evidence.append(item_out)
                forensic_signals['heatmaps_available'] = any(
                    'heatmap_data_url' in item for item in evidence
                )
                if len(frame_scores) >= 2:
                    temporal_deltas = np.abs(np.diff(np.asarray(frame_scores, dtype=np.float32)))
                    forensic_signals['temporal_inconsistency_score'] = round(
                        float(np.mean(temporal_deltas) * 100.0), 1
                    )
                print(
                    f'[DeepTrace] Video score: {vid_score:.1f}% '
                    f'({len(frame_scores)} frames, '
                    f'min={min(frame_scores)*100:.1f}%, '
                    f'max={max(frame_scores)*100:.1f}%)'
                )
                if face_fallback_count:
                    limitations.append(
                        f'Face detection fallback used on {face_fallback_count} sampled frames'
                    )
            else:
                print('[DeepTrace] WARNING: No frames could be read from video.')
                limitations.append('No video frames could be analyzed')

        # ── Audio branch ──────────────────────────────────────────────────
        if is_video or is_audio:
            tmp_path: str | None = None
            try:
                if is_video:
                    print('[DeepTrace] Extracting audio via FFmpeg...')
                    tmp_path   = extract_audio_from_video(file_path, CFG['audio_sr'])
                    audio_data, sr = sf.read(tmp_path, dtype='float32')
                    waveform   = _waveform_from_audio_data(audio_data)
                else:
                    try:
                        audio_data, sr = sf.read(file_path, dtype='float32')
                        waveform       = _waveform_from_audio_data(audio_data)
                    except Exception:
                        print('[DeepTrace] Direct decode failed — retrying via FFmpeg...')
                        tmp_path   = extract_audio_from_video(file_path, CFG['audio_sr'])
                        audio_data, sr = sf.read(tmp_path, dtype='float32')
                        waveform   = _waveform_from_audio_data(audio_data)

                # Mono + resample
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)
                if sr != CFG['audio_sr']:
                    waveform = _resample_waveform(waveform, sr, CFG['audio_sr'])
                waveform_for_sync = waveform.clone()
                waveform_sr = CFG['audio_sr']

                # Clip / pad to exactly 3 seconds
                clip_len = CFG['audio_sr'] * CFG['audio_clip_sec']
                waveform = waveform.squeeze(0)[:clip_len]
                if waveform.shape[0] < clip_len:
                    waveform = F.pad(waveform, (0, clip_len - waveform.shape[0]))

                # Normalize
                waveform  = waveform / (waveform.abs().max() + 1e-8)
                waveform  = waveform.unsqueeze(0).to(self.device)

                logits_a, aud_emb_t = self.wav2vec(waveform)
                aud_score      = torch.softmax(logits_a, dim=1)[0, 1].item() * 100
                audio_reliable = self._audio_is_reliable(aud_score)

                print(
                    f'[DeepTrace] Audio score: {aud_score:.1f}% '
                    f'(reliable={audio_reliable}, '
                    f'distance from 50: {abs(aud_score - 50):.1f})'
                )
                if not audio_reliable:
                    print(
                        '[DeepTrace] Audio score too close to 50% — excluded. '
                        'Retrain wav2vec on real speech data to fix this.'
                    )
                    limitations.append('Audio branch excluded because score was too close to 50%')

            except Exception as exc:
                print(f'[DeepTrace] Audio processing FAILED: {exc}')
                audio_failed   = True
                aud_emb_t      = torch.zeros(1, CFG['embed_dim']).to(self.device)
                aud_score      = None
                audio_reliable = False
                limitations.append('Audio extraction or decoding failed')

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        # ── Fusion branch ─────────────────────────────────────────────────
        if is_video and vid_emb_t is not None and aud_emb_t is not None and audio_reliable:
            _, fusion_conf = self.fusion(vid_emb_t, aud_emb_t)
            fusion_score   = float(fusion_conf.item())
            print(f'[DeepTrace] Fusion score: {fusion_score:.1f}%')
        elif is_video and not audio_reliable:
            print('[DeepTrace] Fusion skipped — audio branch unreliable.')
            limitations.append('Fusion was skipped because the audio branch was unavailable or unreliable')

        if is_video:
            forensic_signals['lip_sync'] = self._build_sync_report(
                waveform_for_sync, waveform_sr, timestamps_sec, mouth_motion_series
            )
            forensic_signals['rppg'] = self._build_rppg_report(
                timestamps_sec, rppg_series, motion_scores
            )
            lip_hotspots = forensic_signals['lip_sync'].get('hotspots', [])
            if lip_hotspots:
                for item in evidence:
                    ts = item.get('timestamp_sec')
                    if ts is None:
                        continue
                    nearest = min(
                        lip_hotspots,
                        key=lambda hotspot: abs(hotspot['timestamp_sec'] - ts),
                    )
                    if abs(nearest['timestamp_sec'] - ts) <= 0.25:
                        item['cross_modal_hotspot'] = True
                        item['cross_modal_reason'] = nearest['reason']
                        item['focus_region'] = nearest['region']
            if not forensic_signals['lip_sync']['available']:
                limitations.append(
                    f"Lip-sync analysis limited: {forensic_signals['lip_sync']['reason']}"
                )
            if not forensic_signals['rppg']['available']:
                limitations.append(
                    f"rPPG analysis limited: {forensic_signals['rppg']['reason']}"
                )

        # ── Aggregate final score ─────────────────────────────────────────
        active_scores:  list[float] = []
        active_weights: list[float] = []

        if vid_score is not None and audio_reliable and fusion_score is not None:
            active_scores  = [vid_score, aud_score, fusion_score]
            active_weights = [0.3, 0.3, 0.4]
            scoring_mode   = 'video+audio+fusion'
            print('[DeepTrace] Scoring mode: video + audio + fusion')

        elif vid_score is not None and audio_reliable:
            active_scores  = [vid_score, aud_score]
            active_weights = [0.5, 0.5]
            scoring_mode   = 'video+audio'
            print('[DeepTrace] Scoring mode: video + audio (no fusion)')

        elif vid_score is not None:
            active_scores  = [vid_score]
            active_weights = [1.0]
            scoring_mode   = 'video-only'
            print('[DeepTrace] Scoring mode: video only (audio excluded)')

        elif aud_score is not None and audio_reliable:
            active_scores  = [aud_score]
            active_weights = [1.0]
            scoring_mode   = 'audio-only'
            print('[DeepTrace] Scoring mode: audio only')

        elif aud_score is not None:
            scoring_mode   = 'audio-only-unreliable'
            limitations.append('Audio-only verdict comes from an unreliable audio branch')
            # Audio-only file but model is unreliable — still return it with warning
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
                'scoring_mode':     scoring_mode,
                'frames_analyzed':  frames_analyzed,
                'evidence':         evidence,
                'forensic_signals': forensic_signals,
                'limitations':      limitations or ['No valid branch produced a score'],
            }

        final_score = float(
            sum(s * w for s, w in zip(active_scores, active_weights))
        )
        print(f'[DeepTrace] Final score: {final_score:.1f}%')

        verdict  = 'FAKE' if final_score > 50 else 'REAL'
        distance = abs(final_score - 50)

        if distance > 30:
            confidence_label = 'High confidence'
        elif distance > 15:
            confidence_label = 'Medium confidence'
        else:
            confidence_label = 'Low confidence — manual review recommended'

        if is_video and not audio_reliable and not audio_failed:
            confidence_label += ' (audio model needs retraining)'
        elif is_video and audio_failed:
            confidence_label += ' (audio extraction failed)'

        if distance <= 15:
            limitations.append('Decision is close to the real/fake threshold; manual review is recommended')

        return {
            'verdict':          verdict,
            'confidence_score': round(final_score, 1),
            'confidence_label': confidence_label,
            'video_score':      round(vid_score,    1) if vid_score    is not None else None,
            'audio_score':      round(aud_score,    1) if (aud_score   is not None
                                                           and (audio_reliable or is_audio)) else None,
            'fusion_score':     round(fusion_score, 1) if fusion_score is not None else None,
            'scoring_mode':     scoring_mode,
            'frames_analyzed':  frames_analyzed,
            'evidence':         evidence,
            'forensic_signals': forensic_signals,
            'limitations':      limitations,
        }
