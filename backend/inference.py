import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import timm
from transformers import Wav2Vec2Model
from facenet_pytorch import MTCNN

# ── Config (must match exactly what was used during training) ─────────────────
CFG = {
    'img_size':          299,
    'frames_per_video':  10,
    'audio_sr':          16000,
    'audio_clip_sec':    3,
    'embed_dim':         512,
    'n_heads':           8,
    'n_layers':          2,
    'dropout':           0.1,
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
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
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

        v_attended, _            = self.vid_to_aud_attn(v, a, a)
        a_attended, _            = self.aud_to_vid_attn(a, v, v)

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, v_attended, a_attended], dim=1)
        fused  = self.transformer(tokens)

        cls_out    = fused[:, 0, :]
        logits     = self.head(cls_out)
        scaled     = logits / self.temperature.clamp(min=0.1)
        confidence = torch.softmax(scaled, dim=1)[:, 1] * 100
        return logits, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Main inference class
# ─────────────────────────────────────────────────────────────────────────────
class DeepTraceInference:
    def __init__(self, models_dir: str):
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models_dir     = Path(models_dir)

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

    @torch.no_grad()
    def predict(self, file_path: str) -> dict:
        file_path = str(file_path)
        is_video  = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        is_audio  = file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))

        vid_score, aud_score, fusion_score = None, None, None
        vid_emb_t, aud_emb_t              = None, None

        # ── Video branch ──────────────────────────────────────────────────
        if is_video:
            cap          = cv2.VideoCapture(file_path)
            total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_scores = []
            vid_embs     = []

            for fi in np.linspace(0, total - 1, CFG['frames_per_video'], dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(rgb)
                face = self.mtcnn(pil)

                if face is None:
                    face_img = pil.resize((299, 299))
                else:
                    face_img = Image.fromarray(
                        face.permute(1, 2, 0).numpy().astype(np.uint8))

                tensor      = self.img_tfm(face_img).unsqueeze(0).to(self.device)
                logits, emb = self.xception(tensor)
                prob_fake   = torch.softmax(logits, dim=1)[0, 1].item()
                frame_scores.append(prob_fake)
                vid_embs.append(emb)

            cap.release()
            if frame_scores:
                vid_score = np.mean(frame_scores) * 100
                vid_emb_t = torch.stack(vid_embs).mean(0)

        # ── Audio branch ──────────────────────────────────────────────────
        if is_video or is_audio:
            try:
                waveform, sr = torchaudio.load(file_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)
                if sr != CFG['audio_sr']:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, CFG['audio_sr'])
                clip_len = CFG['audio_sr'] * CFG['audio_clip_sec']
                waveform = waveform.squeeze(0)[:clip_len]
                if waveform.shape[0] < clip_len:
                    waveform = F.pad(waveform, (0, clip_len - waveform.shape[0]))
                waveform  = waveform / (waveform.abs().max() + 1e-8)
                waveform  = waveform.unsqueeze(0).to(self.device)
                logits_a, aud_emb_t = self.wav2vec(waveform)
                aud_score = torch.softmax(logits_a, dim=1)[0, 1].item() * 100
            except Exception as e:
                print(f'Audio processing failed: {e}')
                aud_emb_t = torch.zeros(1, CFG['embed_dim']).to(self.device)
                aud_score = 50.0

        # ── Fusion branch ─────────────────────────────────────────────────
        if is_video and vid_emb_t is not None and aud_emb_t is not None:
            _, fusion_conf = self.fusion(vid_emb_t, aud_emb_t)
            fusion_score   = fusion_conf.item()

        # ── Aggregate final score ─────────────────────────────────────────
        scores = [s for s in [vid_score, aud_score, fusion_score] if s is not None]
        if   len(scores) == 3: weights = [0.3, 0.3, 0.4]
        elif len(scores) == 2: weights = [0.45, 0.55]
        else:                  weights = [1.0]
        final_score = sum(s * w for s, w in zip(scores, weights))

        verdict = 'FAKE' if final_score > 50 else 'REAL'
        confidence_label = (
            'High confidence'   if abs(final_score - 50) > 30 else
            'Medium confidence' if abs(final_score - 50) > 15 else
            'Low confidence — manual review recommended'
        )

        return {
            'verdict':          verdict,
            'confidence_score': round(final_score, 1),
            'confidence_label': confidence_label,
            'video_score':      round(vid_score,    1) if vid_score    is not None else None,
            'audio_score':      round(aud_score,    1) if aud_score    is not None else None,
            'fusion_score':     round(fusion_score, 1) if fusion_score is not None else None,
        }
