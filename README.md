# DeepTrace

DeepTrace is an AI deepfake detection web application with an accountability-focused workflow. The frontend is built with React + Vite, and the backend is a Flask API that runs the model inference pipeline.

## Repo Structure

- `frontend/`: React + Vite application
- `frontend/src/`: frontend source files
- `backend/`: Flask backend and ML inference code
- `backend/models/`: model weights required by the backend

## Local Setup

### From Repo Root

```bash
npm install
npm run dev
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Environment Variables

Create `frontend/.env` from `frontend/.env.example`.

```env
VITE_BACKEND_URL=http://127.0.0.1:5000
VITE_SUPABASE_URL=https://ktludhxrewwbewkhcnqx.supabase.co
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
```

## Important Notes

- Do not commit `.env` files or local virtual environments.
- If your model files are very large, consider Git LFS or external model storage.
- The backend must have access to the files inside `backend/models/` in production.
- Python 3.12 is recommended for the backend because the pinned PyTorch stack ships Windows wheels for that runtime.
- FFmpeg must be available on `PATH` or via the `FFMPEG_PATH` environment variable for video audio extraction.
- See [MODEL_CARD.md](MODEL_CARD.md) for the current model architecture, inference flow, limitations, and accountable usage notes.
