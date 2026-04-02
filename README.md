# DeepTrace

DeepTrace is an AI deepfake detection web application with an accountability-focused workflow. The frontend is built with React + Vite, and the backend is a Flask API that runs the model inference pipeline.

## Repo Structure

- `src/`: React frontend
- `backend/`: Flask backend and ML inference code
- `backend/models/`: model weights required by the backend
- `deploy.sh` / `deploy.bat`: helper deployment scripts

## Local Setup

### Frontend

```bash
npm install
npm run dev
```

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Environment Variables

Create a root `.env` file from `.env.example`.

```env
VITE_BACKEND_URL=http://127.0.0.1:5000
```

## Deployment

### Frontend on Netlify

- Build command: `npm run build`
- Publish directory: `dist`
- Environment variable: `VITE_BACKEND_URL=https://your-backend-url.onrender.com`

### Backend on Render

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Health check path: `/health`

## Important Notes

- Do not commit `.env` files or local virtual environments.
- If your model files are very large, consider Git LFS or external model storage.
- The backend must have access to the files inside `backend/models/` in production.
