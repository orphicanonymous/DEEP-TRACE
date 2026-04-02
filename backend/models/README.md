# Model Files

This folder is reserved for production model weights used by the backend.

Recommended production setup:

- Do not commit large `.pt` files directly to the main Git repo.
- Store model files in external storage or use Git LFS if you must version them.
- Download or mount the model files during deployment so `backend/app.py` can load them at runtime.

Expected files:

- `fusion_best.pt`
- `wav2vec_best.pt`
- `xception_best.pt`
