# ReadNet public demo: record a reading, get the ASER level and the child's
# mistakes. No API keys: transcription and GOP both run on open wav2vec2
# models baked into the image. Built by readnet/deploy/build_space.py.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1 READNET_PUBLIC=1

# Download both language models now, so the first visitor does not wait.
RUN python -c "from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor; \
[(AutoModelForCTC.from_pretrained(m), Wav2Vec2CTCTokenizer.from_pretrained(m), Wav2Vec2FeatureExtractor.from_pretrained(m)) \
 for m in ('Harveenchadha/vakyansh-wav2vec2-hindi-him-4200', 'Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100')]"

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "-m", "readnet.web", "--host", "0.0.0.0", "--port", "7860"]
