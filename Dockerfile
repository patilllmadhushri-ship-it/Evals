# Container image for hosting the app — Cloudflare Containers, Cloud Run,
# Fly, a plain VM. Streamlit needs a long-lived Python process and WebSockets,
# so it cannot run on an edge runtime such as Cloudflare Workers or Pages.

FROM python:3.12-slim

# ffmpeg widens the accepted upload formats (MP3/M4A/WebM). Without it the app
# still works, but falls back to whatever libsndfile alone can decode.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stt_eval/ ./stt_eval/
COPY app.py .
COPY .streamlit/config.toml ./.streamlit/config.toml

# Results persist here. Mount a volume to keep runs across restarts; without
# one, resumability lasts only as long as the container.
RUN mkdir -p /app/.stt_eval_runs

# Credentials come from the environment, never from the image. Supply
# DEEPGRAM_API_KEY, SARVAM_API_KEY, ELEVENLABS_API_KEY, OPENAI_API_KEY,
# GOOGLE_API_KEY, and one of ANTHROPIC_API_KEY / OPENROUTER_API_KEY at run
# time — plus APP_PASSCODE for anything with a public URL.
ENV PORT=8501
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0"]
