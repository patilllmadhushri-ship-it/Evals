---
title: ReadNet
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Record a child reading Hindi or Marathi, get the ASER level
---

# ReadNet

Record a child reading aloud in Hindi or Marathi and get their ASER reading
level (Beginner, Letter, Word, Paragraph, Story), every word marked right or
wrong, and the letters and sounds they got wrong.

Open the app, allow the microphone, press Record, read the text on screen,
press Stop, then Score this reading. Use Chrome or Edge.

Everything runs inside this Space on open speech models: the Vakyansh
wav2vec2 models for Hindi and Marathi transcribe the reading and score each
sound's pronunciation (GOP). No account or API key is needed, and recordings
are used once and discarded.

Source: https://github.com/patilllmadhushri-ship-it/Evals (the `readnet/` folder).
