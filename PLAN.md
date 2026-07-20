# Project Kickoff: Image to GenZ Caption Generator
**Date**: July 20, 2026 (Day 1 - Hour 1)

## Initial Brainstorm & Goals
The goal of this project is to build a web application that takes an uploaded image and generates a fun, slang-filled GenZ caption with emojis.

### The Plan for the First Hour
1. **Define the Scope**:
   - We need an image processor (Computer Vision) and a caption generation system (NLP).
   - The output caption needs to be in GenZ slang (e.g. "no cap", "fr fr", "vibing", "slaying", "aura points").
2. **Evaluate the Tech Stack**:
   - **Baseline Option**: A classic deep learning pipeline (VGG16 to extract image features and an LSTM network to predict the caption). We can train this on the Flickr8k dataset using Keras.
   - **VLM Option**: Use a Visual Large Language Model (VLM) like the Gemini API. While the baseline option is good for offline/lightweight deployment, an LLM will give us much smarter, contextual slang descriptions.
   - **Decision**: We'll start by building the local VGG16 + LSTM network to establish a baseline model. Then, we will develop a Streamlit frontend. If the local model is too limited in its vocabulary, we will upgrade to a visual LLM (Gemini API) to handle general-purpose captions.
3. **Environment Setup**:
   - Create a `requirements.txt` with essential packages: `streamlit`, `tensorflow`, `keras`, `numpy`, `pillow`, `keras_preprocessing`.

## Todo List for this week:
- Set up local training script / notebook.
- Train local caption generator on Flickr8k.
- Create Streamlit deployment script.
- Pivot / upgrade to VLM API for rich GenZ slang generation.
- Finalize documentation.
