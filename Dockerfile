# Use Python 3.9
FROM python:3.9

# Set working directory
WORKDIR /code

# 1. Install System Dependencies (Needed for Audio/Librosa)
RUN apt-get update && apt-get install -y ffmpeg

# 2. Copy the intricate requirements & Install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 3. Copy Code
COPY . /code

# 4. FIX PERMISSIONS (The Magic Step for Hugging Face)
# Create a writable cache folder for the AI models
RUN mkdir -p /code/cache && chmod -R 777 /code/cache
ENV HF_HOME=/code/cache
ENV TRANSFORMERS_CACHE=/code/cache

# 5. Start Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]