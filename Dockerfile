# Use Python 3.9
FROM python:3.9

# Install system dependencies (REQUIRED for Librosa/MP3)
RUN apt-get update && apt-get install -y ffmpeg

# Set work directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the code
COPY . /code

# Create a writable cache directory for Transformers
# (HuggingFace Spaces are read-only except for /tmp and /app)
ENV TRANSFORMERS_CACHE=/code/.cache
ENV HF_HOME=/code/.cache
RUN mkdir -p /code/.cache && chmod 777 /code/.cache

# Run the app (HF Spaces expects port 7860)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]