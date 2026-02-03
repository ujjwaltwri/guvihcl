import base64
import io
import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def decode_base64(self, base64_string):
        """
        Decodes the Base64 string into a file-like object.
        Match this method name with main.py calls.
        """
        try:
            # Decode the string into bytes
            audio_bytes = base64.b64decode(base64_string)
            # Wrap bytes in a file-like object so librosa can read it
            return io.BytesIO(audio_bytes)
        except Exception as e:
            raise ValueError(f"Failed to decode Base64: {str(e)}")

    def extract_features(self, audio_file):
        """
        Loads the audio file and converts it to a log-mel spectrogram.
        """
        try:
            # Load audio using librosa (it handles MP3 via ffmpeg/audioread)
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Create Mel Spectrogram
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            
            # Convert to log scale (dB) for better model processing
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            
            return log_spectrogram
        except Exception as e:
            raise ValueError(f"Error processing audio features: {str(e)}")