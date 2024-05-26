import numpy as np
import librosa
import sounddevice as sd
import joblib

# Load the trained model
clf = joblib.load('speaker_recognition_model.pkl')

# Load the trained StandardScaler
scaler = joblib.load('standard_scaler.pkl')

# Function to extract features using Fourier Transform
def extract_features(audio_data):
    # Applying Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(audio_data))
    # Compute the log power spectrum
    log_power_spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    return log_power_spectrogram.flatten()

# Function to preprocess audio data for testing
def preprocess_audio(audio_data):
    # Extract features
    features = extract_features(audio_data)
    
    # Check if the number of features matches the expected size
    expected_size = scaler.mean_.shape[0]
    if len(features) != expected_size:
        # Resize the features to match the expected size
        if len(features) < expected_size:
            # If the features are too short, pad with zeros
            features = np.pad(features, (0, expected_size - len(features)), mode='constant')
        else:
            # If the features are too long, trim the extra
            features = features[:expected_size]
    
    # Standardize features
    X_test = scaler.transform(np.array([features]))
    return X_test

# Function to record audio from microphone for a given duration
def record_audio(duration):
    print("Recording audio...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Audio recording complete.")
    return audio_data.flatten()

# Define sampling rate and duration of recording
fs = 44100  # Sampling rate (samples per second)
duration = 10  # Duration of recording in seconds

# Record audio from microphone
audio_data = record_audio(duration)

# Preprocess the recorded audio
X_test = preprocess_audio(audio_data)

# Make prediction
prediction = clf.predict(X_test)

# Print the predicted label
print("Predicted label:", prediction)
