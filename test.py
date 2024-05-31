import numpy as np
import librosa
import joblib
import sounddevice as sd

# Load the trained model, scaler, and maximum feature length
clf = joblib.load('speaker_recognition_model.pkl')
scaler = joblib.load('standard_scaler.pkl')
max_length = joblib.load('max_feature_length.pkl')

# Function to record audio
def record_audio(duration=5, fs=44100):
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Audio recording complete.")
    return audio.flatten(), fs

# Function to preprocess audio
def preprocess_audio(audio_data, sr, target_length):
    y = audio_data
    y = librosa.effects.trim(y)[0]  # Trimming silence
    y = librosa.effects.preemphasis(y)  # Applying preemphasis
    y = librosa.effects.harmonic(y)  # Low-pass filter for denoising
    
    stft = np.abs(librosa.stft(y))
    log_power_spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    combined_features = np.concatenate((log_power_spectrogram.flatten(), mfcc.flatten(), mfcc_delta.flatten(), mfcc_delta2.flatten()))
    
    # Pad or trim features to have the same length as the features used for training
    feature_length = combined_features.shape[0]
    if feature_length < target_length:
        pad_width = [(0, target_length - feature_length)]
        combined_features = np.pad(combined_features, pad_width, mode='constant')
    elif feature_length > target_length:
        combined_features = combined_features[:target_length]
    
    return combined_features

# Record audio from the microphone
audio_data, sr = record_audio()

# Preprocess the recorded audio
features = preprocess_audio(audio_data, sr, max_length)
features = features.reshape(1, -1)  # Reshape for the classifier

# Standardize features
X_test = scaler.transform(features)

# Predict the label
y_pred = clf.predict(X_test)

# Print the predicted label
print(f"Predicted Label: {int(y_pred[0])}")
