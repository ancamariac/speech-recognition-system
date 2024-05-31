import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Function to load and preprocess audio files
def load_audio_files(file_paths):
    audio_data = []
    for file_path in file_paths:
        y, sr = librosa.load(file_path, sr=None)
        y = librosa.effects.trim(y)[0]  # Trimming silence
        y = librosa.effects.preemphasis(y)  # Applying preemphasis
        
        # Applying a low-pass filter for denoising
        y = librosa.effects.harmonic(y)
        
        audio_data.append((y, sr))
    return audio_data

# Function to extract features using MFCC and Fourier Transform
def extract_features(audio_data):
    features = []
    for y, sr in audio_data:
        # Applying Short-Time Fourier Transform (STFT)
        stft = np.abs(librosa.stft(y))
        # Compute the log power spectrum
        log_power_spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate features
        combined_features = np.concatenate((log_power_spectrogram.flatten(), mfcc.flatten(), mfcc_delta.flatten(), mfcc_delta2.flatten()))
        features.append(combined_features)
    return features

# Function to pad or trim features to a specified shape
def pad_or_trim_features(features, target_length):
    padded_features = []
    for feature in features:
        feature_length = feature.shape[0]
        if feature_length < target_length:
            # Pad the feature
            pad_width = [(0, target_length - feature_length)]
            padded_feature = np.pad(feature, pad_width, mode='constant')
        elif feature_length > target_length:
            # Trim the feature
            padded_feature = feature[:target_length]
        else:
            padded_feature = feature
        padded_features.append(padded_feature)
    return padded_features

# Load and preprocess audio data for training (replace with your own file paths)
speaker1_files = ['voices/anca1.wav', 'voices/anca2.wav', 'voices/anca3.wav']  # Add more files for speaker1
speaker2_files = ['voices/malina1.wav', 'voices/malina2.wav', 'voices/malina3.wav']  # Add more files for speaker2
speaker3_files = ['voices/sample1.wav', 'voices/sample2.wav', 'voices/sample3.wav']  # Add more files for speaker3

speaker1_audio = load_audio_files(speaker1_files)
speaker2_audio = load_audio_files(speaker2_files)
speaker3_audio = load_audio_files(speaker3_files)

# Extract features for training
speaker1_features = extract_features(speaker1_audio)
speaker2_features = extract_features(speaker2_audio)
speaker3_features = extract_features(speaker3_audio)

# Find the maximum length among all features
max_length = max(feature.shape[0] for feature in (speaker1_features + speaker2_features + speaker3_features))

# Pad or trim features to have the same length
speaker1_features = pad_or_trim_features(speaker1_features, max_length)
speaker2_features = pad_or_trim_features(speaker2_features, max_length)
speaker3_features = pad_or_trim_features(speaker3_features, max_length)

# Combine features and labels
X_train = np.vstack([speaker1_features, speaker2_features, speaker3_features])
y_train = np.hstack([np.zeros(len(speaker1_features)), np.ones(len(speaker2_features)), np.full(len(speaker3_features), 2)])

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train a classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Save the trained model and scaler to files
joblib.dump(clf, 'speaker_recognition_model.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')

# Save the maximum length for feature padding/trimming
joblib.dump(max_length, 'max_feature_length.pkl')

print("Model, scaler, and feature length saved successfully.")

# Load and preprocess audio data for testing (replace with your own file paths)
unknown_speaker_files = ['anca3.wav']  # Add files for unknown speaker
unknown_speaker_audio = load_audio_files(unknown_speaker_files)
unknown_speaker_features = extract_features(unknown_speaker_audio)

# Pad or trim features to have the same shape as the features used for training
unknown_speaker_features = pad_or_trim_features(unknown_speaker_features, max_length)

# Standardize features
X_test = scaler.transform(np.array(unknown_speaker_features))

# Predict labels for unknown speaker
y_pred = clf.predict(X_test)

# Print predicted labels
print("Predicted labels for unknown speaker:")
for i, label in enumerate(y_pred):
   print(f"File: {unknown_speaker_files[i]}, Predicted Label: {label}")