import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import joblib

# Function to load and preprocess audio files
def load_audio_files(file_paths):
    audio_data = []
    for file_path in file_paths:
        y, sr = librosa.load(file_path, sr=None)
        audio_data.append((y, sr))
    return audio_data

# Function to extract features using Fourier Transform
def extract_features(audio_data):
    features = []
    for y, sr in audio_data:
        # Applying Short-Time Fourier Transform (STFT)
        stft = np.abs(librosa.stft(y))
        # Compute the log power spectrum
        log_power_spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
        features.append(log_power_spectrogram.flatten())
    return features

# Load and preprocess audio data for training (replace with your own file paths)
speaker1_files = ['anca1.wav', 'anca2.wav']  # Add more files for speaker1
speaker2_files = ['oana1.wav', 'oana2.wav']  # Add more files for speaker2
speaker3_files = ['bogdan1.wav', 'bogdan2.wav']  # Add more files for speaker3

# Function to pad or trim features to a specified shape
def pad_or_trim_features(features, target_shape):
    padded_features = []
    for feature in features:
        feature_shape = feature.shape
        if feature_shape[0] < target_shape[0]:
            # Pad the feature
            pad_width = [(0, target_shape[0] - feature_shape[0])] + [(0, 0)] * (len(feature_shape) - 1)
            padded_feature = np.pad(feature, pad_width, mode='constant')
        elif feature_shape[0] > target_shape[0]:
            # Trim the feature
            padded_feature = feature[:target_shape[0], :]
        else:
            padded_feature = feature
        padded_features.append(padded_feature)
    return padded_features

speaker1_audio = load_audio_files(speaker1_files)
speaker2_audio = load_audio_files(speaker2_files)
speaker3_audio = load_audio_files(speaker3_files)

# Extract features for training
speaker1_features = extract_features(speaker1_audio)
speaker2_features = extract_features(speaker2_audio)
speaker3_features = extract_features(speaker3_audio)

# Find the maximum shape among all features
max_shape = max(feature.shape for feature in (speaker1_features + speaker2_features + speaker3_features))

# Pad or trim features to have the same shape
speaker1_features = pad_or_trim_features(speaker1_features, max_shape)
speaker2_features = pad_or_trim_features(speaker2_features, max_shape)
speaker3_features = pad_or_trim_features(speaker3_features, max_shape)

# Combine features and labels
X_train = np.vstack([speaker1_features, speaker2_features, speaker3_features])
y_train = np.hstack([np.zeros(len(speaker1_features)), np.ones(len(speaker2_features)), np.full(len(speaker3_features), 2)])

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train a classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(clf, 'speaker_recognition_model.pkl')

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Save the trained scaler to a file
joblib.dump(scaler, 'standard_scaler.pkl')


# # Load and preprocess audio data for testing (replace with your own file paths)
# unknown_speaker_files = ['anca4.wav']  # Add files for unknown speaker
# unknown_speaker_audio = load_audio_files(unknown_speaker_files)
# unknown_speaker_features = extract_features(unknown_speaker_audio)

# # Pad or trim features to have the same shape as the features used for training
# unknown_speaker_features = pad_or_trim_features(unknown_speaker_features, max_shape)

# # Standardize features
# X_test = scaler.transform(np.array(unknown_speaker_features))

# # Predict labels for unknown speaker
# y_pred = clf.predict(X_test)

# # Print predicted labels
# print("Predicted labels for unknown speaker:")
# for i, label in enumerate(y_pred):
#     print(f"File: {unknown_speaker_files[i]}, Predicted Label: {label}")
