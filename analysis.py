import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.fft import fft, ifft

# Calea către fișierul audio
file_path = 'voices/anca1.wav'

# Încărcarea fișierului audio
y, sr = librosa.load(file_path, sr=None)

# Reprezentarea temporală
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Reprezentarea temporală')
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.show()

# Calcularea Transformatei Fourier
n = len(y)  # Lungimea semnalului
T = 1/sr    # Intervalul de timp între eșantioane
yf = fft(y)
xf = np.fft.fftfreq(n, T)

# Componentele de bază ale transformatei Fourier
yf_magnitude = 2.0/n * np.abs(yf[:n//2])
yf_real = np.real(yf)
yf_imag = np.imag(yf)

# Spectrul de frecvență (magnitudine)
plt.figure(figsize=(14, 5))
plt.plot(xf[:n//2], yf_magnitude)
plt.title('Spectrul de frecvență (Magnitudine)')
plt.xlabel('Frecvență (Hz)')
plt.ylabel('Amplitudine')
plt.grid()
plt.show()

# Componentele reale ale transformatei Fourier
plt.figure(figsize=(14, 5))
plt.plot(xf[:n//2], 2.0/n * yf_real[:n//2])
plt.title('Componenta reală a transformatei Fourier')
plt.xlabel('Frecvență (Hz)')
plt.ylabel('Amplitudine')
plt.grid()
plt.show()

# Componentele imaginare ale transformatei Fourier
plt.figure(figsize=(14, 5))
plt.plot(xf[:n//2], 2.0/n * yf_imag[:n//2])
plt.title('Componenta imaginară a transformatei Fourier')
plt.xlabel('Frecvență (Hz)')
plt.ylabel('Amplitudine')
plt.grid()
plt.show()

# Reconstruirea semnalului folosind doar primele N componente
N = 5
reconstructed_signals = []

for i in range(1, N+1):
    yf_partial = np.zeros_like(yf)
    yf_partial[:i] = yf[:i]
    yf_partial[-i:] = yf[-i:]
    y_partial = ifft(yf_partial)
    reconstructed_signals.append(np.real(y_partial))

# Plotarea semnalelor reconstruite
plt.figure(figsize=(14, 10))

for i in range(N):
    plt.subplot(N, 1, i+1)
    librosa.display.waveshow(reconstructed_signals[i], sr=sr)
    plt.title(f'Semnal reconstruit folosind primele {i+1} componente Fourier')
    plt.xlabel('Timp (s)')
    plt.ylabel('Amplitudine')

plt.tight_layout()
plt.show()

# Spectrograma
plt.figure(figsize=(14, 5))
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrograma')
plt.xlabel('Timp (s)')
plt.ylabel('Frecvență (Hz)')
plt.show()

