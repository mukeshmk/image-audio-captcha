import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd

audio_fpath = "D:\\code\\github-clones\\image-audio-captcha\\t\\"
audio_clips = os.listdir(audio_fpath)
print("No. of .mp3 files in audio folder = ", len(audio_clips))

# Load audio file and visualize its waveform (using librosa)
x, sr = librosa.load(audio_fpath + audio_clips[0], sr=44100)

print(type(x), type(sr))
print(x.shape, sr)
plt.figure(figsize=(5.12, 2.56))
librosa.display.waveplot(x, sr=sr)

# Convert the audio waveform to spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5.12, 2.56))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

# Applying log transformation on the loaded audio signals
plt.figure(figsize=(5.12, 2.56))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
