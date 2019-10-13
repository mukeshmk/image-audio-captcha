import os
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import matplotlib.pyplot as plt
from pydub.silence import split_on_silence
from sklearn.model_selection import train_test_split

FILE_FORMAT = '.mp3'
PWD = os.getcwd()
DATA_PATH = PWD + '\\test\\'
NPY_PATH = PWD + '\\npy\\'
IMG_PATH = DATA_PATH + 'img\\'


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


def wav2mfcc(file_path, n_mfcc=20, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    wave = np.asfortranarray(wave)
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=n_mfcc)

    if max_len > mfcc.shape[1]:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


def save_data_to_array(path=DATA_PATH, max_len=11, n_mfcc=20):
    labels, _, _ = get_labels(path)

    for label in labels:
        mfcc_vectors = []

        wavfiles = [path + wavfile for wavfile in os.listdir(path)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len, n_mfcc=n_mfcc)
            mfcc_vectors.append(mfcc)
        np.save(NPY_PATH + label.strip(FILE_FORMAT) + '.npy', mfcc_vectors)


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(NPY_PATH + labels[0].strip(FILE_FORMAT) + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(NPY_PATH + label.strip(FILE_FORMAT) + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


def convert_audio_to_spectrogram():
    mp3_audio_path = DATA_PATH + '2I1UP8QY' + FILE_FORMAT
    sound = AudioSegment.from_mp3(mp3_audio_path)
    sound_samples = sound.get_array_of_samples()  # Extract signal samples
    samp_freq = sound.frame_rate

    speech_samples_norm = np.array(sound_samples)  # / np.max(np.array(sound_samples))

    start_samp = 0
    end_samp = len(speech_samples_norm)
    win_len = int(samp_freq * .01)  # Window size of 30 ms
    x = librosa.stft(np.array(speech_samples_norm[start_samp:end_samp]), win_length=win_len)
    xdb = librosa.amplitude_to_db(abs(x))
    fg2 = plt.figure(figsize=(5.12, 2.56), dpi=100)
    librosa.display.specshow(xdb, sr=samp_freq, x_axis='time', y_axis='hz', hop_length=win_len / 4)
    plt.axis('off')
    image_path = os.path.join(DATA_PATH, '2I1UP8QY' + '.jpg')
    fg2.savefig(image_path, dpi=100)
    plt.close(fg2)


def split_audio_silence():
    sound = AudioSegment.from_mp3(PWD + '\\00C643SS.wav')
    print(len(sound))
    print(sound.dBFS)
    clips = split_on_silence(sound, min_silence_len=15, silence_thresh=-23)
    print(len(clips))
    c = 0
    for i in clips:
        i.export(str(c) + '.wav')
        c += 1

'''
save_data_to_array(DATA_PATH)

X_train, X_test, y_train, y_test = get_train_test()

max_len = 11
buckets = 20
channels = 1

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)
'''
split_on_silence()
