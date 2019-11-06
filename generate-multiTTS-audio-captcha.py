# !/usr/bin/env python3
import os
import random
import argparse
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
from pydub import AudioSegment

file_format = '.wav'

# This are the dir name from where audio files will be picked up
tts_list = ['pyttsx', 'gtts', 'watson', 'microsoft']
tempDir = 'multiTTS_data'

def create_audio_and_convert(output_dir, captcha_text):
    pwd = os.getcwd()
    work_dir = pwd + '\\symbols\\'
    combined_sounds = None
    for symbol in captcha_text:
        tts_service = random.choice(tts_list)
        raw_path = work_dir + tts_service + '\\' + symbol
        sound = AudioSegment.from_wav(raw_path + ".wav")
        if combined_sounds is None:
            combined_sounds = sound
        else:
            combined_sounds = combined_sounds + sound

    if not os.path.exists(pwd + '\\' + tempDir):
        print("Creating temp directory " + tempDir)
        os.makedirs(pwd + '\\' + tempDir)

    output_dir_tts = pwd + '\\' + tempDir + '\\'
    wav_audio_path = output_dir_tts + captcha_text + '.wav'
    combined_sounds.export(wav_audio_path, format="wav")

    plt.interactive(False)
    clip, sample_rate = librosa.load(wav_audio_path, sr=None)
    fig = plt.figure(figsize=[0.415, 0.210])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = os.path.join(output_dir, captcha_text + '.jpg')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del clip, sample_rate, fig, ax, S



def scramble_captcha_name(captcha_name):
    import hashlib
    m = hashlib.sha1()
    m.update(captcha_name.encode('utf-8'))
    return m.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--scramble', help='Whether to scramble captcha names', default=False, action='store_true')
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--n', help='no of parallel threads to start', type=int, default=6)
    args = parser.parse_args()

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    if args.n is None:
        print("Setting the number of threads to 6")
        args.n = 6

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    process_args = []
    for i in range(args.count):
        captcha_text = ''.join([random.choice(captcha_symbols) for j in range(args.length)])
        captcha_name_scrambled = captcha_text
        if args.scramble:
            captcha_name_scrambled = scramble_captcha_name(captcha_text)
        captcha_file_name = os.path.join(args.output_dir, captcha_name_scrambled + file_format)
        if os.path.exists(captcha_file_name):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, captcha_name_scrambled + '_' + str(version) + file_format)):
                version += 1
            captcha_file_name = os.path.join(args.output_dir, captcha_name_scrambled + '_' + str(version) + file_format)
        process_args.append((args.output_dir, captcha_name_scrambled))

    with Pool(args.n) as p:
        p.starmap(create_audio_and_convert, process_args)


if __name__ == '__main__':
    main()
