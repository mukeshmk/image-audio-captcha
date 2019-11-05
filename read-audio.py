#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
from pydub import AudioSegment
from multiprocessing import Pool

matplotlib.rcParams['figure.figsize'] = [15, 10]


def convert_single_audio_to_image(pwd, input_dir, output_dir, file):
    try:
        print('Processing: {0}'.format(file))
        mp3_audio_path = pwd + '\\' + input_dir + '\\' + file
        # load and process audio
        audio, sr = librosa.load(mp3_audio_path)
        # audio = lowpass(audio, cutoff=3000, sample_freq=sr)
        spec = librosa.stft(np.asfortranarray(audio))
        spec_db = librosa.amplitude_to_db(np.abs(spec))

        # generate plot of size 128x64
        fig = plt.figure(figsize=(1.28, 0.64))
        plt.box(False)
        plt.subplots_adjust(left=0, right=1, bottom=0, wspace=0, hspace=0)
        librosa.display.specshow(spec_db, sr=sr, cmap='gray_r', x_axis='time', y_axis='log')
        fig.savefig(output_dir + '/' + file.strip('.mp3'), bbox_inches=None, pad_inches=0)
        plt.close()
    except Exception as e:
        print('processing {0}: {1}'.format(file, e))


def convert_audio_image(pwd, input_dir, output_dir, n):
    file_names = os.listdir(input_dir)
    process_args = []
    for file in file_names:
        process_args.append((pwd, input_dir, output_dir, file))

    with Pool(n) as p:
        p.starmap(convert_single_audio_to_image, process_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help='input dir of captcha audio set', type=str)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--n', help='no of parallel threads to start', type=int, default=6)
    args = parser.parse_args()

    if args.input_dir is None:
        print("Please specify the captcha image input dir")
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

    tempDir = 'tempDir'
    if not os.path.exists(args.input_dir + tempDir):
        print("Creating temp directory " + tempDir)
        os.makedirs(args.input_dir + tempDir)

    convert_audio_image(os.getcwd(), args.input_dir, args.output_dir, args.n)

    if os.path.exists(args.input_dir + tempDir):
        print("Deleting temp directory " + tempDir)
        os.rmdir(args.input_dir + tempDir)


if __name__ == '__main__':
    main()
