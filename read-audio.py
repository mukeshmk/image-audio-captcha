#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
from pydub import AudioSegment

matplotlib.rcParams['figure.figsize'] = [15, 10]


def convert_audio_image(pwd, input_dir, output_dir):
    file_names = os.listdir(input_dir)
    for file in file_names:
        mp3_audio_path = pwd + '\\' + input_dir + '\\' + file
        sound = AudioSegment.from_mp3(mp3_audio_path)
        wav_audio_path = pwd + '/wavAudio/' + file.strip('.mp3') + '.wav'
        sound.export(wav_audio_path, format="wav")

        plt.interactive(False)
        clip, sample_rate = librosa.load(wav_audio_path, sr=None)
        fig = plt.figure(figsize=[0.415, 0.210])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        filename = os.path.join(output_dir, file + '.jpg')
        plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()
        fig.clf()
        plt.close(fig)
        plt.close('all')
        del clip, sample_rate, fig, ax, S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help='input dir of captcha audio set', type=str)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
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

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    tempDir = 'tempDir'
    if not os.path.exists(tempDir):
        print("Creating temp directory " + tempDir)
        os.makedirs(tempDir)
    convert_audio_image(os.getcwd(), args.input_dir, args.output_dir)
    if os.path.exists(tempDir):
        print("Deleting temp directory " + tempDir)
        os.rmdir(tempDir)


if __name__ == '__main__':
    main()
