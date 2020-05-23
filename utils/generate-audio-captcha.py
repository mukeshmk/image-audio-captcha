# !/usr/bin/env python3
import os
import random
import argparse
from gtts import gTTS

file_format = '.wav'


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
    parser.add_argument('--audio-dict', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.audio_dict is not None:
        args.length = -1
        args.count = -1
        print("Generating Symbol Set Data")

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

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    if args.audio_dict:
        args.count = len(captcha_symbols)
        args.scramble = False

    for i in range(args.count):
        if args.audio_dict:
            captcha_text = captcha_symbols[i]
        else:
            captcha_text = ''.join([random.choice(captcha_symbols) for j in range(args.length)])
        captcha_name_scrambled = captcha_text
        if args.scramble:
            captcha_name_scrambled = scramble_captcha_name(captcha_text)
        captcha_file_name = os.path.join(args.output_dir, captcha_name_scrambled + file_format)
        if os.path.exists(captcha_file_name):
            version = 1
            while os.path.exists(
                    os.path.join(args.output_dir, captcha_name_scrambled + '_' + str(version) + file_format)):
                version += 1
            captcha_file_name = os.path.join(args.output_dir, captcha_name_scrambled + '_' + str(version) + file_format)

        tts = gTTS(captcha_text, 'en')
        if args.audio_dict and not os.path.exists(args.output_dir + '\\' + captcha_name_scrambled):
            os.makedirs(os.path.join(args.output_dir, captcha_name_scrambled))
            captcha_file_name = os.path.join(args.output_dir + '\\' + captcha_name_scrambled,
                                             captcha_name_scrambled + file_format)
        tts.save(captcha_file_name)


if __name__ == '__main__':
    main()
