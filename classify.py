#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
import tensorflow.keras as keras
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def decode(characters, y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--is-audio', help='is audio boolean', type=bool, default=True)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with tf.device('/gpu:0'):
        with open(args.output, 'w') as output_file:
            json_file = open(args.model_name + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name + '.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            for x in os.listdir(args.captcha_dir):
                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                rgb_data = None
                if not args.is_audio:
                    # gray scaling the image
                    gray_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
                    # applying adaptive threshold
                    adaptive = cv2.adaptiveThreshold(gray_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,
                                                     2)

                    kernel = np.ones((1, 1), np.uint8)
                    # dilating the image
                    dilation = cv2.dilate(adaptive, kernel, iterations=1)
                    # applying erode
                    erosion = cv2.erode(dilation, kernel, iterations=1)
                    kernel = np.ones((4, 1), np.uint8)
                    dilation = cv2.dilate(erosion, kernel, iterations=1)
                    # converting back to RGB format to maintain consistence of image shape
                    rgb_data = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)

                image = np.array(rgb_data) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])
                prediction = model.predict(image)
                output_file.write(x + ", " + decode(captcha_symbols, prediction) + "\n")


if __name__ == '__main__':
    main()
