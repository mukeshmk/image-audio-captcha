# image-audio-captcha
generating and solving 8 character image captchas

## Generating Audio Set
```
python generate-audio-captcha.py  --audio-dict True --symbols symbols.txt --output-dir audio-symbols
```

This generates audio file for each character / digit in the symbol set provided and outputs it into 
the target output directory.

## Generating captchas

```
python generate-audio-captcha.py --length 8 --symbols symbols.txt --count 3200 --scramble --output-dir test
```

This generates 3200 audio captchas with 8 characters per captcha, using the
set of symbols in the `symbols.txt` file. The captchas are stored in the folder
`test`, which is created if it doesn't exist. The names of the captcha images
are scrambled.

Without the `--scramble` option, the name of the image is the captcha text.

To train and validate a neural network, we need two sets of data: a big
training set, and a smaller validation set. The network is trained on the
training set, and tested on the validation set, so it is very important that
there are no images that are in both sets.

To generate the training data, the "ground truth" classification for each
training example image must be known. This means that for training, the names
of the captchas *cannot* be scrambled, because otherwise the training process
has no way to check if the answer from the CNN for some captcha is right or
wrong! Make sure not to use the `--scramble` option when generating the
training or validation datasets.
