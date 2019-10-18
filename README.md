# CNN Based Captcha Breaker Project

Required dependencies: python-captcha, opencv, python-tensorflow (CPU or GPU)


## Generating captchas

```
python generate.py --width 128 --height 64 --length 8 --symbols symbols.txt --count 3200 --scramble --output-dir train_data
```

This generates 3200 128x64 pixel captchas with 4 symbols per captcha, using the
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

## Training the neural network

```
python train.py --width 128 --height 64 --length 8 --symbols symbols.txt --batch-size 4 --epochs 2 --output-model char8e6bs4 --train-dataset training_data --validate-dataset validation_data
```

Train the neural network for 2 epochs on the data specified. One epoch is one
pass through the full dataset.

The suggested training dataset size for the initial training for captcha length of 4 symbols 
is 20000 images, with a validation dataset size of 4000 images.

## Running the classifier

```
python classify.py  --model-name char8e6bs4 --captcha-dir test_data/ --output output.txt --symbols symbols.txt
```

With `--model-name test` the classifier script will look for a model called
`test.json` with weights `test.h5` in the current directory, and load the model
up.

The classifier runs all the images in `--captcha-dir` through the model, and
saves the file names and the model's guess at captcha contained in the image in
the `--output` file.
