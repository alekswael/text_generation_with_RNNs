<br />
  <h1 align="center">Language modelling and text generation using RNNs</h1> 
  <h3 align="center">
  Author: Aleksander Moeslund Wael <br>
  </h3>
</p>

## About the project
This repo contains a collection of scripts for builiding a text generator by training a recurrent neural network on a large text dataset. 

### Data
For this project, the [New York Times Comments](https://www.kaggle.com/datasets/aashita/nyt-comments) dataset was used to train the model. The dataset consist of 2+ million comments made to New York Times arcticle sections, collected in various time windows. The data should be downloaded from [here](https://www.kaggle.com/datasets/aashita/nyt-comments) and stored in the `data` folder as shown in the `repository structure` paragraph further down this readme. 

### Model
The text generator is created by training a recurrent neural network model on the comments. The model is created in the `TensorFlow` framework, and is a sequential model with an input embedding layer, a LSTM (Long Short-Term Memory) hidden layer and an output layer.

### Pipeline
The code pipeline consists of a training script, `train_rnn_model.py`, and a text generation script, `generate_text.py`.

Furthermore, a series of helper functions are used in various steps of the pipeline. These functions are located in the `helper_functions.py` script located in the `utils` folder. These functions were created by Ross, but have been slightly modified to fit the pipeline.

The `train_rnn_model.py` script follows these steps:
1. Import dependencies
2. Load data
3. Preprocess data
4. Tokenize data
5. Initialize and train RNN model
6. Save tokenizer and model to `models` folder

The `generate_text.py` script follows these steps:
1. Load tokenizer and model
2. Generate text based on input prompt

## Requirements

The code is tested on Python 3.11.2. Futhermore, if your OS is not UNIX-based, a bash-compatible terminal is required for running shell scripts (such as Git for Windows).

## Usage

The repo was setup to work with Windows (the WIN_ files), MacOS and Linux (the MACL_ files).

### 1. Clone repository to desired directory

```bash
git clone https://github.com/alekswael/assignment-3---rnns-for-text-generation
cd assignment-3---rnns-for-text-generation
```
### 2. Run setup script 
**NOTE:** Depending on your OS, run either `WIN_setup.sh` or `MACL_setup.sh`.

The setup script does the following:
1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
5. Deactivates the virtual environment

```bash
bash WIN_setup.sh
```

### 3. Run pipeline
**NOTE:** Depending on your OS, run either `WIN_run.sh` or `MACL_run.sh`.

1. Run the `*train_model.sh` script.
    
    The script does the following:
    1. Activates the virtual environment
    2. Runs `train_rnn_model.py` located in the `src` folder
    3. Deactivates the virtual environment

```bash
bash WIN_train_model.sh
```

2. Run the `*generate_text.sh` script.
    
    The script does the following:
    1. Activates the virtual environment
    2. Runs `generate_text.py` located in the `src` folder
    3. Deactivates the virtual environment

```bash
bash WIN_generate_text.sh
```

## Generating text
Generating text through the bash script will default to using "Hi, how are you?" as the prompt. However, if you wish to generate text with a different prompt, this can be specified as an argument when running the `generate_text.py` script.

Further model parameters can be set through the ``argparse`` module. However, this requires running the Python script seperately OR altering the `run*.sh` file to include the arguments. The Python script is located in the `src` folder. Make sure to activate the environment before running the Python script.

```
# Activate venv
source ./rnn_for_text_classification_venv/bin/activate # MacOS and Linux
source ./rnn_for_text_classification_venv/Scripts/activate # Windows
```
```
# Run the script with a different prompt
python3 ./src/generate_text.py -p Do you think Hercules is ripped? # MacOS and Linux
python ./src/generate_text.py -p Do you think Hercules is ripped? # Windows
```
### Specifiable arguments
```
generate_text.py [-h] [-p PROMPT] [-w NUM_WORDS]

options:
  -h, --help            show this help message and exit
  -p PROMPT, --prompt PROMPT
                        The prompt to start the text generation from. (default: Hi, how are you?)
  -w NUM_WORDS, --num_words NUM_WORDS
                        Batch size to use when training the model. (default: 5)
```
```
train_rnn_model.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Amount of epochs to train the model for. (default: 100)
  --batch_size BATCH_SIZE
                        Batch size to use when training the model. (default: 128)
```

## Repository structure
This repository has the following structure:
```
│   .gitignore
│   MACL_generate_text.sh
│   MACL_setup.sh
│   MACL_train_model.sh
│   README.md
│   requirements.txt
│   WIN_generate_text.sh
│   WIN_setup.sh
│   WIN_train_model.sh
│
├───.github
│       .keep
│
├───data
│       .keep
│       ArticlesApril2017.csv
│       ...
│
├───models
│       .keep
│
├───src
│       generate_text.py
│       train_rnn_model.py
│
└───utils
        helper_functions.py
        __init__.py
```

## Example of generated text

```
NOTE: Trained for 1 epoch - this is just an example.

PROMPT:  Hi, how are you?
1/1 [==============================] - 0s 382ms/step
1/1 [==============================] - 0s 29ms/step
1/1 [==============================] - 0s 27ms/step
1/1 [==============================] - 0s 24ms/step
1/1 [==============================] - 0s 26ms/step
GENERATED TEXT:  Hi, How Are You? The The The The The
```
