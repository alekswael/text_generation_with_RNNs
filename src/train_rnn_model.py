# data processing tools
import os 
import pandas as pd
# argument parser
import argparse
# to save tokenizer
from joblib import dump
# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.preprocessing.text import Tokenizer
# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# append the current working directory to the path
import sys
sys.path.append(os.path.join(os.getcwd()))
# import helper functions
import utils.helper_functions as hf

def input_parser(): # This is the function that parses the input arguments when run from the terminal.
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) # This is the argument parser. I add the arguments below.
    ap.add_argument("--epochs",
                    help="Amount of epochs to train the model for.",
                    type = int, default=1) # CHANGE THIS TO 100
    ap.add_argument("--batch_size",
                    help="Batch size to use when training the model.",
                    type = int, default=128)
    args = ap.parse_args() # Parse the args
    return args

# set the data directory
def get_data_dir():
    data_dir = os.path.join(os.getcwd(), "data")
    
    return data_dir

# load the comments from the csv files
def load_comments(data_dir):
    all_comments = []
    for filename in os.listdir(data_dir):
        if 'Comments' in filename:
            comment_df = pd.read_csv(os.path.join(data_dir, filename))
            all_comments.extend(list(comment_df["commentBody"].values))
            
    return all_comments

# Removing comments with unkown content
def preprocess_comments(all_comments):
    all_comments = all_comments[:1000] # FOR TESTING ONLY, REMOVE THIS LINE
    all_comments = [c for c in all_comments if c != "Unknown"]

    # Define the corpus and preprocess the data
    corpus = [hf.clean_text(x) for x in all_comments]
    
    return corpus

def tokenize_corpus(corpus):
    # Load tokenizer and fit it on the corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # Define the input sequences
    inp_sequences = hf.get_sequence_of_tokens(tokenizer, corpus)

    # Generate the padded sequences
    predictors, label, max_sequence_len = hf.generate_padded_sequences(inp_sequences, total_words)
    
    return predictors, label, max_sequence_len, total_words, tokenizer

def train_model(epochs_arg, batch_size_arg, predictors, label, max_sequence_len, total_words):
    # Intialize the model
    model = hf.create_model(max_sequence_len, total_words)
    model.summary()

    # Train the model
    history = model.fit(predictors, 
                        label, 
                        epochs=int(epochs_arg),
                        batch_size=int(batch_size_arg), 
                        verbose=1)
    
    return model, history

def save_models(model, tokenizer):
    # Save the model
    model.save(os.path.join(os.getcwd(), "models", "rnn_model"))

    # Save the tokenizer with joblib
    dump(tokenizer, os.path.join(os.getcwd(), "models", "tokenizer.joblib"))

def main(): # This is the main function that runs the program.
    args = input_parser() # Parse the input arguments.
    data_dir = get_data_dir() # Get the data directory.
    all_comments = load_comments(data_dir) # Load the comments.
    corpus = preprocess_comments(all_comments) # Preprocess the comments.
    predictors, label, max_sequence_len, total_words, tokenizer = tokenize_corpus(corpus) # Tokenize the corpus.
    model, history = train_model(args.epochs, args.batch_size, predictors, label, max_sequence_len, total_words) # Train the model.
    save_models(model, tokenizer) # Save the model and tokenizer.
    
if __name__ == "__main__":
    main()