# data processing tools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# import argparse
import argparse
# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# Add the current working directory to the path
import sys
sys.path.append(os.getcwd())
# import helper functions
import utils.helper_functions as hf
# load tokenizer
from joblib import load

def input_parser(): # This is the function that parses the input arguments when run from the terminal.
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) # This is the argument parser. I add the arguments below.
    ap.add_argument("-p",
                    "--prompt",
                    help="The prompt to start the text generation from.",
                    type = str, default="Hi, how are you?") # CHANGE THIS TO 100
    ap.add_argument("-w",
                    "--num_words",
                    help="Batch size to use when training the model.",
                    type = int, default=5)
    args = ap.parse_args() # Parse the args
    return args

def load_saved_model():
    # load tokenizer
    tokenizer = load(os.path.join(os.getcwd(), "models", "tokenizer.joblib"))
    
    # Define the maximum sequence length
    max_sequence_len = 271 # This value is extracted from the training script

    # Load the saved model
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "models", "rnn_model"))
    
    return tokenizer, max_sequence_len, model

def main():
    args = input_parser() # Parse the input arguments.
    tokenizer, max_sequence_len, model = load_saved_model()
    hf.generate_text(args.prompt, args.num_words, model, max_sequence_len, tokenizer)

if __name__ == "__main__":
    main()