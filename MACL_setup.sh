# Create a virtual environment
python3 -m venv rnn_for_text_classification_venv

# Activate the virtual environment
source ./rnn_for_text_classification_venv/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt

# deactivate
deactivate

#rm -rf rnn_for_text_classification_venv