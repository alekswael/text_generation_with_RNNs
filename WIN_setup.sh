# Create a virtual environment
python -m venv rnn_for_text_classification_venv

# Activate the virtual environment
source ./rnn_for_text_classification_venv/Scripts/activate

# Install requirements
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt

# deactivate
deactivate

#rm -rf rnn_for_text_classification_venv