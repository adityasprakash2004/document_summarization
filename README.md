 # Document Summarization Tool

A Python service for document summarization using the BART model and Hugging Face Transformers. Includes:

* A script to fine-tune the model on custom data (`train.py`).
* A Flask API server for real-time summarization (`app.py`).

## Requirements

* Python 3.8 or higher
* pip
* (Optional but Recommended) GPU with CUDA support



## Installation

1. Clone the repository

2. Create and activate a virtual environment:


python -m venv venv
# Windows PowerShell: 
.\venv\Scripts\Activate.ps1
# macOS/Linux: 
source venv/bin/activate


3. Install dependencies:

pip install -r requirements.txt


### Training

To fine-tune the summarization model (Highly Recommended but not required):


python train.py --model_name_or_path facebook/bart-base \
--output_dir outputs \
--max_input_length 1024 \
--max_target_length 128 \
--epochs 3 \
--train_batch_size 4 \
--eval_batch_size 4 \
--learning_rate 3e-5


### Running the API

Start the Flask server:

python app.py

The server will be available at the provided localhost link outputted in th terminal
