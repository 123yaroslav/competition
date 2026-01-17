
Predictorium
leaderboard
submit
discord

docs/
quick start
data
submission guide
rules
timeline
prizes
faq
get help

switch
competition
ne_onnn
3112/3420

out
How to make a submission
This page covers the technical requirements for your submission, including the code format, how to package your files, and the resource limits.

What to submit
This is a code competition. You'll submit a .zip file containing all the code and artifacts needed to generate predictions.

The key requirements are:

The zip file must contain a solution.py file at its root.
Your solution.py must define a class named PredictionModel.
This class must have a predict(self, data_point) method.
The PredictionModel class
Here’s the required structure for your PredictionModel class:


import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        # Initialize your model, load weights, etc.
        pass

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # This is where your prediction logic goes.
        if not data_point.need_prediction:
            return None

        # When a prediction is needed, return a numpy array of length 2 (for t0 and t1).
        # Replace this with your model's actual output.
        prediction = np.zeros(2)
        return prediction
Your predict method will receive a DataPoint object with the current market state. Your code should return None if need_prediction is False, and a NumPy array of shape (2,) otherwise.

The DataPoint object comes from the provided utils.py file and has the following attributes:

seq_ix: int: The ID for the current sequence.
step_in_seq: int: The step number within the sequence.
need_prediction: bool: Whether a prediction is required for this point.
state: np.ndarray: The current market state vector (containing p0..p9, v0..v9, dp0..1, dv0..1).
NOTE
Remember to handle the model's internal state. When you encounter a new sequence (a new seq_ix), you must reset any recurrent state.
Including other files
Most solutions will need more than just solution.py. You can include other files in your zip archive, such as:

Model weights (e.g., .pt, .h5, .onnx files).
Helper Python modules (.py files).
Configuration files (e.g., .json, .yaml).
Small data files.
Just make sure solution.py is at the root of the archive.

How to package your solution
You need to package all your files into a single .zip archive.

macOS/Linux
Windows
Open a terminal.
cd into the directory that contains your solution files.
Run the following command:

zip -r ../submission.zip .  
This creates submission.zip in the parent directory, containing everything from your current folder.

Resource and time limits
Your submitted code will run in an isolated environment with the following constraints:

CPU: 1 vCPU core
No GPU
RAM: 16 GB
Storage: Local SSD
Time limit: Your code must finish generating predictions for the entire test set in 60 minutes or less.
No internet access: The execution environment is offline.
How submissions are scored
Docker
When you submit a solution, a scoring Docker container is deployed. The intricacies of the scoring process is not important here, but some details might be useful for you.

The container image is based on python:3.11-slim-bookworm
The environment variables used by some ML libraries are configured to prevent them from attempting to access the network
Below is the part of Dockerfile. You might want to use it locally for debug purposes.


FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
 && python -m pip install orbax-checkpoint \
 && python -m pip check && pip cache purge

# Keep heavy libs strictly offline at runtime 
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_DISABLED=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false \
# Redirect all caches into /app
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    NUMBA_CACHE_DIR=/app/.cache/numba

### R̴̨̋E̴̟͝S̸̪̚T̸̢͘ ̶̜̈́I̴͖͗S̷̢͗ ̴̘̂C̶̕͜E̶̋͜N̵̼̓S̴̙͠O̴̘͐R̶̼͑Ě̵͕Ḓ̵̋ ###
Libs and packages
You may notice a requirements.txt is mentioned in Dockerfile above. We tried to install some reasonable set of popular libs often used in ML.

NOTE
If you need any package added to the scorer docker image — please drop us a line in Discord or email: get help
Here's the full requirements.txt:
PREV
←
data
NEXT
rules
→