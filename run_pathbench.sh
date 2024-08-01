#If virtual environment does not exist, construct one using pip
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

#Set slideflow backends
export SF_SLIDE_BACKEND=cucim
export SF_BACKEND=torch
#Token required to access the gated models
export HF_TOKEN=YOUR_HUGGINGFACE_TOKEN

#Set the config file
CONFIG_FILE=your_config_file.yaml

#Run the program
python3 main.py $CONFIG_FILE