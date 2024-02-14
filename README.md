# RVC CLI Inference
This project provides a simple yet comprehensive wrapper for the cli inference script from the RVC-Project. There are a multitude of WebUIs covering the process of training models and performing inference. However many of them feel overcomplicated for the relatively simple task of performing inference on an audio file.

## Features
- simplify the inference process by using a cli interface
- organize your models with categories
- create a persistent set of parameters using the settings.json file

Consult the [Wiki](https://github.com/Chromfalke/RVC-CLI-Inference/wiki) for more details on these features.

## Requirements
- Developed and tested with Python 3.8.18
- Pip
- Virtualenv or conda environment

## Setup
1. Clone this repository or download as a zip file.
2. Download 'hubert_model.pth' and 'rmvpe.pt' for example from [here](https://huggingface.co/datasets/SayanoAI/RVC-Studio/tree/main) and place them inside the 'rvc' folder
3. Navigate to the project directory and create a virtual environment `conda create -n rvc-cli python=3.8.18`
4. Activate the virtual environment `conda activate rvc-cli`
5. Install the required packages using pip `pip install -r requirements.txt`
6. Run the main.py file. The script will prompt you for any needed parameters. `python main.py`

A detailed walkthrough or the individual steps can be found in the [Wiki](https://github.com/Chromfalke/RVC-CLI-Inference/wiki).

## Limitations
The project was developed and tested on Linux using conda for the cirtual environmant.

At the moment only the inference process itself is supported. That means that the input files should consist of clear and clean speech without music or background noise. If you want to make a song cover for example you'll need to extract the vocals first with another tool.

## Disclaimer
This project is for educational and research purposes only. We do not endorse or promote the use of generative AI for unethical or illegal purposes. We are not responsible for any damages or liabilities arising from the use or misuse of the generated voice overs.

## Credits
This project uses code from the following repository:
- [Retrieval-based Voice Conversion WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) by RVC-Project

## Links
Looking to train your own models? Want something with a GUI? Check out these projects:
- [Retrieval-based Voice Conversion WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) by RVC-Project
- [RVC-Studio](https://github.com/SayanoAI/RVC-Studio) by SayanoAI