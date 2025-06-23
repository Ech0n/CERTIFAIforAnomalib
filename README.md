# CERTIFAI for MVtecAD
fork of CERTIFAI modified to work on mvtec image data.


## Installation
This project was run on wsl+ubuntu22.04.4 and python 3.10.17.  
Make sure CUDA drivers are installed

1. Clone this repository  

2. Create virtual enviroment and activate it
```
python -m venv .venv
source .venv/bin/activate
```

3. Install required libraries
```
pip install -r requirements.txt
```

3. Install torch with cuda compatible with your gpu. For me it was version 12.6:
```
pip install torch==2.7.0+cu126 torchvision==0.18.0+cu126 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```


## Usage

Example usage is shown in file `main.py`

For populations larger than aprox. 500 large swap file will be necessary   

You can use script resize.py to create a resized category directory in the dataset or you can modify the transforms inside mvTek.py to resize the images in the datamodule  

#### class CertifaiMvTekWrapper usage:
1. class CertifaiMvTekWrapper() trains a patchcore classificator on initialization
2. initalize CertifAi using one of those functions:
    - initCertifAi   
    or
    - ~~initCertifAiWithMultipleSamples (this one uses multiple samples to generate starting population)~~
3. Start evolution by calling:
    - fit  
    or  
    - fitWithTrainImageAsSample (This is my expirimental function to test if calculating distance between normal class image and generated one is a better fitness function than distance between starting image and generated one )