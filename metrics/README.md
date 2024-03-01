# GIPHY Celebrity Detector Installation Guide (by Zilan Wang)

This guide offers a detailed, step-by-step procedure for installing and configuring the GIPHY Celebrity Detector. We recommend initially consulting the [official installation instructions](https://github.com/Giphy/celeb-detection-oss/tree/master/examples). Should you encounter any difficulties or achieve less than satisfactory results, please consider following the instructions provided herein to guarantee a successful installation.

## Installation Steps

1. Clone the repository:

```
git clone https://github.com/Giphy/celeb-detection-oss.git
cd celeb-detection-oss
conda create -n GCD python=3.6
conda activate GCD
```

2. Modify the `setup.py` file:
- Open `celeb-detection-oss/setup.py`
- Locate line 37 and replace `x.req` with `x.requirement`

3. Update the requirements:
- Open `celeb-detection-oss/requirements_cpu.txt`
- Comment out the numpy version on line 8 and `torch==0.4.1` at the end of the file.
```
pip install -r requirements_gpu.txt
pip install -e .
cd examples
cp .env.example .env
pip install imageio==2.4.1
pip install --upgrade scikit-image
```

4. Download and replace the model files:
 ```
 wget https://s3.amazonaws.com/giphy-public/models/celeb-detection/resources.tar.gz
 ```
 - Extract the `resources.tar.gz` file and replace the `resources` folder inside `celeb-detection-oss/examples` with the extracted one.

5. Modify the face detection network configuration:
 - Open `celeb-detection-oss/model_training/preprocessors/face_detection/network.py`
 - On line 88, modify to include `allow_pickle=True`:
   ```
   data_dict = np.load(data_path, encoding='latin1', allow_pickle=True).item()
   ```