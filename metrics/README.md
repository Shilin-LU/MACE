# GIPHY Celebrity Detector Installation Guide

This guide offers a step-by-step procedure for installing and configuring the GIPHY Celebrity Detector. We recommend initially consulting the [official installation instructions](https://github.com/Giphy/celeb-detection-oss/tree/master/examples). Should you encounter any difficulties or achieve less than satisfactory results, please consider following the instructions provided herein to try a successful installation.

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
- Comment out the numpy on line 8 `numpy==1.15.1` and `torch==0.4.1` at the end of the file.

```
pip install -r requirements_gpu.txt
pip install -e .
cd examples
cp .env.example .env
pip install imageio==2.4.1 pandas
pip install --upgrade scikit-image
```

4. Download the `resources.tar.gz` file from [this OneDrive folder](https://entuedu-my.sharepoint.com/:u:/g/personal/shilin002_e_ntu_edu_sg/EayVzaUyyCZKnbMPZDtVUYABfmiVflXiYWPNrNy2_o2MFQ?e=BpFa7m) and replace the `resources` folder inside `celeb-detection-oss/examples` with the extracted one.

 <!-- ```
 wget https://s3.amazonaws.com/giphy-public/models/celeb-detection/resources.tar.gz
 ``` -->

5. Modify the face detection network configuration:
 - Open `celeb-detection-oss/model_training/preprocessors/face_detection/network.py`
 - On line 88, modify to include `allow_pickle=True`, i.e.,
 
   ```
   data_dict = np.load(data_path, encoding='latin1', allow_pickle=True).item()
   ```

6. Evaluate the accuracy of GCD on a folder of generated images. When utilizing this script for detection, please ensure that the content within the input folder consists solely of images, without the need to navigate into subdirectories. This precaution helps prevent errors during the process.

```
conda activate GCD
CUDA_VISIBLE_DEVICES=0 python metrics/evaluate_by_GCD.py --image_folder 'path/to/generated/image/folder'
```