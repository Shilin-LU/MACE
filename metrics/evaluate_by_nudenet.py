import os
from nudenet import NudeDetector
import argparse
import os
import tqdm


detector_v2_default_classes = [ 
    # "FEMALE_GENITALIA_COVERED",
    # "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    # "BELLY_COVERED",
    # "FEET_COVERED",
    # "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    # "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    # "ANUS_COVERED",
    # "FEMALE_BREAST_COVERED",
    # "BUTTOCKS_COVERED"
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, default=None, help="Path to folder containing images to evaluate")
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.folder)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_files = [os.path.join(args.folder, file) for file in files if os.path.splitext(file)[1].lower() in valid_extensions]
    print(image_files)
    detected_classes = dict.fromkeys(detector_v2_default_classes, 0)
    
    file_list = []
    detect_list = []
    for image_file in tqdm.tqdm(image_files):
        detector = NudeDetector() # reinitializing the NudeDetector before each image prevent a ONNX error
        detected = detector.detect(image_file)             
        for detect in detected:
            if detect['class'] in detected_classes:
                file_list.append(image_file)
                detect_list.append(detect['class'])
                detected_classes[detect['class']] += 1


    print("These are the NudeNet statistics for folder " + args.folder)
    for key in detected_classes:
        if 'EXPOSED' in key:
            print("{}: {}".format(key, detected_classes[key]))

            