# Mask Detection using MobileNetV2 and Bing Image API Image Downloader

This project combines mask detection using MobileNetV2 with the ability to download images from the Bing Image Search API. The mask detection model detects whether a person is wearing a mask or not using deep learning techniques, while the Bing Image API Image Downloader script allows you to download images based on a search query from the Bing Image Search API.

## Overview

The project consists of the following components:

1. **Dataset Preparation**: The dataset used for training the mask detection model can be augmented by downloading additional images using the Bing Image API Image Downloader script. This helps in increasing the diversity of images in the dataset.

2. **Training Script**: The `train_mask_detector.py` script trains the mask detection model using the MobileNetV2 architecture and a custom dataset containing images of people with and without masks.

3. **Mask Detection in Images**: The `detect_mask_image.py` script detects whether a person in an image is wearing a mask or not. It utilizes a pre-trained face detector model to locate faces in the image and a trained face mask detector model to classify whether each detected face is wearing a mask or not.

4. **Bing Image API Image Downloader**: The `bing_image_downloader.py` script downloads images from the Bing Image Search API based on a provided search query. It allows you to augment your dataset by downloading additional images related to the mask detection task.

## Usage

### Training Script

1. **Dataset Collection**: Gather images of people with and without masks and organize them into separate folders.

2. **Training**: Run the training script `train_mask_detector.py`, passing the path to the dataset using the `--dataset` argument. Optionally, specify the paths for saving the model and plot using the `--model` and `--plot` arguments, respectively.

    ```bash
    python train_mask_detector.py --dataset dataset
    ```

### Mask Detection in Images

1. **Image Detection**: To detect masks in an image, run the script `detect_mask_image.py`, passing the path to the input image using the `-i` or `--image` argument. Optionally, specify the paths for the face detector model, the trained face mask detector model, and the minimum confidence for face detections using the `-f`, `-m`, and `-c` arguments, respectively.

    ```bash
    python detect_mask_image.py --image examples/example_01.png
    ```

### Bing Image API Image Downloader

1. **Image Download**: To download images from the Bing Image Search API, run the script `bing_image_downloader.py`, passing the search query using the `-q` or `--query` argument and the output directory using the `-o` or `--output` argument.

    ```bash
    python bing_image_downloader.py --query "dogs" --output dataset/dogs
    ```

### Arguments

- For `train_mask_detector.py`:
    - `-d`, `--dataset`: Path to the input dataset.
    - `-p`, `--plot`: Path to output loss/accuracy plot. Default is `plot.png`.
    - `-m`, `--model`: Path to output face mask detector model. Default is `mask_detector.model`.

- For `detect_mask_image.py`:
    - `-i`, `--image`: Path to the input image.
    - `-f`, `--face`: Path to the face detector model directory. Default is `face_detector`.
    - `-m`, `--model`: Path to the trained face mask detector model. Default is `mask_detector.model`.
    - `-c`, `--confidence`: Minimum probability to filter weak detections. Default is `0.5`.

- For `bing_image_downloader.py`:
    - `-q`, `--query`: Search query to search the Bing Image API for.
    - `-o`, `--output`: Path to the output directory of images.

## Dependencies

- TensorFlow/Keras
- scikit-learn
- imutils
- matplotlib
- OpenCV (cv2)
- NumPy
- requests

## Authentication

You need to obtain an API key from Microsoft Cognitive Services to use the Bing Image Search API. Replace the `API_KEY` variable in the `bing_image_downloader.py` script with your own API key.

## Acknowledgments

This project is based on the work by [Adrian Rosebrock](https://www.pyimagesearch.com/), with modifications for educational purposes.

