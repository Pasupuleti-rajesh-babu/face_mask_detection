
# Mask Detector using MobileNetV2

This project aims to detect whether a person is wearing a mask or not using deep learning techniques. We utilize the MobileNetV2 architecture pretrained on the ImageNet dataset for feature extraction and then fine-tune it on our custom dataset containing images of people with and without masks.

## Overview

The project consists of the following components:

1. **Dataset Preparation**: The dataset used for training the model should be organized into two folders: `with_mask` and `without_mask`, containing images of people with and without masks, respectively.

2. **Training Script**: The `train_mask_detector.py` script is responsible for loading the dataset, preparing the data, defining the model architecture, training the model, and saving the trained model to disk. It also generates a plot showing the training loss and accuracy over epochs.

3. **Model Evaluation**: After training, the model is evaluated on a separate test set to assess its performance. The evaluation includes metrics such as accuracy, precision, recall, and F1-score.

4. **Inference**: Once trained, the model can be used for inference to predict whether a person in a given image is wearing a mask or not.

## Usage

1. **Dataset Collection**: Gather images of people with and without masks and organize them into separate folders.

2. **Training**: Run the training script `train_mask_detector.py`, passing the path to the dataset using the `--dataset` argument. Optionally, specify the paths for saving the model and plot using the `--model` and `--plot` arguments, respectively.

    ```bash
    python train_mask_detector.py --dataset dataset
    ```

3. **Model Evaluation**: After training, the script will automatically evaluate the model on the test set and display a classification report containing metrics such as accuracy, precision, recall, and F1-score.

4. **Inference**: Use the trained model for inference by loading it using TensorFlow or Keras and passing the image through the model to get predictions.

## Dependencies

- TensorFlow/Keras
- scikit-learn
- imutils
- matplotlib

## Acknowledgments

This project is based on the work by [Adrian Rosebrock](https://www.pyimagesearch.com/), with modifications for educational purposes.
