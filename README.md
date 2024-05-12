# Malaria Detection using Deep Learning

## Introduction:
This project aims to develop a deep learning model for classifying whether a blood smear image is infected with malaria or uninfected. Malaria is a serious global health issue affecting millions of people each year, particularly in tropical and subtropical regions. Rapid and accurate diagnosis is crucial for effective treatment and management of the disease. Leveraging advanced technology and artificial intelligence, this project seeks to provide a solution for automated malaria detection, potentially improving diagnostic accuracy and saving lives worldwide.

## Project Overview:
- Dataset: The dataset used for training and evaluation consists of 27,558 cell images, including equal instances of parasitized and uninfected cells from thin blood smear slide images. The dataset is available through the TensorFlow Datasets library.
- Model Architecture: The deep learning model is built using TensorFlow and Keras. It comprises convolutional neural network (CNN) layers followed by batch normalization and max-pooling layers to extract relevant features from the input images. The final layer utilizes sigmoid activation for binary classification.
- Model Training: The model is trained using the Adam optimizer and binary cross-entropy loss function. The training process involves 30 epochs with a batch size of 1. The model achieves a high accuracy rate on both the training and validation datasets.
- Evaluation: The model is evaluated on a separate test dataset to assess its performance after training. The evaluation metrics include loss and accuracy, which indicate the model's effectiveness in malaria detection.
- Prediction: The trained model predicts new blood smear images. Images are preprocessed, normalized, and resized before being fed into the model. Predictions are made based on the model's output probability, with a threshold of 0.5 for classification.

## Usage:
1. Dataset: The malaria dataset can be obtained from the TensorFlow Datasets library or through the provided citation.
2. Model Training: Use the provided code to train the deep learning model on the dataset. Adjust hyperparameters as needed for optimal performance.
3. Evaluation: Evaluate the trained model using the test dataset to measure its accuracy and loss.
4. Prediction: Utilize the saved model to make predictions on new blood smear images. Preprocess the images and pass them through the model to obtain classification results.

## Requirements:
- TensorFlow
- TensorFlow Datasets
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Contact:
For any inquiries or feedback regarding this project, please feel free to contact Dominic Makana Ocharo at ocharodominic01@gmail.com or via phone at +254746073062.

