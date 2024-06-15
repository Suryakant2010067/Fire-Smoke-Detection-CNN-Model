# Fire-Smoke-Detection-CNN-Model

This project aims to develop a model that can automatically detect fire and smoke in images and using CNN Model.
this Project is published IEEE.
Motivation
Fire and smoke are common hazards that can cause severe damage to life and property. Early detection of fire and smoke can help in preventing or mitigating the consequences of such disasters. However, traditional fire and smoke detection methods, such as a mail alart and alarms, may not be effective in some scenarios, such as outdoor environments, large areas, or complex scenes. Therefore, there is a need for a system that can use the power of  deep learning to analyze visual data and identify fire and smoke events in real time.
Objectives
The main objectives of this project are:
To collect and annotate a large-scale dataset of fire and smoke images from various sources and scenarios.
To design and implement a deep neural network model that can accurately classify fire and smoke in random images.
To evaluate the performance of the model on various metrics, such as Precision, Recall, Mean Average Precision, F1 score, Confidence etc. and curves formed using these parameters
To provide real-time fire and smoke detection from Image feeds or user uploads.
Methodology
The methodology of this project consists of the following steps:

1️⃣  Data collection:
    We collected fire and smoke images and videos from IndianAI.gov and Roboflow public datasets.

2️⃣Data Augmentation:
We manually labelled the collected data (fire, smoke and firefighters) using bounding boxes (for localization) with the help of tools such as Makesense.ai to facilitate the annotation process.

3️⃣  Data preprocessing:
We performed data augmentation techniques, such as cropping, resizing, flipping, rotating, etc., to increase the diversity and robustness of the data. We also normalized the data to have zero mean and unit variance.
  

4️⃣  Model development:
We designed and implemented a deep neural network model that can perform fire and smoke detection in images. We used frameworks such as TensorFlow or PyTorch to build the model. We explored different architectures, such as convolutional neural networks (CNNs), or attention mechanisms, to achieve the best results.

5️⃣  Model training:
We trained the model on the preprocessed data using a suitable loss function, such as binary cross-entropy or mean squared error. We will use optimization algorithms, such as stochastic gradient descent (SGD) or Adam, to update the model parameters. We will also use regularization techniques, such as dropout or batch normalization, to prevent overfitting.

6️⃣  Model Validation And Testing:
We evaluated the model's performance on a separate test set using various metrics, such as accuracy, precision, recall, F1-score, etc. We also compared the model with other various models to demonstrate its effectiveness.





Dataset Link:- https://www.kaggle.com/kutaykutlu/forest-fire
