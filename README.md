# Brain-Tumor-Detection

This project involves building a convolutional neural network (CNN) to detect and classify brain tumors from MRI images. The model is built using TensorFlow and Keras and utilizes data augmentation for better generalization.

# Table of Contents
* Introduction
* Dataset
* Installation
* Usage
* Model Architecture
* Results
* Contributing
* Acknowledgements

# Introduction
Brain tumors are abnormal growths in the brain that can be life-threatening if not detected and treated early. This project aims to develop a deep learning model to classify brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor.

# Dataset
The dataset used for this project is the Brain Tumor MRI Dataset from Kaggle, which includes images categorized into four types of brain tumors:

* Glioma
* Meningioma
* No tumor
* Pituitary

# Installation
To run this project, you will need to have Python and the following libraries installed:

* numpy
* pandas
* seaborn
* matplotlib
* tensorflow

 You can install the required libraries using pip:
        pip install numpy pandas seaborn matplotlib tensorflow

# Usage
1. Clone the repository:
        git clone https://github.com/your-username/brain-tumor-detection.git

2. Navigate to the project directory:
        cd brain-tumor-detection
   
3. Set the path to the dataset:
        Update the dataset_path variable in the script with the path to your dataset.

4. Run the script:
        python brain_tumor_detection.py

# Model Architecture
The model architecture includes several convolutional layers followed by max-pooling layers. The final layers are fully connected, with dropout regularization to prevent overfitting. The model summary is as follows:

* Conv2D and MaxPooling2D layers
* Flatten layer
* Dense layers with dropout
* Output layer with softmax activation

# Results
The model is evaluated on the test dataset, and the performance metrics include accuracy, precision, recall, and F1-score. The results are visualized using plots for training/validation accuracy and loss, as well as a confusion matrix.

# Contributing
Contributions are welcome! If you have any improvements or suggestions, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch: git checkout -b feature-branch-name.
3. Make your changes and commit them: git commit -m 'Add new feature'.
4. Push to the branch: git push origin feature-branch-name.
5. Open a pull request.

# Acknowledgements

* Kaggle for the Brain Tumor MRI Dataset
* TensorFlow and Keras for the deep learning framework
* The open-source community for their contributions
  
