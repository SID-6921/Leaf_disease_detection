# Leaf Disease Classification and Segmentation 🌿📊🖼️


**Leaf Disease Classification and Segmentation** is an advanced deep learning project focusing on identifying, classifying, and segmenting leaf diseases. This project leverages convolutional neural networks and state-of-the-art models like DenseNet201 and DeepLabV3+ to address challenges in plant disease detection effectively.

## 🌟 Features
- 📷 **Classification and Segmentation**: Classifies 67 different leaf diseases and segments diseased areas from leaf images.
- 💡 **Advanced Neural Architectures**: Utilizes **DenseNet201** for classification and **DeepLabV3+ with ResNet101** for segmentation.
- 📊 **High Accuracy**: Achieves 97.03% classification accuracy across 67 classes.
- 🔄 **Data Augmentation**: Implements various augmentation techniques to enhance generalization and model robustness.
- 🖼️ **Visualization Tools**: Provides detailed visualization of predictions, training history, and confusion matrices.

## 📋 Table of Contents
- [Classification](#classification)
  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Model](#model)
  - [Features](#features)
  - [Results](#results)
  - [Confusion Matrix](#confusion-matrix)
- [Segmentation](#segmentation)
  - [Overview](#overview-1)
  - [Dataset](#dataset-1)
  - [Model](#model-1)
  - [Features](#features-1)
  - [Results](#results-1)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🌿 Classification

### Overview
This project demonstrates the complete workflow for training a **convolutional neural network** using the **DenseNet201** architecture to classify **67 classes** of plant diseases from images. It covers parameter settings, data preprocessing, model definition using the Functional API, and evaluation with a Kaggle-hosted notebook.

### Dataset
The dataset is a combination of multiple publicly available datasets from Kaggle, called **Leaf Disease Detection Dataset**. The dataset consists of images of healthy and diseased leaves, organized into training, validation, and test sets to ensure rigorous model evaluation.


### Model
The model is built on **DenseNet201**, trained using **TensorFlow/Keras**. The trained model is available on Kaggle for download.


### Features
- **DenseNet201 Architecture**: For deep feature extraction and effective classification.
- **Data Augmentation**: Techniques like rotation, shifts, and flips to increase the diversity of training samples.
- **Custom Callbacks**: To monitor training and optimize performance.
- **Model Visualization**: Functions to visualize dataset images and model predictions.
- **Training History**: Saves and provides insights into model performance over epochs.

### Results
The trained classification model achieves an impressive accuracy of **97.03%** on the test dataset.

### Confusion Matrix
Provides a detailed confusion matrix for evaluating model performance across all 67 classes.

## 🍃 Segmentation

### Overview
The segmentation aspect of the project utilizes the **DeepLabV3+ model** with a **ResNet101 backbone** for precise segmentation of diseased leaf regions, improving the understanding of how diseases affect plants.

### Dataset
For segmentation, we utilized a specialized **Leaf Disease Segmentation Dataset** that contains annotated images for semantic segmentation tasks. This dataset is used for training, while a broader dataset is used for extended evaluation.


### Model
The segmentation model uses the **DeepLabV3+ with ResNet101** backbone, and is available for use on Kaggle.


### Features
- **Data Loading and Processing**: Efficient handling of image and mask data, resized to 256x256 pixels.
- **Data Augmentation**: Techniques like flips, rotations, and shifts to enhance dataset quality.
- **Custom Convolutional Block**: Incorporates **Atrous Spatial Pyramid Pooling (ASPP)** for capturing multi-scale contextual information.
- **DeepLabV3+ Model Architecture**: Utilizes a ResNet101 backbone for precise segmentation.
- **Model Training and Evaluation**: Uses **binary cross-entropy** loss and **Adam optimizer**, with visualization tools for monitoring training progress.

### Results
- **On Given Dataset**: Demonstrates high accuracy in segmenting various leaf diseases.
- **On Extended Dataset**: Evaluates the robustness of segmentation on a broader range of leaf diseases.

## 🚀 Usage
To get started with the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/leaf-disease-classification-segmentation.git
   ```
2. Install Dependencies:
   ```bash
   pip install numpy pandas tensorflow matplotlib seaborn tqdm pillow tf_explain
3. Run the Notebook: Open the notebook and run each cell sequentially to train the model.
Pretrained Model: If you prefer to use the pretrained models, download them from the provided links and update the notebook accordingly.
🤝 Contributing
We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature-branch).
5. Open a pull request.
Please make sure to follow our contributing guidelines and maintain code quality.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact
For any inquiries, suggestions, or feedback, feel free to reach out to us at snanda@gitam.in


## 🌟 Show Your Support
If you found this project useful, please ⭐ star this repository to help others find it!

