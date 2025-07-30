# BrainTumor_Deep_learning
Brain Tumor Detection using Convolutional Neural Networks (CNN)
This project implements a Convolutional Neural Network (CNN) for the classification of brain MRI images into two categories: "Brain Tumor" (Cancer) and "Healthy" (Not Cancer). The notebook covers the entire machine learning pipeline, from data acquisition and preprocessing to model building, training, evaluation, and prediction on new images.

Dataset
The dataset used in this project consists of brain MRI images, categorized into "Brain Tumor" and "Healthy". The notebook includes steps to download and extract this dataset from a Dropbox link.

Dataset Structure (after extraction):

Brain Tumor Data Set/
├── Brain Tumor/  # Contains MRI images with brain tumors
└── Healthy/      # Contains MRI images of healthy brains

Key Features & Methodologies
Data Acquisition: Downloads the dataset from a provided Dropbox URL using wget.

Data Extraction: Unzips the downloaded dataset.

Data Splitting: Implements a custom function (datafolder) to split the dataset into training, validation, and testing sets with specified ratios (e.g., 70% train, 15% validation, 15% test). This involves creating new directories and copying/moving images.

Image Preprocessing & Augmentation:

Uses Keras' ImageDataGenerator for:

Rescaling pixel values (1./255).

Applying data augmentation techniques (shear, zoom, horizontal flip) to the training set to prevent overfitting.

Creating batches of image data for model training.

Convolutional Neural Network (CNN) Model:

Builds a sequential CNN model using Keras, incorporating:

Conv2D layers for feature extraction.

MaxPool2D layers for dimensionality reduction.

Dropout layers for regularization.

Flatten and Dense layers for classification.

The model is compiled with adam optimizer and binary_crossentropy loss, suitable for binary classification.

Model Training:

Trains the CNN model using the prepared image data.

Implements EarlyStopping to prevent overfitting by monitoring validation accuracy.

Uses ModelCheckpoint to save the best-performing model during training.

Model Evaluation:

Evaluates the trained model's accuracy on the test dataset.

Visualizes the training history (loss vs. validation loss) to assess model performance and identify overfitting/underfitting trends.

Prediction:

Demonstrates how to load a saved model.

Performs a prediction on a single image to classify whether it shows a brain tumor or not.

Technologies & Libraries Used
Python

NumPy: For numerical operations.

Matplotlib: For data visualization.

OS Module: For interacting with the operating system (directory creation, file operations).

Shutil: For high-level file operations (copying files).

Keras (TensorFlow backend): For building, training, and evaluating the deep learning model.

Conv2D, MaxPool2D, Dropout, Flatten, Dense, Sequential

ImageDataGenerator

EarlyStopping, ModelCheckpoint

load_model

Glob: For pathname pattern matching.

⚙️ How to Run the Notebook
Clone the Repository:

git clone <your-repo-url>/BrainTumorDetection.git
cd BrainTumorDetection

Open in Google Colab:

This notebook is designed to be run in Google Colab due to its reliance on !wget for data download and GPU acceleration for model training.

Upload the BrainTumor.ipynb file to Google Colab.

Run All Cells: Execute all cells in the notebook sequentially.

Go to Cell > Run All in Google Colab.

Observe Output:

The notebook will download and unzip the dataset.

It will print the model summary, training progress (epochs, accuracy, loss), and evaluation results.

Plots for training/validation loss will be displayed.

A prediction for a sample image will be printed to the console.

📈 Example Results (from notebook output)
Found 3219 images belonging to 2 classes.
Found 689 images belonging to 2 classes.
Found 15 images belonging to 2 classes.

Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_12 (Conv2D)          (None, 222, 222, 16)      448       
 conv2d_13 (Conv2D)          (None, 220, 220, 36)      5,220     
 max_pooling2d_9 (MaxPooling2D) (None, 110, 110, 36)      0         
 conv2d_14 (Conv2D)          (None, 108, 108, 64)      20,800    
 max_pooling2d_10 (MaxPooling2D) (None, 54, 54, 64)        0         
 conv2d_15 (Conv2D)          (None, 52, 52, 128)       73,856    
 max_pooling2d_11 (MaxPooling2D) (None, 26, 26, 128)       0         
 dropout_6 (Dropout)         (None, 26, 26, 128)       0         
 flatten_3 (Flatten)         (None, 86528)             0         
 dense_6 (Dense)             (None, 64)                5,537,856 
 dropout_7 (Dropout)         (None, 64)                0         
 dense_7 (Dense)             (None, 1)                 65        
=================================================================
Total params: 5,638,245 (21.51 MB)
Trainable params: 5,638,245 (21.51 MB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________

Epoch 1/30
... (training output) ...
Epoch 6/30
8/8 ━━━━━━━━━━━━━━━━━━━━ 4s 491ms/step - accuracy: 0.7222 - loss: 0.5871 - val_accuracy: 0.6000 - val_loss: 0.8064
Epoch 6: early stopping

The accuracy of the model is 62.40928769111633%
The person does not have brain tumor
