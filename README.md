

# 🍅 Tomato Leaf Disease Classification

## Solution Description
The goal of this project is to develop a deep learning model that classifies tomato leaf diseases using image data. The dataset used is from [Kaggle - Tomato Leaf Dataset](https://www.kaggle.com/api/v1/datasets/download/kaustubhb999/tomatoleaf), which consists of various categories of tomato leaf conditions, including bacterial spot, early blight, late blight, and healthy leaves.


## Steps Taken:

### Data Preprocessing:
- Loaded and extracted the dataset from the provided Kaggle API link.
- Separated the dataset into training and validation folders.
- Applied image rescaling to normalize pixel values between [0, 1].
- Augmented data using the `ImageDataGenerator` to improve model generalization.
- Defined target size `(224, 224)` to resize images for model input.
- Split the dataset with 80% used for training and 20% for validation.

---

### Model Selection:
- Implemented a Convolutional Neural Network (CNN) using TensorFlow and Keras.
- Model architecture includes:
    - 3 Convolutional layers with ReLU activation and MaxPooling.
    - Flattened feature maps for Dense layers.
    - Fully connected Dense layer with 128 neurons.
    - Final Dense layer with 10 output neurons (representing 10 classes) and Softmax activation.
- Compiled the model using:
    - Optimizer: `Adam`
    - Loss: `sparse_categorical_crossentropy`
    - Metric: `accuracy`

---

### Model Training & Evaluation:
- Trained the CNN for 20 epochs using a batch size of 32.
- Used the validation set to monitor the model’s performance.
- Achieved a validation accuracy of approximately **{val_accuracy * 100:.2f}%**.

---

## Final Model & Prediction:
- The trained model is capable of predicting the type of tomato leaf disease given an input image.
- Predictions can be made using the `model.predict()` function after loading the test dataset.

---

## Future Improvements:
- Experiment with deeper architectures or pre-trained models such as VGG16, ResNet, or EfficientNet.
- Implement data augmentation techniques such as rotation, zoom, and flipping.
- Fine-tune the model using techniques like dropout to prevent overfitting.
- Incorporate early stopping and learning rate scheduling for optimized training.

---
