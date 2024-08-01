# Dogs vs Cats Classification

This project demonstrates three approaches to classify images of dogs and cats using Convolutional Neural Networks (CNNs): a basic CNN model, a CNN model with data augmentation, and transfer learning using the VGG16 model.

## Dataset

The dataset used is the Dogs vs Cats dataset from Kaggle, containing 25,000 images. Download and extract it from [Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats).

# Dogs vs Cats Classification

## Data Preprocessing
- Split the dataset into training and testing sets.
- Resize images to 256x256 pixels.

## Models

### Normal CNN Model
**Architecture:**
- Multiple convolutional layers with ReLU activation and batch normalization.
- MaxPooling layers.
- Fully connected layers with dropout and batch normalization.
- Output layer with sigmoid activation for binary classification.

**Compilation:**
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Metrics: Accuracy

**Training:**
- Train for 10 epochs.

### CNN with Data Augmentation
**Data Augmentation:**
- Apply rescaling, shear range, zoom range, and horizontal flip.

**Architecture and Compilation:**
- Same as the Normal CNN Model.

**Training:**
- Train with early stopping to monitor validation loss.

### Transfer Learning with VGG16
**Architecture:**
- Use VGG16 pretrained on ImageNet, excluding the top layer.
- Trainable layers only in the last block of VGG16.
- Add fully connected layers with dropout and batch normalization.
- Output layer with sigmoid activation.

**Compilation:**
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Metrics: Accuracy

**Training:**
- Train with early stopping to monitor validation loss.

## Results
- **Normal CNN Model:** Achieved an accuracy of around 85% on the validation set.
- **Data Augmentation:** Improved the model's performance and achieved an accuracy of around 92%.
- **Transfer Learning:** Achieved an accuracy of around 98% on the validation set with fine-tuning.

## Conclusion
This project demonstrates the effectiveness of data augmentation and transfer learning in improving the performance of deep learning models for image classification tasks. The transfer learning approach with the VGG16 model yielded the best results.

## Feedback and Contributions

Feel free to provide feedback or contribute to this project. If you have suggestions for improvements or additional features, your contributions are welcome!
