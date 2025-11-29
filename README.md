# 🐾 Smart Animal Classifier  
*A Multi-Model CNN System for Animal Identification*

## 1️⃣ Overview
Smart Animal Classifier is a deep-learning project that allows users to upload an image of an animal and receive an instant, AI-powered prediction. The system uses a pipeline of **multiple specialized convolutional neural networks (CNNs)** to classify animals at both a **general** and **fine-grained** level.

The model is capable of:

- 🐶 **Classifying 70+ dog breeds**
- 🦋 **Classifying 70+ butterfly species**
- 🐱 **Recognizing multiple cat breeds**
- 🐦 **Identifying various bird families**
- 🐾 **Classifying 14 general animal categories**

This makes it suitable for wildlife apps, educational tools, veterinary tech, and large-scale image-sorting systems.

---

## ✨ Key Features
- **User-uploaded image recognition**  
  Upload any animal image; the system automatically preprocesses it.

- **Multi-model pipeline**  
  A general classifier routes the image to the appropriate fine-grained model.

- **High-resolution CNN inference**  
  Uses center-crop + aspect-ratio-preserving resizing for consistent results.

- **Specialized models for specific groups**
  - 70 dog breeds  
  - 70 butterfly classes  
  - Cat breed classifier  
  - Bird classifier  

- **Real-time predictions**  
  Designed for both desktop and mobile integration.

---

## 2️⃣ Prerequisites

Before running the project, make sure the following Python packages are installed:

- torch
- torchvision
- Pillow
- matplotlib


GUI packages
- streamlit
- gradio

Install them using the command below:

```
pip install torch torchvision pillow matplotlib
```

## 3️⃣ Running the Code

After installing the dependencies, you're ready to start training the model.
Run the following command in your terminal:
```
python train_animals.py
```
The script will automatically load your local images, train the model, and save the result.

---

