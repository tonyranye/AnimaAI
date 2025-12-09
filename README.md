# 🐾 Smart Animal Classifier  
*A Hierarchical Multi-Model CNN System for Animal Identification*

---

## 1️⃣ Project Overview

Smart Animal Classifier (AnimaAI) is a deep-learning system that identifies animals from images using a **hierarchical pipeline of specialized convolutional neural networks (CNNs)**. Instead of relying on a single classifier to recognize dozens of species and breeds, our model routes an image through multiple networks to produce a more accurate prediction.

The system is capable of:

- 🐶 Classifying **70+ dog breeds**  
- 🦋 Classifying **70+ butterfly species**  
- 🐱 Recognizing **multiple cat breeds**  
- 🐦 Identifying **various bird families**  
- 🐾 Classifying **14 general animal categories**

This makes it suitable for wildlife research tools, educational apps, and automated image-sorting systems.

---

## 2️⃣ Installation

Install required packages:

```bash
pip install torch torchvision pillow matplotlib gradio
```

---

## 3️⃣ Running the Classifier App

Start the Gradio interface:

```bash
python myApp.py
```

This will:

- Load the router model  
- Load all available specialist models  
- Launch an interactive web app  
- Allow users to upload images for real-time predictions  

## 4️⃣ Problem Statement

Image classification is one of the most fundamental tasks in machine learning. Animals are one of the most common real world targets: from wildlife monitoring and ecological research to educational apps and pet-ID tools. However animals are visually diverse and often have many breeds with similar appearances. A single monolithic neural network that tries to classify everything at once can struggle to distinguish each animal due to things like: large label space, imbalance datasets between species, fine-grained distinctions between visually similar classes, etc. Our project, AnimalAI, tackles this by combining a general router CNN with a set of specialized CNNs. Users upload an image and the system: uses a router model to predict a coarse animal category (e.g. Cat, dog, bird, butterfly, etc.) then, it routes the images to specialist models(e.g. Dog breeds, cat breeds, birds vs butterflies, etc.) finally, the network returns a final prediction with probabilities via Streamlit. 

---
## 5️⃣ Datasets

Here are the datasets used:
- [https://www.kaggle.com/datasets/yapwh1208/cats-breed-dataset?resource=download]([url](https://www.kaggle.com/datasets/yapwh1208/cats-breed-dataset?resource=download))

---
## 6️⃣ Training the Models

Training is performed through `train_animals.py` and additional scripts for specialist models.

### Dataset Structure

The main dataset is organized as:

```
Animals/
  cats/
  dogs/
  birds/
  butterfly/
  snake/
  elephant/
  ...
```

Each folder contains images for a single coarse class.

Specialist datasets exist for:

- Cat breeds  
- Dog breeds  
- Butterflies  
- Birds  
- Big cats  
- Snake / Chameleon  
- Birds vs butterflies (ambiguity resolution)

### Training Details

- **Architecture:** ResNet-18 (ImageNet pretrained)  
- **Loss:** CrossEntropy  
- **Optimizer:** Adam (lr = 1e-4)  
- **Batch size:** 32  
- **Epochs:** 5–20 depending on dataset size  
- **Train/Val split:** 80/20  

### Data Preprocessing

All images are:

- Resized to **224×224**
- Normalized using ImageNet mean and std
- Randomly flipped horizontally during training

### Model Saving

Each trained model is saved as:

```python
{
  "model_state_dict": ...,
  "class_names": ...,
  "label_to_idx": ...
}
```

Models are stored in the `Models/` directory.

---

## 7️⃣ Deployment & Real-World Integration

This project demonstrates how a trained neural network can be integrated into an actual software application.  
Using Gradio, we built a functional web-based classifier that:

- Accepts user-uploaded images  
- Feeds them through the router → specialist pipeline  
- Outputs the predicted category or breed with probabilities  

This shows how machine learning systems can move from development into real, accessible tools.

---

## ⭐ Credits

Developed for:

**CSCI 4050U – Machine Learning**  
Ontario Tech University  

Team Members:  
-  Christopher Kiige
-  Tony Akinniranye
-  Lucas Fenkam
-  Jedidiah Dennis
