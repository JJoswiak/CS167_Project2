# CS 167 — Project 2  
## Comparing Neural Network Architectures for Image Classification

## Author  
**Joseph Joswiak**

---

## Overview  
The goal of this project was to design and conduct a machine learning experiment, analyze multiple neural network architectures, and interpret their performance on an image classification task.

I selected Option 1: Comparing Neural Networks on Image Data, where I implemented and evaluated three different deep learning models on the bfgmss dataset:
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Fine-tuned AlexNet

The project is structured as a research-style lab report, with code, experiments, visualizations, and written analysis embedded directly in Jupyter notebooks.

---

## Project Goals
- Apply deep learning techniques using PyTorch  
- Compare different neural network architectures  
- Tune hyperparameters  
- Evaluate models using accuracy, loss curves, and confusion matrices  
- Interpret why certain models perform better than others  

---

## File Descriptions

---

### `joswiak-project2.ipynb`
The Kaggle notebook containing:
- Dataset loading and preprocessing  
- Model implementations (MLP, CNN, AlexNet)  
- Training loops and evaluation logic  
- Written markdown sections covering:
  - Problem definition  
  - Experimental design  
  - Parameter tuning decisions  
  - Initial interpretation of results  

This notebook serves as the main project and written report.

---

### `JoswiakNotebook5.ipynb`
A results focused analysis Google Document that expands on the experiments by:
- Visualizing training and validation performance  
- Displaying confusion matrices for each model  
- Comparing performance across architectures  

This notebook displays visuals and compares models to support conclusions.

---

## Models Implemented

Multilayer Perceptron (MLP):
- A baseline fully connected neural network applied to flattened image data.

Convolutional Neural Network (CNN):
- Uses convolutional and pooling layers to leverage spatial structure in images.

AlexNet (Fine-Tuned):
- A pretrained deep convolutional network adapted to the dataset using transfer learning.

---

## Evaluation Methods
- Training and validation accuracy  
- Loss curves  
- Confusion matrices  
- Cross model performance comparison  

---

## Key Takeaways
- Model architecture has a major impact on image classification performance
- The fine-tuned AlexNet model with 20 epochs performed the best 
- Confusion matrices and visualizations reveal insights not captured by accuracy  

---

## Tools & Libraries
- Python  
- PyTorch  
- NumPy  
- Matplotlib  
- Kaggle

---

## How to Run
1. Open the notebooks in Kaggle  
2. Make sure required Python libraries are installed  
3. Run cells from the top to the bottom to show results and graphs

---
