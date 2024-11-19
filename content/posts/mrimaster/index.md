---
author: ["Lixian Wang"]
title: Brain Cancer Diagnosis by Artificial Intelligence
tags: ["programming","ai","medical image"]
date: "2024-10-15"
ShowToc: true
TocOpen: true
math: true
---

## Introduction
### MRI Technologies
Magnetic Resonance Imaging (MRI) is a non-invasive imaging technique widely used in diagnosing various medical conditions, especially when it comes to examining soft tissues like the brain. MRI uses strong magnetic fields and radio waves to create detailed images of the body's internal structures. This level of detail is crucial for identifying abnormalities, such as tumors, and has become a cornerstone in the diagnosis and monitoring of brain cancers.

### Three types of tumors (source: Cleveland Clinic website)

- A **glioma** is a tumor that forms when glial cells grow out of control. Normally, these cells support nerves and help your central nervous system work. Gliomas usually grow in the brain, but can also form in the spinal cord. Gliomas are **malignant** (cancerous), but some can be very slow growing.
- A **meningioma** is a tumor that forms in your meninges, which are the layers of tissue that cover your brain and spinal cord. They’re **usually not cancerous** (benign), but can sometimes be cancerous (malignant). Meningiomas are treatable.
- **Pituitary** adenomas are benign tumors on your pituitary gland. They’re **noncancerous**, but they can interfere with normal pituitary function and cause certain health conditions. Healthcare providers treat pituitary adenomas with surgery, medication, radiation or a combination of these therapies.

## How it started
As I explore different applications of computer vision to improve work and life quality, I’ve noticed its significant growth in the field of medical imaging. Along the way, I discovered several high-quality MRI datasets on Kaggle, along with some inspiring notebooks (links provided at the end). I decided it would be a great opportunity to dive in and build my first end-to-end computer vision project— **MRIMaster**.

The goal of this project is to automatically classify MRI images of patients' brains into four categories: no tumor, glioma tumor, meningioma tumor, and pituitary tumor. This automation will enable doctors to focus on more critical tasks and make MRI services more affordable, benefiting both patients and the healthcare system

## Challenges
The images from the Kaggle dataset are well-labeled and divided into four categories for training and evaluation. However, the challenge lies in the fact that tumors can vary in size, and each MRI image represents only a two-dimensional slice of the tumor and surrounding normal brain tissue. For this task, I chose a **convolutional neural network (CNN)** as the classifier, as it is widely used and excels in image classification problems. 


## Solution
Since this is an ongoing project, I will not only present the current solution but also propose future developments to enhance overall performance and facilitate the model's deployment.

---
### Status of Quo
#### 1. Overview of the End-to-End Project

{{< figure src="images/workflow.png" align=center title="This is the typical workflow including AWS service" >}}

Since AWS EC2 GPU instances are quite expensive, I primarily work on my local machine and use Kaggle/Colab GPU resources to train and fine-tune models. Once the model choice is finalized, I plan to initialize cloud infrastructure reproducibly using Terraform and offload the heavy GPU tasks to an EC2 instance. To present prediction results, I’ve built a simple API service with FastAPI, allowing image uploads and returning predictions with corresponding confidence values. For cloud deployment, AWS SageMaker could be a more affordable solution. 

#### 2. Metrics of training process

{{< figure src="images/metrics1.png" align=center title="Loss and accurancy during training process over epochs for training and evaluation sets" >}}

As observed, overfitting becomes evident after 30 epochs. Let's take a closer look at the model's performance across each category
{{< figure src="images/metrics2.png" align=center title="Confusion matrix for the evaluation set" >}}


More details (precision, recall and f1-score)
|catergory|precision|recall|f1-score|
|:---------:|:---------:|:------:|:--------:|
|glioma|0.88|0.92|0.90|
|meningioma|0.86|0.82|0.84|
|no tumor|0.86|0.85|0.85|
|pituitary|0.95|0.95|0.95|


There is definitely room for improvement in this model to enhance its predictions.

#### 3. How to understand the model prediction
To optimize the model (e.g.,hyperparameter tuning), we first need to understand where the model is performing well or poorly, and why. This is where Class Activation Mapping (CAM) becomes very useful, as it helps make convolutional neural networks more transparent to humans. Specifically, we use Grad-CAM to visualize the importance of each pixel in an image for any given prediction.

{{< figure src="images/misclassified.png" align=center title="The model makes such obvious mistakes with very high confidence (over 0.95)" >}}


Here are a few examples illustrating how poorly the model can perform on specific cases that should be relatively easy to identify as either tumors or healthy brains. Most concerning is the model's high confidence when making these incorrect predictions. We don’t fully understand how the model arrives at such poor decisions, despite maintaining a decent overall success rate. This inconsistency is uncommon for human experts, who generally maintain steady performance and quality of work. To improve, we need to take a closer look at the model’s 'thinking' process before we can truly claim to understand it.


##### Grad-CAMs for each convolutional neural layer

In order to obtain the class-discriminative localization map **Grad-CAM** $L_{Grad-CAM}^c$ of width $u$ and height $v$ for any class $c$, we first calculate the gradient of the score for class $c$, $y^c$, with respect to feature map (heat map) activations $A^k$ of a convolutional layer, i.e. $\frac{\partial y^c}{\partial A^k}$. These gradients flowing back are global-average-pooled over the width and height dimensions (indexed by i and j
respectively) to obtain the neuron importance weights $\alpha_k^c$:

$$
\alpha_k^c = \frac{1}{Z}\sum_i{\sum_j{\frac{\partial y^c}{\partial A_{ij}^k}}}
$$
We perform a weighted combination of forward activation maps, and follow it by a ReLU to obtain, 
$$
L_{Grad-CAM}^c = ReLU\left(\sum_k{\alpha_k^cA^k}\right)
$$
From https://arxiv.org/abs/1610.02391

 

The first row of images represents the dot product of the activation maps $A^k$ and averaged gradients $\alpha_k^c$ for each activation map in that layer after applyting ReLU. As we can observe, the size of each activation map decreases in the deeper layer due to **MaxPool2d**.
<img src="assets/heatmap1.png" alt="Description" style="width:600px; height:auto; display:block; margin-left:auto; margin-right:auto;">

The second row show the upsampled activation map. Now we aim to upsample $L_{Grad-CAM}^c$ after each convoluational layer to match the original size. This step is crucial for illustrating the importance of each pixel in the original image for class $c$. To enhance the visual effect, we overlay the original image with the upsampled activation map (heatmap).

<img src="assets/heatmap2.png" alt="Description" style="width:600px; height:auto; display:block; margin-left:auto; margin-right:auto;">

The third row reveals, surprisingly, that the pixels associated with the tumor (white area at the top middle) are contributing the most to the incorrect prediction of 'no tumor'. 

<img src="assets/heatmap3_misclassified.png" alt="Description" style="width:600px; height:auto; display:block; margin-left:auto; margin-right:auto;">

#### 4. How to leverage existing model (Transfer learning)
With a few lines of code, we can download the vgg16 model and easily adapt it to our classification task.
```python
from torchvision import models
vgg16 = models.vgg16(pretrained=True,dropout=0.2)
# Freeze the parameters in the feature extraction layers
for param in vgg16.parameters():
    param.requires_grad = False
# Modify the classifier to match 4 output classes
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 4)
```
This pre-trained model has undergone 60 epochs of training, during which most parameters were frozen, with only 16,388 out of 134 million parameters being trainable.  
<img src="assets/metrics_comparsion.png" alt="Description" style="width:400px; height:auto; display:block; margin-left:auto; margin-right:auto;">

From the comparison of training process, we can highlight a few key points:
1. Overfitting is effectively suppressed in the VGG16 model.
2. The performance of the VGG16 model saturates early in the training phase but ultimately performs as well as the smaller CNN model after 60 epochs. This indicates that VGG16 requires less training to perform well, which is not surprising given its prior knowledge as a pre-trained model.

### To Do List

- [  ] Add object detection of the tumor (size)
- [  ] set up metrics, optimize the model via a variety of experiments listed in the preceeding section


⬛⬛⬛⬛⬜⬜⬜⬜⬜⬜(40% complete)

## Technical Stack
Development: PyTorch, Numpy, FastAPI, Unicorn, Pillow, Pandas, Matplotlib, Seaborn, Metaflow, openCV, sklearn, Git

Deployment: AWS EC2, AWS S3, Docker, Terraform

## Reference
- Kaggle dataset: https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri
- Kaggle notebook: https://www.kaggle.com/code/pkdarabi/brain-tumor-detection-by-cnn-pytorch
- Grad-CAM: https://arxiv.org/abs/1610.02391