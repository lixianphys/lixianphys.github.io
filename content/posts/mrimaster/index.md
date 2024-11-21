---
author: ["Lixian Wang"]
title: Brain Cancer Diagnosis Assisted by Artificial Intelligence
tags: ["AI","medical imaging","brain tumor"]
categories: ["AI medicine","Computer Vision"]
date: "2024-10-15"
ShowToc: true
TocOpen: true
math: true
draft: false
---

## Background
### MRI Technologies
Magnetic Resonance Imaging (MRI) is a non-invasive imaging technique widely used in diagnosing various medical conditions, especially when it comes to examining soft tissues like the brain. MRI uses strong magnetic fields and radio waves to create detailed images of the body's internal structures. This level of detail is crucial for identifying abnormalities, such as tumors, and has become a cornerstone in the diagnosis and monitoring of brain cancers.

### Three types of brain tumors
{{< figure src="images/brain-tumor.jpg" align=center title="Brain tumors can be benign (not cancerous) or malignant (cancerous). There are over 150 different types of brain tumors. Source: Cleverland Clinic" >}}

There are way more than three types, more authentic information can be found from [Cleveland Clinic Website](https://my.clevelandclinic.org/health/diseases/6149-brain-cancer-brain-tumor), including the below descriptions of brain tumors:

>- A **glioma** is a tumor that forms when glial cells grow out of control. Normally, these cells support nerves and help your central nervous system work. Gliomas usually grow in the brain, but can also form in the spinal cord. Gliomas are **malignant** (cancerous), but some can be very slow growing.
>- A **meningioma** is a tumor that forms in your meninges, which are the layers of tissue that cover your brain and spinal cord. They’re **usually not cancerous** (benign), but can sometimes be cancerous (malignant). Meningiomas are treatable.
>- **Pituitary** adenomas are benign tumors on your pituitary gland. They’re **noncancerous**, but they can interfere with normal pituitary function and cause certain health conditions. Healthcare providers treat pituitary adenomas with surgery, medication, radiation or a combination of these therapies.

---

## Open Source Project - MRIMaster
As a newbie in the field of computer vision, I started to search for and explore a variety of applications that can actually contribute positively to the human beings. During this journey, the significant growth in artificial-intelligence (AI)-assisted medical imaging has attracted my most attention. Thanks to the open-sourced dataset and notebooks on Kaggle, I had the chance to dive in this specialized field and built my first end-to-end computer vision project— **MRIMaster**. Check out the source code on [Github](https://github.com/lixianphys/MRImaster).

The ultimate goal is to build a comprehensive and intelligent assistant to help doctors to manage all kinds of medical images and automate the process of diagnosis, prognosis under the supervision of human doctors. However, this long-term goal must sound too ambitious to me technically and financially, which should be divided into a cluster of less ambitious short- and medium-term goals. Here are a few on the top of my mind:

- Classify MR images of patients' brains into four categories: no tumor, glioma tumor, meningioma tumor, and pituitary tumor.
- Provide pixel-level segmentation capability for quick-labelling of edema, non-enhancing and enhancing tumors of multi-modal MR images (FLAIR, T1w, t1gd and T2w).

Moving towards the ultimate goal by stepping through each small achievement makes me satisfied. No need to mention the positive effect if this project succeeds. 

---

## Current Challenges
- **Quality of Dataset.**
Given that images from most Kaggle dataset are well-labelled and placed in order for training and evaluation, the major challenge lies in the fact that tumors may vary dramatically in size and shape, depending on stages of cancer. As a result, the training dataset is required to cover as many as stages, sizes and shapes of various brain tumors to ensure that training dataset is not severely biased. Secondly, each MR image is only a representation of a two-dimensional slice of tumors. This can be partially solved by using a complete dataset of 3D or 4D images (.nii, .nii.gz), which is not so easy to access. Also, high-dimensional datasets request much more calculation power and memory at the stages of training and inference. Not to mention the incued cost of time and money. Regarding the segmentation task, human-involved labelling can be quite expensive, which explains the rareness of such labelled masks. 

- **Choice of Models.**
For the non-binary classification task, a **convolutional neural network (CNN)** model is usually sufficient to extract different levels of features and widely used in image classification problems. With limited resouces, it is practical to train such a small model from scratch. An alternative approach is to transfer learning from a much larger pre-trained model e.g., VGG16. One of its advantages is the "common sense" of pre-trained model, gained by exposing to hundreds if not thousands of categories of images. Certainly, this common sense will be more than helpful in processing any other images (recognizing edges, background and etc.). 
For the segmentation task, a 3D-Unet model is used to re-generate the labelled mask. However, due to my limited knowledge of most advanced models and difficulties in deploying the best but costly model, we hardly know if these many choices actually make the most sense at all.  

---

## Status Quo
Since it is work in progress, I will keep updating this post to make sure major upgrades will be captured here, ranging from new features, better performance, deployment and infrastructure. So, you know, this is an ever-growing end-to-end project. 

### Work Flow in Progress

{{< figure src="images/workflow.png" align=center title="This is the typical workflow including AWS service" >}}

The original plan is to use as many as AWS infrastructures such as S3, EC2, Gateway and SegaMaker. Deploying AWS with [Terraform](https://www.terraform.io/) is convinient and reproducible. Since I am on my own pocket, the cost managment is crucial to keep this project alive, so keeping the AWS infra alive all the time is not possible at this stage. Apart from AWS deployment, I primarily worked on my local CPU and time to time took advantage of free Kaggle/Colab GPU time. In the future, I probably will try out more solutions offered by Google and other smaller but deployment-oriented service providers. 

What has not been included here is the usage of [MLflow](https://mlflow.org/). Spinning up a local MLflow server is most likely sufficient for training tasks on daily basis. And it is also possible to be seamlessly integrated into a workflow including training at remote machines. In parallel, I am also testing how [metaflow](https://metaflow.org/) can be most beneficial to this project. 

Next, we move to the performance of models at the stages of training and evaluation.

### Non-binary Classification - CNN Model

**Metrics of Training and Evaluation Process**
{{< figure src="images/metrics1.png" align=center title="Loss and accurancy during training process over epochs for training and evaluation sets" >}}

As observed, overfitting becomes evident after 30 epochs. Let's take a closer look at the performance across each category of tumors
{{< figure src="images/metrics2.png" align=center title="Confusion matrix for the evaluation set" >}}

More details (precision, recall and f1-score) is provided in this table
|catergory|precision|recall|f1-score|
|:---------:|:---------:|:------:|:--------:|
|glioma|0.88|0.92|0.90|
|meningioma|0.86|0.82|0.84|
|no tumor|0.86|0.85|0.85|
|pituitary|0.95|0.95|0.95|


There is definitely room for improvement in general, especially for predicting glioma and pituitary tumors.

**The model makes obvious mistakes.**
Here are a few examples illustrating the model is stupid in identifying obvious tumors as healthy brains and vice versus. Most concerning is that the model is quite confident when making wrong predictions. It is hard to understand how the model arrives at such poor decisions, despite maintaining a decent overall success rate. This inconsistency of performance is very different from human experts. 

{{< figure src="images/misclassified.png" align=center title="The model makes such obvious mistakes with very high confidence (over 0.95)" >}}


**How to understand the mistakes.**
The next step is to optimize the model. To do this by fine-tuning hyperparameters, it is necessary to understand where and how the model performs not so well. This is where Class Activation Mapping (CAM) becomes extremely useful and makes the decision by neural networks transparent and readable to humans. Here we use Grad-CAM ([Paper](https://arxiv.org/abs/1610.02391)) to visualize how much each pixel contributes to any given predictions.

**Grad-CAMs for each convolutional neural layer.**
In order to obtain the class-discriminative localization map (**Grad-CAM**) $L_{Grad-CAM}^c$ of width $u$ and height $v$ for any class $c$, we need to first calculate the gradient of the score for class $c$, $y^c$, with respect to feature map (heat map) activations $A^k$ of a convolutional layer, i.e. $\frac{\partial y^c}{\partial A^k}$. These gradients flowing back are global-average-pooled over the width and height dimensions (indexed by i and j
respectively) to obtain the neuron importance weights $\alpha_k^c$:

{{< figure src="images/GradCAM-architecture.jpg" align=center title="GradCAM architecture. Source:https://learnopencv.com/intro-to-gradcam/#GradCAM-Visualizations" >}}
$$
\alpha_k^c = \frac{1}{Z}\sum_i{\sum_j{\frac{\partial y^c}{\partial A_{ij}^k}}}
$$
We perform a weighted combination of forward activation maps, and follow it by a ReLU to obtain, 
$$
L_{Grad-CAM}^c = ReLU\left(\sum_k{\alpha_k^cA^k}\right).
$$

After knowing the math, we willl show step by step the process of obtaining a GradCAM image.

{{< figure src="images/heatmap1.png" align=center title="The weighted activation map after each layer." >}}

The first row of images represents the dot product of the activation maps $A^k$ and averaged gradients $\alpha_k^c$ for each activation map in that layer after applying ReLU function. As we can observe, the size of each activation map decreases in the deeper layer due to **MaxPool2d**.

{{< figure src="images/heatmap2.png" align=center title="Overlay of the original image and upsampled activation maps." >}}

The second row show the upsampled activation map. Now we aim to upsample $L_{Grad-CAM}^c$ after each convoluational layer to match the original size. This step is crucial for illustrating the importance of each pixel in the original image for class $c$. To enhance the visual effect, we overlay the original image with the upsampled activation map (heatmap).

{{< figure src="images/heatmap3_misclassified.png" align=center title="'Tumor' pixels leads to 'no tumor' prediction" >}}
The third row reveals that the pixels associated with the tumor (white area at the top middle) are contributing the most to the incorrect prediction of 'no tumor'. I have to admit that, this is a big surprise and reflects how little we understand this model. Further investigation is still underway to hopefully understand the decision-making mechanism of this model.

### Non-binary Classification - Leveraging pretrained model and Transfer Learning
With a few lines of code, we downloaded the `vgg16` model and easily adapt it to our classification task.
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
{{< figure src="images/metrics_comparsion.png" align=center title="Pretrained VGG16 model versus small CNN model" >}}

For a brief comparison, we highlight two key points:
- Overfitting is effectively suppressed in the VGG16 model.
- The performance of the VGG16 model saturates early in the training phase but ultimately performs as well as the smaller CNN model after 60 epochs. This indicates that VGG16 requires less training to perform well, which is not surprising given its prior knowledge as a pre-trained model.

### Segmentation: 3D Unet Model (In Preparation)
The 3D Unet model is perfectly suitable for the combintation of task and [datasets](https://www.med.upenn.edu/sbia/brats2017.html).
{{< figure src="images/u-net-architecture.png" align=center title="Unet Architecture. Source:https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/" >}}


## Reference
- Kaggle dataset: https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri
- Kaggle notebook: https://www.kaggle.com/code/pkdarabi/brain-tumor-detection-by-cnn-pytorch
- Grad-CAM: https://arxiv.org/abs/1610.02391
- BRATS2017: https://www.med.upenn.edu/sbia/brats2017.html
- MRIMaster: https://github.com/lixianphys/MRImaster/tree/published
