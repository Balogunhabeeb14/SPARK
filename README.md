# ♻️ SPARK — Smart Plastic Automated Recognition Kit

> **AI-powered plastic recognition and automated sorting**

SPARK (**Smart Plastic Automated Recognition Kit**) is a collaborative student research and engineering project developed at the **School of Computer Science and Engineering, University of Westminster, London, UK**.

The project explores how **computer vision, deep learning, embedded systems, and IoT** can be combined to recognise plastic waste and support automated sorting.

SPARK was developed by postgraduate students from **Artificial Intelligence and Environmental Science** through the **University of Westminster's Student as Researcher Programme**.

---

## 🎥 Demo

### See SPARK in action

**[▶️ Watch the SPARK prototype demonstration](YOUR_VIDEO_LINK_HERE)**

The demonstration shows the SPARK prototype using the deployed **MobileNetV3-Large** model to recognise plastic waste and provide the classification used by the sorting system.

> **Tip:** If you upload the video to GitHub, replace `YOUR_VIDEO_LINK_HERE` with the GitHub video URL. A short demo video or GIF near the top of the README is highly recommended because it immediately shows what the project actually does.

---

## 📌 Overview

Plastic waste is difficult to sort automatically because different materials can look very similar and real-world images are rarely perfect.

SPARK was built to explore a practical question:

> **Can a lightweight AI model running on embedded hardware recognise plastic waste well enough to support an automated sorting system?**

To investigate this, we developed and compared three deep-learning models, selected a model for deployment, and integrated it into a physical prototype.

The final system uses **MobileNetV3-Large** for plastic classification.

---

## ✨ Features

* 🧠 AI-based plastic waste classification
* 📷 Computer vision and image processing
* 🔬 Multiple CNN architectures evaluated
* ⚡ Lightweight model selected for edge deployment
* 🤖 Automated sorting prototype
* 🔌 Embedded-system integration
* 🌐 IoT capabilities
* 📱 Real-world image validation
* 📊 Three rounds of experimentation
* ♻️ Designed around practical plastic waste sorting challenges

---

## 🏗️ System Architecture

The SPARK system follows an end-to-end pipeline from image capture to sorting:

```text
┌──────────────────┐
│   Plastic Waste  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Image Capture   │
│     / Camera     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Image Processing │
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  MobileNetV3-Large   │
│    AI Classifier     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Classification Result│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Sorting Decision   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Automated Sorting   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  IoT / Monitoring    │
└──────────────────────┘
```

---

## 🧠 Machine Learning

Three convolutional neural network architectures were developed and evaluated:

| Model             | Evaluated | Deployed |
| ----------------- | :-------: | :------: |
| EfficientNet-B3   |     ✅     |     ❌    |
| ResNet-50         |     ✅     |     ❌    |
| MobileNetV3-Large |     ✅     |   **✅**  |

### EfficientNet-B3

EfficientNet-B3 was investigated as a high-capacity image-classification model and provided one of the comparison points during model evaluation.

### ResNet-50

ResNet-50 was included because of its established performance in image-classification tasks and its use of residual connections to support deeper networks.

### MobileNetV3-Large

MobileNetV3-Large was selected as the **deployment model**.

For SPARK, model selection was not based solely on classification performance. The model also needed to be practical for deployment on embedded hardware, where **memory, computational requirements, and inference speed** are important considerations.

MobileNetV3-Large provided the most suitable balance for the prototype and was therefore used for the final deployment.

---

## 📊 Dataset

### WaDaBa Dataset

The primary dataset used for model development and evaluation was the **WaDaBa dataset**.

The SPARK team was **granted access to the dataset for use in this project**. It provided the image data used during the model development and evaluation stages.

The dataset was used to train and evaluate:

* EfficientNet-B3
* ResNet-50
* MobileNetV3-Large

Using a common dataset allowed us to compare the three architectures under the same experimental conditions.

> **Dataset attribution:** Please include the official WaDaBa citation and attribution information provided by the dataset creators. The dataset is used in this project with permission.

---

## 🧪 Experimental Evaluation

The project was carried out through **three rounds of experimentation**.

### Round 1 — Model Development

The first round focused on preparing the dataset and developing the three candidate models.

```text
WaDaBa Dataset
      │
      ▼
Pre-processing
      │
      ▼
Training
      │
      ├───────────────┐
      ▼               ▼
EfficientNet-B3    ResNet-50
      │               │
      └───────┬───────┘
              ▼
       MobileNetV3-Large
```

The purpose of this stage was to establish the initial performance of the different architectures on the plastic classification task.

### Round 2 — Model Evaluation & Selection

The second round focused on comparing the models and considering their suitability for the actual SPARK hardware.

We looked beyond model performance alone and considered the practical requirements of running AI on an embedded device.

Following this evaluation, **MobileNetV3-Large was selected for deployment**.

### Round 3 — Real-World Validation

For the final round, we collected our own images of different types of plastic waste using **mobile phones**.

We intentionally captured images with different levels of quality, including:

* Good-quality images
* Poor-quality images
* Different lighting conditions
* Different backgrounds
* Different camera angles
* Different object orientations
* Different distances from the object

This gave us an additional way to test the deployed model outside the controlled conditions of the main dataset.

---

## 📱 Real-World Image Collection

The additional mobile-phone images were collected by the project team specifically to see how the model would behave with more realistic inputs.

Instead of testing only clean and well-composed images, we included photographs where the plastic item might be:

* poorly lit,
* partially unclear,
* viewed from an unusual angle,
* surrounded by a distracting background, or
* captured at lower image quality.

This was important because a real sorting system cannot assume that every image will look like a carefully prepared dataset image.

The additional images were used for **validation and testing**, rather than replacing the main WaDaBa dataset.

---

## 🔄 End-to-End Workflow

The complete SPARK workflow can be summarised as:

```text
       Plastic Waste
             │
             ▼
       Image Capture
             │
             ▼
      Image Pre-processing
             │
             ▼
    MobileNetV3-Large Model
             │
             ▼
     Plastic Classification
             │
             ▼
       Sorting Decision
             │
             ▼
      Physical Sorting
             │
             ▼
        IoT / System
         Monitoring
```

---

## 🛠️ Technology Stack

### AI & Computer Vision

* Python
* Deep Learning
* Convolutional Neural Networks
* EfficientNet-B3
* ResNet-50
* MobileNetV3-Large
* Image classification
* Computer vision

### Embedded System

* Embedded AI inference
* Camera/image acquisition
* Hardware control
* Automated sorting mechanism

### IoT

* Device communication
* System monitoring
* Connected-device integration

### Development

* Python
* Git
* GitHub
* Jupyter Notebooks
* Model training and evaluation tools

> **Note:** Add the exact frameworks and hardware used by the project here, for example PyTorch/TensorFlow, OpenCV, Raspberry Pi, ESP32, Arduino, etc.

---

## 📁 Repository Structure

```text
SPARK/
│
├── assets/
│   ├── spark-demo.mp4
│   └── images/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── validation/
│
├── models/
│   ├── efficientnet_b3/
│   ├── resnet50/
│   └── mobilenetv3_large/
│
├── src/
│   ├── training/
│   ├── inference/
│   ├── preprocessing/
│   └── sorting/
│
├── hardware/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── requirements.txt
├── README.md
└── LICENSE
```

Update this structure to match the actual repository.

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd SPARK
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the inference application:

```bash
python src/inference/predict.py
```

> Update the commands above to match the actual entry points in the repository.

---

## 📈 Project Workflow

```text
Requirements
     ↓
Dataset Preparation
     ↓
Model Development
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
MobileNetV3-Large Selection
     ↓
Embedded Deployment
     ↓
Real-World Validation
     ↓
Prototype Testing
     ↓
Demo
```

---

## 🌍 Why SPARK?

Material recovery facilities (MRFs) rely on combinations of mechanical, optical, and manual sorting technologies. However, increasingly complex packaging and plastic materials can make identification and separation difficult.

Multilayer films, polymer blends, composite packaging, and opaque or black plastics can present challenges for conventional identification systems (OECD, 2022). Labour requirements, operational costs, inconsistent manual throughput, and cross-contamination are additional challenges reported in the waste-sorting literature (Citrasari *et al*., 2019; Lubongo and Alexandridis, 2022; Son and Ahn, 2024).

SPARK explores an alternative approach using **AI-powered visual recognition combined with embedded automation**.

The goal is not to replace existing industrial recycling infrastructure, but to demonstrate how intelligent recognition and automation could contribute to future waste-sorting systems.

---

## 🎯 Project Objectives

The project set out to:

* Develop an AI system for plastic waste recognition.
* Compare different deep-learning architectures.
* Identify a model suitable for embedded deployment.
* Deploy the selected model in a physical prototype.
* Connect AI classification to an automated sorting process.
* Test the model using both an established dataset and independently collected images.
* Evaluate performance under both good- and poor-quality image conditions.
* Demonstrate how AI, embedded systems, and IoT can work together in a practical application.

---

## 👥 Team

SPARK was developed as a collaborative postgraduate student project at:

**School of Computer Science and Engineering**
**University of Westminster**
**London, United Kingdom**

The team includes postgraduate students from **Artificial Intelligence and Environmental Science**.

The project brought together different areas of expertise to address the same problem from both a **technical** and **environmental** perspective.

---

## 🙏 Acknowledgements

We would like to sincerely thank the **University of Westminster's Student as Researcher Programme** for supporting SPARK and making it possible for us to undertake this project as student researchers.

The programme gave us the opportunity to move beyond classroom-based learning and work collaboratively on a real-world problem, from initial ideas and experimentation through to the development of a working prototype.

We would also like to thank the **School of Computer Science and Engineering, University of Westminster**, for providing the academic environment, resources, and support that helped us develop the project.

We are grateful to the **WaDaBa dataset providers** for granting us access to the dataset and permitting its use for this project. The dataset was an important part of our model development and evaluation.

We also acknowledge everyone who provided guidance, technical support, feedback, and encouragement throughout the development of SPARK.

---

## 🚧 Project Status

**Status: Working Research Prototype**

SPARK is currently a working prototype demonstrating AI-based plastic recognition and automated sorting.

There is considerable scope for future development, including:

* Expanding the range of plastic categories.
* Increasing the size and diversity of the training dataset.
* Improving recognition of difficult and low-quality images.
* Optimising model inference for edge devices.
* Improving the physical sorting mechanism.
* Expanding IoT monitoring and reporting.
* Testing with larger and more varied waste streams.
* Evaluating the system under longer-term real-world operating conditions.

---

## 📚 References

* OECD (2022). *Global Plastics Outlook: Economic Drivers, Environmental Impacts and Policy Options*.
* Citrasari *et al.* (2019).
* Lubongo and Alexandridis (2022).
* Son and Ahn (2024).

---

## 📄 Disclaimer

SPARK is an academic research and engineering prototype developed for educational, research, and demonstration purposes. It is not currently intended for commercial or industrial-scale deployment.

The project demonstrates the feasibility of combining AI-based plastic recognition with embedded automated sorting; further development, testing, and validation would be required before deployment in a real material recovery facility.
