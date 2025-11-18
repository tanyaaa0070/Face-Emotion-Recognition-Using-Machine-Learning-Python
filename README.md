
***

# 🎭 Face Emotion Recognition Using Machine Learning (Python)

This project performs **real-time facial emotion detection** using a **Convolutional Neural Network (CNN)** model trained on grayscale facial images (48×48). It uses **OpenCV** for face detection and **TensorFlow/Keras** for emotion classification.

**Emotion classes detected:**

* 😡 Angry  
* 🤢 Disgust  
* 😱 Fear  
* 😊 Happy  
* 😐 Neutral  
* 😢 Sad  
* 😮 Surprised

***

## 🚀 Features

✔️ Real-time face detection using OpenCV  
✔️ Emotion classification with a pre-trained CNN  
✔️ Uses `.h5` model weights only (no separate architecture file)  
✔️ Supports image and webcam input  
✔️ Lightweight and fast  
✔️ Compatible with CPU or GPU

***

## 📁 Project Structure

```
.
├── images/                   # sample images
├── facialemotionmodel.h5     # model weights
├── realtimeDetection.py      # real-time detection script
├── trainmodel.ipynb          # training notebook (optional)
├── requirements.txt          # dependencies
└── README.md                 # documentation
```

***

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/tanyaaa0070/Face-Emotion-Recognition-Using-Machine-Learning-Python.git
cd Face-Emotion-Recognition-Using-Machine-Learning-Python
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

***

## 🎥 Real-Time Emotion Detection

Run the detection script:

```bash
python realtimeDetection.py
```

The webcam will open and display:

* Detected faces highlighted by rectangles  
* Predicted emotion label shown near the face

Make sure the `facialemotionmodel.h5` file is in the same folder as the script.

***

## 🧠 Model Details

The emotion recognition model is a **Convolutional Neural Network (CNN)** trained on 48×48 grayscale face images.

Detection and inference pipeline:

1. Face detection using OpenCV’s Haar Cascade  
2. Face extraction, resizing to 48×48, and normalization  
3. CNN predicts the emotion class from the processed face image

***

## 📦 Requirements

Key dependencies:

* Python 3.x  
* TensorFlow / Keras  
* NumPy  
* OpenCV  
* Pillow  
* tqdm (optional)  
* scikit-learn (optional)

For full details, see `requirements.txt`.



## 🤝 Contributing

Contributions and suggestions are welcome! Feel free to open issues or pull requests.

***

## 📜 License

This project is licensed under the **MIT License**.

***

## ⭐ Support

If you find this project useful, please give it a star ⭐ on GitHub!

***



***

