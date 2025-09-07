
---

# 🖼️ Image Detection using U-Net

This repository contains a Jupyter Notebook implementation of an **image segmentation pipeline** using a **U-Net model** built with TensorFlow/Keras. The project focuses on detecting and segmenting objects in images based on provided masks.

---

## 📌 What's Inside the Code

The notebook `Image_Detection.ipynb` is organized into the following key parts:

1. **Google Drive Integration**

   * Mounts Google Drive to access datasets (`images` and `annotations` folders).

2. **Imports and Dependencies**

   * Uses `TensorFlow`, `NumPy`, `Matplotlib`, and `Keras` layers/models for deep learning and visualization.

3. **Data Preprocessing**

   * Loads raw images and their corresponding annotation masks.
   * Resizes inputs to `(128x128)` and normalizes pixel values.
   * Provides functions to preprocess image–mask pairs for training.

4. **Dataset Preparation**

   * Defines paths for **training**, **validation**, and **test** sets.
   * Loads datasets into NumPy arrays for model input/output.

5. **U-Net Model Architecture**

   * Implements a U-Net with:

     * Convolutional layers with ReLU activation.
     * Batch Normalization for stability.
     * MaxPooling for downsampling.
     * UpSampling and concatenation for decoder path.
     * Dropout layers for regularization.

6. **Evaluation Metrics**

   * Custom metrics implemented:

     * **IoU (Intersection over Union)**
     * **Dice Coefficient**
     * **Pixel Accuracy**

---

## 🚀 How to Run

1. Clone the repository and open the notebook:

   ```bash
   git clone https://github.com/your-username/image-detection-unet.git
   cd image-detection-unet
   jupyter notebook Image_Detection.ipynb
   ```

2. Ensure your dataset is structured as:

   ```
   gdrive/MyDrive/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── annotations/
       ├── train/
       ├── val/
       └── test/
   ```

3. Run the notebook step by step:

   * Mount Google Drive.
   * Preprocess images and masks.
   * Train the U-Net model.
   * Evaluate using IoU, Dice, and pixel accuracy.

---

## 📊 Expected Output

* **Training Logs**: Loss reduction and validation performance over epochs.
* **Evaluation Metrics**: IoU, Dice Score, and Accuracy.
* **Visualizations**: Comparison of predicted masks vs. ground-truth annotations.

---

## 🛠️ Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy
* Matplotlib
* Google Colab or Jupyter Notebook

---

## 📌 Notes

* The model is designed for **binary segmentation** (foreground vs. background).
* You can extend it to multi-class segmentation by adjusting the final layer and loss function.
* Recommended to use GPUs for faster training.

---

## 📜 License

This project is licensed under the MIT License. This project is for educational and research purposes. Feel free to use and modify.

---
