# 🧠 Multimodal Fact-Checking using CLIP Classifier

This repository is a **simplified demo** and review of the paper:

> **"Open-Domain, Content-Based, Multimodal Fact-Checking of Out-of-Context Images via Online Resources"**  
> *Sahar Abdelnabi, Rakibul Hasan, and Mario Fritz — CVPR 2022*

📄 [Original Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Abdelnabi_Open-Domain_Content-Based_Multi-Modal_Fact-Checking_of_Out-of-Context_Images_via_CVPR_2022_paper.html)  
💻 [Official Repository](https://github.com/S-Abdelnabi/OoC-multi-modal-fc)

---

## 🎯 Overview

This project demonstrates how to use a **fine-tuned CLIP model** to classify **falsified vs. pristine image-caption pairs**, based on the approach in the original paper.

It is built on top of the `Clip_Classifier` model from the official repo and is designed to provide a **clear, end-to-end demo** of multimodal fact-checking.

---

## 🖼️ Example Inputs

| Image | Caption |
|-------|---------|
| ![example image 1](0670_067.jpg) | *President Obama arrives in Ohio.* | Prediction: Falsified
| ![example image 2](0705_423.jpg) | *Hillary Clinton speaks at a book signing for "Hard Choices" at a Barnes & Noble in New York City on June 10, 2014, the day of the book’s release.* | Prediction: Pristine

---

## 📦 Requirements

- Fine-tuned CLIP model
- `Clip_Classifier` model
- VisualNews dataset

👉 All the necessary components and models are provided in the original repo:  
https://github.com/S-Abdelnabi/OoC-multi-modal-fc

---

## 🛠️ Files and Usage

### `Collecting_test_images_captions.py`

- Saves a sample of pristine and falsified image-caption pairs.
- No preprocessing is applied.

### `demo.py`

- Loads an image and its caption along with the label.
- Passes them to the `clip_classifier` model to **predict authenticity**.

---

## Reference

Abdelnabi, Sahar, Rakibul Hasan, and Mario Fritz.
"Open-domain, content-based, multi-modal fact-checking of out-of-context images via online resources."
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.





