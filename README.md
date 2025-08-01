# FPGA-Based-Kidney-Cancer-Detector

This repository contains the complete implementation for a Convolutional Neural Network (CNN)-based model trained to detect kidney cancer from medical images and deployed onto an FPGA using the `hls4ml` and Vitis HLS toolchains. It includes code, datasets, Jupyter notebooks, and the research paper.

## Abstract

In this work, a Convolutional Neural Network (CNN) was developed and deployed on an FPGA for the detection of kidney cancer. The CNN model was initially trained and validated in the Google Colab environment, achieving a high validation accuracy of **97.53%**. 

To make the model compatible with FPGA hardware, the trained CNN was converted using the `hls4ml` framework, which facilitates the translation of machine learning models into hardware description languages suitable for FPGAs. The converted model was successfully implemented on a Kintex-7 FPGA.

FPGA deployment enhances system performance by leveraging the parallel processing capabilities of the hardware, enabling faster inference times and energy-efficient operation, making it suitable for real-time medical applications. This work demonstrates the potential of integrating CNN models into FPGA systems for high-accuracy, low-latency medical diagnostics.

## Features

- CNN trained using grayscale CT scan images
- Preprocessing with data augmentation
- Conversion to HLS using `hls4ml`
- Deployment on FPGA (Kintex-7, Vitis HLS)
- Resource usage analysis (LUTs, FFs, DSPs)
- Comparison with other deep learning models

## Tech Stack

- Python 3.9
- TensorFlow, Keras
- OpenCV
- NumPy, Matplotlib
- `hls4ml`
- AMD Vitis AI
- Kintex-7 FPGA

## Dataset

The dataset used consists of CT scan images of cancerous and non-cancerous kidneys, publicly available on **[Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)** and sourced from hospitals in Dhaka, Bangladesh. The dataset contains 12,446 unique data within it in which the cyst contains 3,709, normal 5,077, stone 1,377, and tumor 2,283.


