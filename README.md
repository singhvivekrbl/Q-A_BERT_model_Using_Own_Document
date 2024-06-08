# Enhanced Question-Answering System using a document to answer any question related to document.

## Overview
This project is to build a question-answering system using attention-based models- BERT. Here you can register any document and ask any question from that dcoument

## Setup
1. Clone the repository.
2. Create a virtual environment and install dependencies:
    ```bash
    conda create --name qa-env python=3.8
    conda activate qa-env
    pip install -r requirements.txt
    ```

## Training
1. Prepare the dataset:
    ```bash
    python preprocess_data.py 
    ```

## Evaluation
1. Reference the model output:
    ```bash
    python run_inference.py
    ```
