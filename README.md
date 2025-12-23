AI Image Captioning with Encoder-Decoder & Attention

This repository contains a deep learning project that automatically generates descriptive captions for images. It utilizes a CNN-RNN architecture with a Bahdanau-style Attention mechanism, implemented using TensorFlow and Keras.
🚀 Overview

The model translates visual information into natural language by following an encoder-decoder framework:

    Encoder: Extracts high-level spatial features from input images using a pre-trained InceptionResNetV2 model.

    Attention: Allows the decoder to focus on specific parts of the image while generating each word in the sequence.

    Decoder: A Gated Recurrent Unit (GRU) that predicts the next word in the caption based on previous words and the attended image features.

🛠️ Technology Stack

    Framework: TensorFlow 2.x / Keras

    Feature Extractor: InceptionResNetV2 (Pre-trained on ImageNet)

    Dataset: MS COCO Captions (via TensorFlow Datasets)

    UI/Deployment: Gradio

    Data Pipeline: tf.data API for efficient preprocessing and batching

🏗️ Model Architecture
Image Encoder

    Input: 224×224×3 images.

    Process: The pre-trained InceptionResNetV2 (excluding the top classification layers) acts as a feature extractor. The resulting spatial map is flattened and passed through a Dense layer to match the ATTENTION_DIM.

Caption Decoder

    Embedding Layer: Converts word indices into dense vectors of fixed size (128).

    GRU Layer: Processes the sequence of embeddings and maintains a hidden state.

    Attention Layer: Computes attention weights over the encoder's spatial features, creating a context vector.

    Output: A Dense layer with a Softmax-like activation (logits) over the VOCAB_SIZE (20,000 tokens).

📊 Dataset

We use the COCO Captions dataset, which provides multiple human-written descriptions for each image.

    Standardization: Captions are converted to lowercase and punctuation is removed.

    Tokenization: We use the TextVectorization layer to manage a vocabulary of 20,000 words.

    Special Tokens: Every caption is wrapped with <start> and <end> tokens to signal the beginning and end of generation.

🚀 Getting Started
Prerequisites

  
    pip install tensorflow tensorflow-datasets matplotlib numpy gradio

Training

The training process uses Teacher Forcing, where the actual caption is shifted by one position to serve as the target for the next-word prediction.

    Batch Size: 8 (Optimized for Kaggle/Free-tier GPUs)

    Loss Function: Masked Sparse Categorical Cross-entropy.

Inference

During prediction, the model uses a loop to generate one word at a time. The predicted word at step t is fed back as the input for step t+1 until the <end> token is generated or the maximum length is reached.
🖥️ Usage

To generate a caption for a local image:
Python

    img, result = predict_caption("your_image.jpg")
    print("Generated Caption:", " ".join(result))
