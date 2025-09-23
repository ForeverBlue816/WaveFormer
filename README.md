# WaveFormer: A Lightweight Transformer Model for sEMG-based Gesture Recognition

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2506.11168)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation of the paper **"WaveFormer: A Lightweight Transformer Model for sEMG-based Gesture Recognition"**.

WaveFormer is a lightweight transformer-based architecture specifically tailored for surface electromyographic (sEMG) gesture recognition. Traditional deep learning models are often too large and computationally expensive for deployment on resource-constrained embedded systems. Our work addresses this challenge, demonstrating that careful architectural design can eliminate the traditional trade-off between model size and accuracy.

## ✨ Key Features

* **Lightweight Design**: With only **3.1 million parameters**, the model is perfectly suited for wearable devices and embedded systems with limited computational power.
* **Innovative WaveletConv Module**: Introduces a novel learnable wavelet transform that adaptively integrates time-domain and frequency-domain features to enhance feature extraction. It uses depthwise separable convolutions to ensure both efficiency and compactness.
* **Efficient Transformer Architecture**: Employs efficient Transformer blocks with Rotary Positional Embedding (RoPE) to effectively capture long-range dependencies in the sEMG signals.
* **State-of-the-Art Performance**: Achieves SOTA performance on multiple public sEMG datasets, outperforming much larger foundation models and other lightweight architectures.
    * **95%** classification accuracy on the EPN612 dataset.
    * **81.93%** accuracy on the challenging Ninapro DB6 inter-session protocol.
* **Real-Time Inference**: Achieves an inference latency of just **6.75 ms** per sample on a standard CPU (Intel Core i7-11800H) with INT8 quantization, making it ideal for real-time applications.

## 🔬 Model Architecture

The WaveFormer pipeline is illustrated below:

<p align="center">
  <img src="figure/architecture.png" width="800"> </p>

1.  **Input Signal Processing**: The raw multi-channel sEMG signal (`C x T`) is first preprocessed, including filtering and normalization.
2.  **Patch Embedding**: A learnable 2D convolution layer partitions the signal into non-overlapping patches and maps them to a fixed-dimensional latent vector, creating a 2D feature map (`D x C x N`).
3.  **WaveletConv**: This core module performs multi-level wavelet decomposition and reconstruction on the 2D feature map to extract rich, multi-scale time-frequency features.
4.  **Transformer Encoder**: The wavelet-enhanced features are flattened and fed into a 6-layer Transformer encoder. The encoder uses **RoPEAttention** to capture global temporal correlations.
5.  **Classification Head**: Finally, a simple linear classification head predicts the gesture class based on the Transformer's output.

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Clone this repository
git clone [https://github.com/your-username/WaveFormer.git](https://github.com/your-username/WaveFormer.git)
cd WaveFormer

# Create and activate a conda environment (recommended)
conda create -n waveformer python=3.9
conda activate waveformer

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Datasets

We used four public datasets in our experiments: EPN-612, Ninapro DB5, Ninapro DB6, and UCI EMG.

Please download the datasets and place them in the `data/` directory. We provide preprocessing scripts to perform the filtering, normalization, and windowing as described in the paper.

```bash
python scripts/preprocess_data.py --dataset EPN612
```

### 3. Train the Model

Use the following command to start training. You can customize the run by modifying the config files or command-line arguments. Training parameters such as learning rate, weight decay, batch size, and epochs are detailed in the paper.

```bash
python train.py --dataset EPN612 --batch_size 64 --lr 4e-5 --epochs 30
```

### 4. Evaluate the Model

Evaluate a trained model using a saved checkpoint.

```bash
python evaluate.py --dataset EPN612 --model_path /path/to/your/checkpoint.pth
```

## Citation

If you use our code or model in your research, please cite our paper:

```bibtex
@article{chen2025waveformer,
  title={WaveFormer: A Lightweight Transformer Model for sEMG-based Gesture Recognition},
  author={Chen, Yanlong and Orlandi, Mattia and Rapa, Pierangelo Maria and Benatti, Simone and Benini, Luca and Li, Yawei},
  journal={arXiv preprint arXiv:2506.11168},
  year={2025}
}
```

## Acknowledgments

This research was partly supported by the EU Horizon Europe project IntelliMan (g.a. 101070136), the PNRR MUR project ECS00000033ECOSISTER, and the ETH Zürich's Future Computing Laboratory funded by a donation from Huawei Technologies. The research was also partially supported by the EU Horizon Europe project HAL4SDV (g.a. 101139789).

## License

This project is licensed under the [MIT License](LICENSE).
