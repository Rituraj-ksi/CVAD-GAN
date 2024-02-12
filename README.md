# CVAD-GAN: Constrained video anomaly detection via generative adversarial network - CVAD_gan.ipynb 

This Jupyter notebook (`CVAD_gan.ipynb`) contains the implementation of a CVAD-GAN: Constrained video anomaly detection via generative adversarial network components. The notebook demonstrates the training of the model and provides functionalities for performing inference and visualizing results.

## Overview

The notebook implements a CVAD-GAN model, a hybrid architecture that combines the benefits of both Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs). This model is designed for image generation and anomaly detection.

Peds1 (#Video No.- 21)
<img src='Static/Ped1.gif' align="center" width="100%">
Peds2 (#Video No.- 4)
<img src='Static/Peds2.gif' align="center" width="100%">
Avenue (#Video No.- 12)
<img src='output1.gif' align="center" width="100%">

## Real Life Highway Video

** Note ** We are preparing a new dataset which is not mentioned in the manuscript which is still under Cleaning and adaption process as the purpose of this dataset to take Video Anomaly Detection Problems to Real-Time Video Detection And to Hardware Video Anomaly Detection. below is the result:- 
<img src='Static/file.gif' align="center" width="100%">

The Anomaly in this video is "Bike" going over the Divider which is a clear violation of the traffic rules which is considered as Anomaly and on basis of this we have identified some of the classes such as :
1. Cattels,

2. Wrong direction,

3. Slow/(Fast Speed exceed limit) Moving,

4. Person/Intruder,

5. Poor Visibility,

6. Fallen Object,

7. Stopped Vehicle

It is taken under some conditions such as only taking 5 frames per second out of 30 frames per second from the video. We are still trying to benchmark it on different models and other SOTA models such as :

1. [STemGAN](https://doi.org/10.1007/s10489-023-04940-7):spatio-temporal generative adversarial network for video anomaly detection  (OURS)
2. [A2D-GAN](http://dx.doi.org/10.1016/j.engappai.2023.107830): Attention-guided generator with dual discriminator GAN for real-time video anomaly detection  (OURS)
3. [VALD-GAN](https://doi.org/10.1007/s11760-023-02750-5): VALD-GAN: video anomaly detection using latent discriminator augmented GAN (OURS)
4. [VALT-GAN](https://doi.org/10.1109/SSCI52147.2023.10371992): Video Anomaly Latent Training GAN (VALT GAN): Enhancing Anomaly Detection Through Latent Space Mining (OURS)
5. [ASTNet](https://doi.org/10.1007/s10489-022-03613-1): Attention-based residual autoencoder for video anomaly detection
6. [DeepOC](https://doi.org/10.1109/TNNLS.2019.2933554): A Deep One-Class Neural Network for Anomalous Event Detection in Complex Scenes
7. [Future Frame Prediction for Anomaly Detection – A New Baseline](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Future_Frame_Prediction_CVPR_2018_paper.pdf)



## Dependencies
- python==3.8
- Jupyter Notebook==7.0.8
- scikit_learn==0.23.2
- scipy==1.7.3
- tqdm==4.64.0
- keras==3.0.4
- numpy==1.21.5
- matplotlib==3.5.3
- Pillow==9.5.0
- OpenCV==4.9.0

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter notebook:
```bash
jupyter notebook CVAD_GAN.ipynb
```
2. Execute the cells in the notebook sequentially.

3. Adjust hyperparameters, paths, or other configurations as needed.

## Training
Execute the cells related to model training to train the CVAD-GAN model. This involves loading the dataset, defining the model architecture, and training for a specified number of epochs.

## Inference
Execute the cells related to model inference to generate images and visualize results. This may involve loading a pre-trained model or using the trained model from the training phase.

## Results
View the generated images and anomaly detection results in the notebook. Optionally, save generated images in the img directory.

## File Structure
```bash
├── CVAD_GAN.ipynb                # Jupyter notebook for CVAD-GAN implementation
├── requirements.txt              # List of dependencies
├── img/                          # Directory to save generated images
├── models/                       # Directory to save trained models (if applicable)
├── dataset/                      # Directory containing the dataset
    ├── train/Video1/0.png...     # Training data
    ├── test/Video1/0.png...      # Testing data
```
