# Weapon-Generator-using-WGAN-GP-
## Introduction
This repository is a Keras implementation of the WGAN-GP for generating images of weapons.<br />
Most of the code was inspired by [this repository](https://github.com/eriklindernoren/Keras-GAN/tree/master/wgan_gp) by Erik Linder-Norén.
## Training Data
The training data are 128x128 images of weapons with transparent background(png files).<br />
(The images tested are not uploaded to github due to copyrights.)<br />
## Results
The model will produce 128x128 images of weapons.<br />
![](https://i.imgur.com/L7FHOk5.png)

## Usage
### Prerequisite
Place your training data under ./RealImages/.
### Train
    python3 WeaponGenerator.py
At predefined intervals, images generated by the model will be saved in ./OutputImages/, while trained models will be saved in ./TrainedModel/.
