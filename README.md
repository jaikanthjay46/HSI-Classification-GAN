# HSI-Classification-GAN
A implementation of HSI Classification using GANs
There are two kinds of GAN implemented, 1D GAN uses only spectral features, and 3D GAN uses both spatial and spectral features.
To run the code locally, install the required dependencies (Keras, scikit-learn, patchify, numpy, pandas),

## Steps to run locally
  ### 1D-GAN
  #### Train Phase
  `python3 1dgan_train.py`
  #### Test Phase
  `python3 1dgan_test.py`
  ### 3D-GAN
  #### Train Phase
  `python3 3dgan_train.py`
  #### Test Phase
  `python3 3dgan_test.py`
   
The code uses the pretrained models to predict by default, you should be able to change the model used in the test script.

## Steps to run on Google Colab
  
  - Download and import the ipynb file in Colab
  - Add the ![indianpines-hyperspectral.zip](https://drive.google.com/file/d/1bdYy7yCo48XzqdRRr_ZwmrKCyZFwPLfG/view?usp=sharing)  dataset zip file to the drive root.
  - Also, add the pretrained models to models/ folder in your drive.
  
  


