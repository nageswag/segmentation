# Image segmentation
Includes quick prototype code for performing various methods of image segmentation. Most of the code adapted from different sources from the 'internet'. So there is nothing fancy about this repo itself! ;-)

Code tested on python3 on Ubuntu 18.04. Dependencies are to be found in 'requirements.txt' which should ideally be installed using 
```
pip install -r requirements.txt
```
If you have messed up system installations between python2 and python3, try to use the version exclusively, like pip3. If still something does not work, manual installation of the packages would be a saver!

*Not in this repo*
- All the scripts in this repo have been trained/tested with the Cityscapes dataset. The dataset is huge and it is not even clear if the license allows uploading it to unrestricted repo. So, one can download it from here after registration: https://www.cityscapes-dataset.com/ (Don't forget to grab a lunch pr take a nap after pressing *download dataset*)
- The model weights are heavy-weight, so not uploaded. So to get the weights, the networks have to be trained locally on your machine (and ofcourse grab another meal, a KFC bucket sort of thing ;-)).
- Weights for the pre-trained VGG16 encoder can be downloaded from [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)

## Classical segmentation
Contains Laplacian operator and Laplacian of Gaussian implementations
```
python3 laplace_operator.py
```

## Unsupervised segmentation
Is done using k-means clustering (only 4 classes are assumed)
```
python3 k_means.py
```

## Supervised segmentation
The base for the supervised segmentation is copied (why re-invent the wheel, huh?) from here: https://github.com/dhkim0225/keras-image-segmentation.git. Basically contains three different network architectures - **FCN**, **U-Net** and **PSPNet**. However, the scripts needed some adaptation to get them running, especially training and testing. I have additionally added a script containing the evaluation metrics: *dice coefficient*, *jaccard index* and *pixel accuracy*.

*Training the model*
As described in the original repo, the training can be done using
```

```
