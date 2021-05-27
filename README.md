# Image segmentation

Includes quick prototype code for performing various methods of image segmentation. Most of the code adapted from different sources from the 'internet'. So there is nothing fancy about this repo itself! ;-)

Code tested briefly on python3 on Ubuntu 18.04. *Disclaimer:* There could be mistakes and not everything may work <br\>
Dependencies are to be found in 'requirements.txt' which should ideally be installed using 
```
pip install -r requirements.txt
```
If you have messed up system installations between python2 and python3, try to use the version exclusively, like pip3. If still something does not work, manual installation of the packages would be a saver!

*Not in this repo*
- All the scripts in this repo have been trained/tested with the Cityscapes dataset. The dataset is huge and it is not even clear if the license allows uploading it to unrestricted repo. So, one can download it from here after registration: https://www.cityscapes-dataset.com/ (Don't forget to grab a lunch or take a nap after pressing *download dataset*)
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
The base for the supervised segmentation is copied (why re-invent the wheel, huh?) from this [repo](https://github.com/dhkim0225/keras-image-segmentation.git). Basically contains three different network architectures - **FCN**, **U-Net** and **PSPNet**. However, the scripts needed some adaptation to get them running, especially training and testing. I have additionally added a script containing the evaluation metrics: *dice coefficient*, *jaccard index* and *pixel accuracy*.

*Preparing Cityscapes data* <br/>
```
python3 dataset_parser/make_h5.py --path "/downloaded/leftImg8bit/path/" --gtpath "/downloaded/gtFine/path/"
```

*Training the model* <br/>
Only 3 classes (person, car, road) and the background are considered. As described in the original repo, the training of a model foo can be done using
```
python3 train.py --model foo --vgg /path/of/pretrained_vgg_weights
```

*Testing the trained models* <br/>
The output of the models can typically be tested using test images from the dataset, which needs to be mentioned within the test script.
```
 python3 test.py --model foo
 ```
 The test scripts masks the images (onyl for the considered classes) and also computes the above mentioned metrics, for the overall as well as class wise instances.
 
 ### Example results
 Coarse results based on just 3 epochs for each model (higher iterations would ofcourse yield better results) <br/>
 *Test image: frankfurt_000001_054219* <br/>
 ![](https://github.com/nageswag/segmentation/blob/feature/collective-code-for-segmentation-and-friends/supervised_segmentation/semantic_segmentation/supervised_seg_results/frankfurt_000001_054219_test_image.png)
 
 *Segmented image from fcn* <br/>
 ![](https://github.com/nageswag/segmentation/blob/feature/collective-code-for-segmentation-and-friends/supervised_segmentation/semantic_segmentation/supervised_seg_results/frankfurt_000001_054219_fcn_res_color.png)
 
 *Segmented image from u-net* <br/>
 ![](https://github.com/nageswag/segmentation/blob/feature/collective-code-for-segmentation-and-friends/supervised_segmentation/semantic_segmentation/supervised_seg_results/frankfurt_000001_054219_unet_res_color.png)
 
 *Segmented image from pspnet* <br/>
 ![](https://github.com/nageswag/segmentation/blob/feature/collective-code-for-segmentation-and-friends/supervised_segmentation/semantic_segmentation/supervised_seg_results/frankfurt_000001_054219_pspnet_res_color.png)
 
