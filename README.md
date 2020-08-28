# StitchNet

This is the code of my Bachelor's thesis 'StitchNet: Image Stitching using Autoencoders
and Deep Convolutional Neural Networks'.

### Abstract of Thesis
This thesis explores the prospect of artificial neural networks for image processing tasks. 
More specifically, it aims to achieve the goal of stitching multiple
overlapping images to form a bigger, panoramic picture. Until now, this task
is solely approached with ”classical”, hardcoded algorithms while deep learning
is at most used for specific subtasks. This thesis introduces a novel end-to-end
neural network approach to image stitching called StitchNet, which uses a pre-
trained autoencoder and deep convolutional networks.
Additionally to presenting several new datasets for the task of supervised image
stitching with each 120’000 training and 5’000 validation samples, this thesis
also conducts various experiments with different kinds of existing networks de-
signed for image superresolution and image segmentation adapted to the task of
image stitching. StitchNet outperforms most of the adapted networks in both
quantitative as well as qualitative results.

### Autoencoder - Results
<p>
<img src="https://github.com/mauricerupp/StitchNet/blob/master/images/autoencresult.jpg?raw=true" alt="drawing" width="400"/>
</p>

### Network Architecture
<p align="center">
<img src="https://github.com/mauricerupp/StitchNet/blob/master/images/stitchnet.jpg" alt="drawing" width="650"/>
</p>

### Pre-Trained StitchNet with First Dataset - Results
<p align="center">
<img src="https://github.com/mauricerupp/StitchNet/blob/master/images/sn1_s1.jpg" alt="drawing" width="700"/>
</p>

### Pre-Trained StitchNet with Second Dataset - Results
<p align="center">
<img src="https://github.com/mauricerupp/StitchNet/blob/master/images/sn1_s2.jpg" alt="drawing" width="700"/>
</p>
