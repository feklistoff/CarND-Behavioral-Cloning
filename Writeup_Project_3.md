## **Behavioral Cloning Project**

### 1. Dataset overview

Initially, I recorded my driving on track one, there were two full laps (clockwise and counter-clockwise) and a few takes of 
two most difficult turns. However, that wasn't enough for the model (even with sophisticated data preprocessing pipeline) to
successfully drive on track two, so I ended up recording two laps of track two as well.
I had `12757` samples. Each sample was a shot from three cameras (left, center, right) and an angle value.

A few random images from the dataset:

<img src="writeup_imgs/dataset_imgs.png" width="700px">

When I plotted andles distribution I got this:

<img src="writeup_imgs/angles_distribution.png" width="350px">

### 2. Dataset preprocessing

#### Dealing with angles

As we can see dataset contains a lot of '`0`' values. In order to fix this and to add some recovery to my data, as it is described [in this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by NVIDIA, I devided dataset into three pieces using the fact that `-1 < angle < 1`:
* driving straight and add recovery if `-0.35 < angle < 0.35`
* turn left if `-0.55 < angle <= -0.35` and sharp turn left if `angle <= -0.55`
* turn right if `0.35 <= angle < 0.55` and sharp turn left if `angle >= 0.55`

#### Dealing with images

In order to make my model more robust I used these steps when dealing with images:
* First, I cropped and resized images. Cropped because I needed only bottom halves of images (where the road) and resized to 64x64 because otherwise I had too many parameters.

<img src="writeup_imgs/resized_imgs.png" width="350px">

* Images were corrected with `cv2.equalizeHist()`

<img src="writeup_imgs/eqhist_imgs.png" width="350px">

* Next, random light and random shadow

<img src="writeup_imgs/shadow_imgs.png" width="350px">

* After that random shift

<img src="writeup_imgs/shifted_imgs.png" width="350px">

* Random rotation

<img src="writeup_imgs/rotated_imgs.png" width="350px">

* As NVIDIA suggested I used YUV color space

* And the final step, random flip

After all steps I got this distribution:

<img src="writeup_imgs/augm_angles_distribution.png" width="350px">

### 3. Model Architecture

When choosing the 'right' model I started with my own architecture and quickly switched to NVIDIA-like model because it showed best results.

I ended up using these parameters:
* image size `64x64`
* batch size `64`
* Adam optimizer with learning rate `0.001`

Network:


### 4. Creation of the Training Set & Training Process

