# ASL Translator ![CI build ver](https://img.shields.io/badge/ALST-v1.5-yellow.svg) ![CI status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![CI backend](https://img.shields.io/badge/tensorflow-v1.7.0-blue.svg) ![CI neuralnet](https://img.shields.io/badge/keras-v2.1.5-blue.svg) ![CI sklearn](https://img.shields.io/badge/scikit%20learn-0.19.1-blue.svg) ![CI photoshopper](https://img.shields.io/badge/pillow-5.1.x-blue.svg)


Version 1 of **American Sign Language Translator** is a simple photo to text translator that will be groundwork for a larger project of turning images of human hands into words and phrases in real time.

For me, the overall goal of this capstone is to design a python based neural network that mirrors what [Google Translate](https://www.google.com/search?q=google+translator) does, but for sign language.
## Dataset
I found a well made starter dataset [here](http://www.idiap.ch/resource/gestures/) that seems to be made for a project like this. I am currently using the Sebastien Marcel Static Hand Posture Database. I would suggest starting here before moving on to the entire alphebet as the images are small and there are about 5000 of them. Although the classes are imbalanced, I was able to beat baseline.

## Installation

### My Setup
* Mac
* Python 3 on Anaconda 64-bit
* Jupyter Notebook executor
* Keras Neural network
* Tensorflow backend
* I run all my neural network projects inside of its own separate source activate environment in order to protect my computer and defend against update conflicts.

## Import list

```python
import os, glob, cv2, random, re, keras, tensorflow, itertools
import numpy as np
import pandas as pd
import skimage.io as skio
import skimage.transform as skt
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from IPython.core.interactiveshell import InteractiveShell
from sklearn.cross_validation import train_test_split, cross_val_score
```
###### While a few of these packages do the same things, you never know how they work with Keras until you try them. You can use one resizing method if you wish.
###### When converting to gray scale with matplotlib images are displayed in a hue of color (yellow-green in my case) rather then black and white. Until you pair the image with a grayscale displayer. Interesting.

## Problem
There are a lot of sign languages, over 200 in fact. There have been many projects that either take on American or another individual sign language I did not see much that resembled versatility of google translate for consumers.

## Hypothesis
A well trained convolutional neural network should be able to easily distinguish hand sign images, this project will test the scalability of the concept.
## Preprocessing
My images are being set to float data type, turned gray, and reduced in size before they enter the neural network. In this dataset almost all of the images are 60x80 but neural nets work better with squares and as small of an image you can get so I went with 64x64.

```python
all_images = []
largest = 0,0
for image in image_paths:
    img = skio.imread(image, as_grey=True)
    img = img.astype('float32')
    img = skt.resize(img,(64, 64))
    all_images.append(img)

    '''sanity check'''
    if img.shape > largest:
        largest = img.shape
        print(image)
        print(largest)

converted_X = np.array(all_images)
```
###### Earlier in the code I got an idea of the image sizes I am working with with that simple if-statement. In my sanity check I am making sure that only one size and file are returned


## Neural Network Development
###### Since it is only guessing one letter it is a pretty standard binary logistic classifier.

```python
model = Sequential()
model.add(Conv2D(filters = 6,
                 kernel_size= 3,
                 activation='relu',
                 input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(filters = 16,
                 kernel_size= 3,
                 activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```
If you noticed I am resizing my images again as it is read into the neural network. This is because code fluidity is lost in translation between the two packages. The first example doesn't accept or need the one because I am setting an image that is in greyscale. So the single dimension is understood. In the second example, the neural network is just working with images so it needs to know how many dimensions they will have and thus the (64, 64, **1**) needs to be stated.
#### [Orientation Example code](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).
I will be using an orientation method to balance in the future. However, there will be many words like queen that I will never find nearly enough images of and the image rotation will have its limits, because I plan to restrict its range of rotation between 10 - 15 degrees of the original.

## Results
Current stage of project is able to identify the target with 92% accuracy trained on a dataset of about 5000 images; with 500 images of the target. It is able to correctly identify images that are not the target with 99% accuracy. The baseline for this project was at 89%. While I am satisfied with my first results. I am going to work on finding or making a better dataset and look into how similar signs can be to use code that orientates an image multiple times.

<br/>
<p>
  <img src='https://github.com/DietMocha/ASL_translator/tree/master/pics/cm1.jpg'>
  <img src='https://github.com/DietMocha/ASL_translator/tree/master/pics/roc1.png'>
</p>
<br>

## Upcoming Hazards
* **Orientations concern me**. To balance my original classes of A, B, C, V, pointer, and high-five I looked into rotating random images until my dataset was balanced. However, this would not be future proof and expandable sign language depends on orientation for information and many phrases or words can seem similar; especially when focusing on the top 21 most known languages.
* **Multi appendage and body inclusion words**. SL words and phrases can be one handed, two handed, or even involve the body.
* **Signs that involve motion concern me**. Many words and phrases in SL are conveyed though motion, an action or an series of movements. I am not certain at this point that this type of neural network will be able to guess based on a string of images like a video.
* **Android based app**. The neural network will need to be trained at the PC level as phone hardware cannot support neural networks, but it will need to gather data from the app. So I will need to learn the language of android.
* **Rare words**. I will undoubtedly not be able to work with a lot of images for certain words, this will only magnify as I include other languages, and I certainly will not be able to take 100 pictures of every letter, word, and phrase of any language as I am not train in sign language.

## Version Milestones
* ~~v1 Accurately predict any one letter in the American sign language alphabet.~~
* v2 Accurately decipher images of individual letters in the American alphabet.
* v3 Accurately decipher images of words of simple the American sign language.
* v4 Accurately predict what phrases are said using the body.
* v5 Accurately predict what is being said in a video of a speech being made in sign language.
* v6 Repeat process for 2-3 different sign languages.
* v7 Integration for app and translation between languages.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
Once again, I suggest looking into creating a sandbox environment for this project as it uses Tensorflow.

### All my love to these folks

[General Assembly](https://generalassemb.ly/) - for bringing me this far in just a few months of training.

[Marco Tavora](https://github.com/marcotav), [Ravi Mehta](https://github.com/rdamehta), [Evan Kranzler](https://github.com/theelk801), and [Tova Hirsch](https://github.com/tovahirsch) - for all the late night dms about concepts over parts of our neural network capstones before the course completed. <3
