![localizer](https://github.com/ivan-alles/localizer/workflows/CI/badge.svg)

# Localizer

[![Video Intro](/assets/youtube_thumbnail.jpg)](https://youtu.be/M1_5VaDYxK4 "Video Intro")

Localizer is a neural network for 2D object detection. 
Unlike popular algorithms finding bounding boxes, Localizer predicts accurate object coordinates
and orientation. This data is essential in robotic applications to plan precise motions and avoid collisions.

Check out the online [hand detector app](https://ivan-alles.github.io/localizer/) to see Localizer in action.

Provided enough training data, Localizer:
* Reaches accuracy less than 2 pixels for the position and  2 degrees for orientation.
* Works with rigid and flexible objects.
* Detects an unlimited number of object categories.
* Adapts to variations in scale, point of view, and lighting.

Transfer learning reduces the amount of training data and time by 10-20 times.

Localizer powered various industrial and hobby projects in:
* Robot control
* Manufacturing quality assurance
* Object counting

This repository contains:
* The source code
* Examples of datasets and models
* A pretrained transfer learning model
* A hands-on demo app for training and running models

## Setup
These instructions use Windows syntax. 

1. To run neural networks on a GPU (highly recommended), 
   install the required **[prerequisites](https://www.tensorflow.org/install/gpu)** for TensorFlow 2.
2. Get the source code into your working folder.
3. Install the dependencies: `pipenv sync`.
4. Activate the pipenv environment: `pipenv shell`.
5. Add localizer to python: `set PYTHONPATH=.`.  

## Hands-on python demo

<img src="./assets/hands_on.gif">

You can interactively train and run a model on images from your web camera in the hands-on demo app. Run 
`python localizer\hands_on_demo.py [CAMERA_ID]` and follow the on-screen instructions. 
You can select a camera with the optional `CAMERA_ID`parameter. It is an integer with the default value of 0. 

## Converting an existing dataset

A dataset for the Localizer is a JSON list of images, each containing a list of objects 
with the position, orientation, and category:

```json
[
 {
  "image": "myimage.png",
  "objects": [
   {
    "category": 0,
    "origin": {
     "x": 100,
     "y": 200,
     "angle": 3.14159
    }
   }
  ]
 }
]
```

If you have your dataset, you need to convert it to this format.

## Creating a new dataset 

You can use [Anno](https://github.com/urobots-io/anno/) to label images manually. To do this:

1. Download and install [Anno](https://github.com/urobots-io/anno/).
2. Copy `localizer\dataset_template.anno` into the directory with your images and rename it (e.g. `mydataset.anno`).
3. Open `mydataset.anno` with Anno.
4. Label objects with the **object** marker. 
5. Label images without any objects with the **empty** marker.

This anno file can be specified in the model configuration:

```json
{
  "dataset": "path/mydataset.anno"
} 
```

## Training a model
Run `python localizer\train.py PATH_TO_MODEL_CONFIG`. 

For example: `python localizer\train.py models\tools\config.json`.

This command will train and save a Tensorflow model under `models\tools\model.tf`.

## Running a model
Run `python localizer\predict_for_images.py PATH_TO_MODEL_CONFIG IMAGES_DIR`.
 
For example: `python localizer\predict_for_images.py models\tools\config.json datasets\tools`.

## Transfer learning
Transfer learning reduces the number of labeled images required for training by more than 10 times:

<img src="./assets/Training performance.svg">

To use transfer learning in training, add the following to the configuration:

```json
{
  "transfer_learning_base": "models/transfer_learning_base/features.tf",
  "pad_to": 32
} 
```
The input shape must be divisible by 32, for example:

```json
{
  "input_shape": [
    384,
    384,
    3
  ]
} 
```
