# Burn-Segmentation
This tutorial is based on the Detectron2 repository by Facebook. This notebook shows training on your own custom instance segmentation objects.

Accompanying Blog Post
We recommend that you follow along in this notebook while reading the blog post on how to train Detectron2 for custom instance segmentation, concurrently.

Steps Covered in this Tutorial
In this tutorial, we will walk through the steps required to train Detectron2 on your custom instance segmentation objects. We use a public American Sign Language instance segmentation dataset, which is open source and free to use. You can also use this notebook on your own data.

To train our segmenter we take the following steps:
- Install Detectron2 dependencies
- Download custom instance segmentation data from Roboflow
- Visualize Detectron2 training data
- Write our Detectron2 Training configuration
- Run Detectron2 training
- Evaluate Dectron2 performance
- Run Detectron2 inference on test images
- Export saved Detectron2 weights for future inference

About
Roboflow enables teams to deploy custom computer vision models quickly and accurately. Convert data from to annotation format, assess dataset health, preprocess, augment, and more. It's free for your first 1000 source images.

Looking for a vision model available via API without hassle? Try Roboflow Train.
![Roboflow Wordmark](https://i.imgur.com/dcLNMhV.png)
