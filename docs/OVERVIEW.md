# Package

This package will have three main parts:

1. Pipeline
2. Tools
3. Visualization

Each working together to allow complete control of the pipeline.

# Pipeline

The stages of the pipeline are as follows:

1. Pre-processing
    1. Get depth and color frames
    2. Calculate height frame
    3. Normalize height frame
2. Computer Vision
    1. Run segmentation on color frame, get mask
    2. Run segmentation on height frame, get mask
    3. Run keypoint detection on masked color
3. Post-processing
    1. Filtering
        - Calculate IOU between color and depth frames
        - Calculate zero percentage on depth frame
    2. Processing
        1. Smooth masks
        2. Interpolate zeroed heights
        3. Remove background from color image
        4. Rotate images, masks, keypoints to horizontal
4. Modeling
    1. Feature extraction
    2. Weight prediction

Each stage will be separated into its own part of the package, and you'll be able to step through each stage by itself.

# Tools

Our tool set will include:

1. Dataset Tools
    - Convert depth image to normalized height
    - Dataset combining, splitting into train/val/test

2. Basic YOLO Fine-Tuning
    - Select the model path, training params
    - Select the dataset folders to be utilized
    - Output is the tuned model and parameter summary

3. Pipeline Eval
    - Select inputs (labelled or unlabelled)
    - Returns timings, benchmarks, filterings through each step of the pipeline

# Visualization

Running list of visualizations:
- Color and depth images after each stage
