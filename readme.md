# TeacherTrack-SmartBoard

TeacherTrack-SmartBoard is a gesture-based drawing application that uses YOLO for bracelet detection and MediaPipe for hand gesture recognition. The project allows users to draw and erase on a virtual canvas using hand gestures.



## Setup

1. Clone the repository:
    ```sh
    git clone [<repository-url>](https://github.com/ChunchuManoj/Gesture-Based-Drawing-Using-Object-Detection.git)
    cd TeacherTrack-SmartBoard
    ```

2. Create and activate a conda environment:
    ```sh
    conda create -n bracelet python=3.8
    conda activate bracelet
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Training the Model

To train the YOLO model, run:
```sh
python src/train.py
```
Using the Pre-trained Model
To use the trained model for real-time bracelet detection, run:

Gesture-Based Drawing
To start the gesture-based drawing application, run:

Testing GPU Availability
To test if your GPU is available, run:

Configuration
The configuration for training is specified in args.yaml.

Data
The data configuration is specified in data.yaml.

Results
Training results are saved in the train directory, including:

args.yaml: Training arguments
results.csv: Training metrics
best.pt: Best model weights
last.pt: Last model weights
