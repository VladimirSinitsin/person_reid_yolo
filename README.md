# Person re-identification with YOLOv5s

## Setup
1. Open Anaconda Prompt and navigate into project directory `cd path_to_repo`
2. Run `conda env create` from project directory (this will create a brand new conda environment).
3. Run `activate reid_torch` (if you want to run scripts from your console otherwise set the interpreter in your IDE)

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies.
## Getting Started

First of all, you need to place your video file in the directory: `person_reid_yolo/test_data/video.mp4`. 
And change the corresponding constant in the config file: 
https://github.com/VladimirSinitsin/person_reid_yolo/blob/c2d3ffe6c12f1ee7698df02e85f0929ea7a210af/config.py#L20
Now you can run the file `run.py`: 
```
python run.py
```

## Demonstration
https://github.com/VladimirSinitsin/person_reid_yolo/blob/cfe146617254ebe1ca1e244e4ccb4fafdd789163/test.gif
