# Zhaorui's work for course project  
An implementation of markerless augmentment reality, based on openCV3.0 bindings for python3.  

Ideas come from book: [Mastering OpenCV](https://github.com/MasteringOpenCV/code)   
copyright by Packt Publishing 2012  

## Dependencies
- openCV3.0+
- python 3.4+
- numpy
- matplotlib

## To run
```
python3 main.py
```

## Project Goal
1. OpenCV implementation of markerless AR
	- Rough. PC-end. Realtime.
	- With a simple openGL rendered object.
2. An Unity-based SDK-based application of AR
	- Possibly markerless
	- Animation of the characters 
	- Interaction techniques, voice control, target follow.  

## This week finished:
- Feature extracting and matchting  
- Camera Calibration  
- Pose estimation  
- Solve PnP  
- Projection  


## Currently work on:
- fixing translation  
	- rendered cube on the center of the roi image.? why?
- make tracking robust to movement
	- severe lag when false positive points detected, and being drawn in renderCube() funciton.
