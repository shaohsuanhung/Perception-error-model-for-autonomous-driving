# Self Driving Internship   
## Introduction
The 14 weeks full-time research internship. The main goal of the project is to discuss different combination of sensors (camera, radar, LiDAR, camera+radar, camera+radar+LiDar, etc.) under different environment conditions (rain,sunny, dim-light, overexposure light, etc.). 

## Project
### Problem statement
The designed HMM-based model using multi-modal sensors (camera, radar, LiDAR, camera+radar, camera+radar+LiDar) react under different environment conditions (rain,sunny, dim-light, overexposure light,etc.). The designed HMM-based model is compared with the baseline. (Apollo perception module)
### Project phase 1
1. Literature review on sensor modeling.
2. Converting nuScenes dataset (radar data especially) to the .record file (see ```./scripts/nuscenes_converter```) that can play on the Apolloauto system. Using the perception modules in the Apollo system to investigate perception outcome that perceived by Apollo, and compute the metrics (e.g., IOU) with the ground truth.

### Project phase 2
Coming up with a self-designed perception model such as HMM, then compare and analyses the perception module of the Apollo.s

### Schedule
![](documents/images/Gantt%20Chart.PNG)