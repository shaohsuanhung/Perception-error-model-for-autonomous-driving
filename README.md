# self-driving-internship-project
===
## Introduction
The 14 weeks full-time research internship. The main goal of the project is to discuss different combination of sensors (camera, radar, LiDAR, camera+radar, camera+radar+LiDar,etc.) under different environment conditions (rain,sunny, dim-light, overexposure light,etc.). 

## Project phase
### Phase 1
1. Literature review on sensor modeling.
2. Converting nuScenes dataset (radar data especially) to the .record file that can play on the Apolloauto system. Using the perception modules in the Apollo system to investigate perception outcome that perceived by Apollo, and compute the metrics (e.g., IOU) with the ground truth.

### Phase 2
come up with a self-designed perception model such as HMM, then compare and analyses the perception module of the Apollo.s