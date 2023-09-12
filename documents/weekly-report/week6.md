
# Week 6 (Aug. 28 ~ Sept1.) 
**Summary of the week**  
1. Solve the issue that the output of perception module is incorrect orientation.
2. Find computation resource (GPU) from TU Delft and TU Eindhoven side. From TU Delft side: Since installation of Apollo need root user access, the server center can't grant for root user. From TU/e side: the server center also can't grant for root user. Also consider usign the PC in labortory, but the Cuda version of RTX4090 GPU is too new. The apollo still cann't support this version of cuda.([issue from official github page](https://github.com/ApolloAuto/apollo/issues/14821).) In conclusion, **I still have to stick on my own laptop and suffer from the memory issue and waiting for 5 minutes frezze after launching the perception module**.
## Weekly outcome
- output of perception module is incorrect orientation (-90 deg. compare to the correct orientation). Modify the the 3 dimensional projection transformation matrix: 
```
# Incorrect transformation (x,y,z translation and pitch, yaw, roll)
LIDAR_TRANSFORM = np.array([[ 0.0020333, 0.9997041, 0.0242417, 0.9437130],
                            [-0.9999805, 0.0021757,-0.0058486, 0.0000000],
                            [-0.0058997,-0.0242294, 0.9996890, 1.8402300],
                            [ 0.0000000, 0.0000000, 0.0000000, 1.0000000]])

# Correct transformation (rotate the yaw axis by -90 deg.)
LIDAR_TRANSFORM = np.array([[-0.9997041, 0.0020333, 0.0242417, 0.943713 ],
                            [-0.0021757,-0.9999805,-0.0058486, 0.       ],
                            [ 0.0242294,-0.0058997, 0.9996890, 1.84023  ],
                            [ 0.0000000, 0.0000000, 0.0000000, 1.0000000]])
```
## Next week task

### Urgent
- Keep asking for the GPU resource. (require: higher GPU RAM > 16 GB, prefer RTX30, RTX20 series)
### Normal
- Finish prepare the dataset (w, w') for PEMs training. (for both .record file and .txt file). 