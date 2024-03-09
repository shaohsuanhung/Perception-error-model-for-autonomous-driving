## Introduction
The scripts are published in the Apollo 8.0 ver, under the directory [/apollo/modules/tools/dataset/nuscenes](https://github.com/ApolloAuto/apollo/tree/master/modules/tools/dataset/nuscenes). Originally, the scripts can only convert the camera and LiDAR data, not including the annotations and the radar data. So the script: `dataset_converter.py`, `main.py`, `nuscenes_lib.py` are modified for this project use. Use the following command to convert the nuscenes dataset:
## Convert dataset
You can use below command to convert nuscenes dataset to apollo record file. There maybe multi sense in one dataset, and we create a record file for each scene.

```shell
python3 main.py -i {nuscenes_dataset_path} -o {output path} -m {gt or detection flag}
```
The name of the record file is the `scene_token.record`. If you do not specify a path, the file will be saved in the current path. There are only two for the augment `-m` . `-m  gt` records (1) the sensory data of LiDAR, radars, and cameras, (2) annotations in the channels `/apollo/perception/obstacles`, you would see the 3D bounding boxes ground truth in the Dreaview UI. While `-m detection` means records only (1) the sensory data of LiDAR, radars, and cameras data.

## Convert calibration
You can use below command to convert nuscenes calibration to apollo calibration files. There maybe multi sense in one dataset, and we create calibration files for each scene.

```shell
python3 main.py -i {nuscenes_dataset_path} -o {save files path} -t=cal
```

#### Camera intrinsics
Camera intrinsics matrix. ref [link](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html)
- D: The distortion parameters, size depending on the distortion model. For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
- K: Intrinsic camera matrix for the raw (distorted) images.
- R: Rectification matrix (stereo cameras only)
- P: Projection/camera matrix
### For transform nuScenes dataset sensor calibration files
When convert the calibration, the script [TransformTreeGenerator](./TransformTreeGenerator.py) will generate the corresponding dag files, pb.txt files, and launch under the `/apollo/modules/transform/` directory. Noted that the calibration should be to `/apollo/modules/transform/nuScenes_Calibration/` use the following command: (Please run that file under the your apollo directory)
```shell
python3 main.py -i {nuscenes_dataset_path} -o {path to apollo}/modules/transform/nuScenes_Calibration -t=cal
```
## Convert lidar pcd
You can use below command to convert nuscenes lidar pcd to normal pcl file, which you can display in visualizer like `pcl_viewer`.

```shell
python3 main.py -i nuscenes_lidar_pcd_file -t=pcd
```
If you do not specify a name, the default name of the file is `result.pcd`, which is saved in the current directory.
