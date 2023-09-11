# Apollo Sensor Configuration for nuScenes dataset
There are many configuration files to modify. Please put follow the below instructions.   
1. Path files under `/apollo/module/transform/`:  
    * Put `static_transform_nu.launch` file under `/apollo/module/transform/launch`.  
    * Put `static_transform_nu.dag` file under `/apollo/module/transform/dag`.  
    * Put `static_transform_conf_nu.pb.txt` file under `/apollo/module/transform/conf`.  
2. Path files under `/apollo/module/perception`:
    * Put `perception_all_nu.launch` file under `/apollo/module/perception/production/launch`.
    * Put `dag_streaming_perception_nu.dag` file under `/apollo/module/perception/production/dag`.
    * Put `sensor_meta_nu.pt` file under `/apollo/module/perception/production/data/perception/common`.
    * Put `perception_common_nu.pt` file under `/apollo/module/perception/production/conf/perception`.
    * Put `nu_front_radar_component_conf.pb.txt`, `nu_front_left_radar_component_conf.pb.txt`, `nu_front_right_radar_component_conf.pb.txt`,`nu_rear_right_radar_component_conf.pb.txt`,`nu_rear_left_radar_component_conf.pb.txt` file under `/apollo/module/perception/production/conf/perception/radar`.
    * Put `nu_recognition_conf.pb.txt`, `nu_velodyne32_detection_conf.pb.txt` file under `/apollo/module/perception/production/conf/perception/lidar`.
    * Put `fusion_component_conf_nuscenes.pb.txt` file under `/apollo/module/perception/production/conf/perception/fusion`.
    * Put `lidar_tracking_pipeline_nu.pb.txt`, and `multi_sensor_fusion_pipeline_nu.txt` file under `/apollo/module/perception/pipeline/config`.
    * Put  `nu_front_left_radar_extrinsics.yaml`, `nu_front_radar_extrinsics.yaml`, `nu_front_right_radar_extrinsics.yaml`, `nu_rear_left_radar_extrinsics.yaml`,`nu_rear_right_radar_extrinsics.yaml` file under `/apollo/module/perception/data/params`.
3. Put `nu_velodyne32_novatel_extrinsics.yaml` file under `apollo/modules/drivers/lidar/velodyne/params`.  

Here is the directory of the configuration.
```
Apollo/module
├── ...
├── drivers
│   ├── BUILD
|   ├── ...
│   ├── lidar
│   │   ├── BUILD
│   │   ├── ...
│   │   └── velodyne
│   │       ├── BUILD
|   |       ├── ...
│   │       ├── params
│   │       │   ├── nu_velodyne32_novatel_extrinsics.yaml
│   │       │   ├── ...
│   │       │   └── ...
|   |       └── ...
├── perception
│   ├── data
│   │   ├── BUILD
│   │   └── params
|   |       ├── ...
│   │       ├── nu_front_left_radar_extrinsics.yaml
│   │       ├── nu_front_radar_extrinsics.yaml
│   │       ├── nu_front_right_radar_extrinsics.yaml
│   │       ├── nu_rear_left_radar_extrinsics.yaml
│   │       ├── nu_rear_right_radar_extrinsics.yaml
│   │       └── ...
│   ├── pipeline
│   │   ├── BUILD
│   │   ├── config
│   │   │   ├── lidar_tracking_pipeline_nu.pb.txt
│   │   │   ├── multi_sensor_fusion_pipeline_nu.txt
│   │   │   └── ...
│   ├── production
│   │   ├── BUILD
│   │   ├── conf
│   │   │   └── perception
│   │   │       ├── fusion
│   │   │       │   ├── fusion_component_conf_nuscenes.pb.txt
|   |   |       |   └── ...
│   │   │       ├── fusion_component_conf.pb.txt
|   |   |       |   ├── ...
│   │   │       │   ├── nu_recognition_conf.pb.txt
│   │   │       │   ├── nu_velodyne32_detection_conf.pb.txt
|   |   |       |   └── ...
│   │   │       ├── perception_common_nu.flag
│   │   │       ├── radar
|   |   |       |   ├── ...
│   │   │       │   ├── nu_front_left_radar_component_conf.pb.txt
│   │   │       │   ├── nu_front_radar_component_conf.pb.txt
│   │   │       │   ├── nu_front_right_radar_component_conf.pb.txt
│   │   │       │   ├── nu_rear_left_radar_component_conf.pb.txt
│   │   │       │   ├── nu_rear_right_radar_component_conf.pb.txt
|   |   |       |   └── ...
|   |   |       └── ...
│   │   ├── dag
|   |   |   ├── ...
│   │   │   ├── dag_streaming_perception_nu.dag
|   |   |   └── ...
│   │   ├── data
│   │   │   └── perception
│   │   │       ├── common
│   │   │       │   ├── sensor_meta_nu.pt
│   │   │       │   └── ...
│   │   │       ├── ...
│   │   │       ├── lidar
|   |   |       |   ├── nu_velodyne32_detection_conf.pb.txt
|   |   |       |   ├── nu_recognition_conf.pb.txt
│   │   │       │   └── ...
|   |   |       └── ...
|   |   |
│   │   |── launch
│   │   |   ├── ...
│   │   |   ├── perception_all_nu.launch
│   │   │   └── ...
│   │   └── ...
├── transform
|   ├── ...
│   ├── conf
│   │   ├── static_transform_conf_nu.pb.txt
│   │   └── ...
│   ├── dag
│   │   ├── ...
│   │   └── static_transform_nu.dag
│   ├── launch
│   │   ├── ...
│   │   └── static_transform_nu.launch
|   └── ...
|...