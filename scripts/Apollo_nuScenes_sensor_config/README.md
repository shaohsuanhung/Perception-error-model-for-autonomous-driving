# Apollo Sensor Configuration for nuScenes dataset
There are many configuration files to modify. Please put follow the below instructions.   
* Path files under `/apollo/module/transform/`:  
    1. Put `static_transform_nu.launch` file under `/apollo/module/transform/launch`.  
    2. Put `static_transform_nu.dag` file under `/apollo/module/transform/dag`.  
    3. Put `static_transform_conf_nu.pb.txt` file under `/apollo/module/transform/conf`.  
* Path files under `/apollo/module/perception`:
    1. Put `perception_all_nu.launch` file under `/apollo/module/perception/production/launch`.
    2. Put `dag_streaming_perception_nu.dag` file under `/apollo/module/perception/production/dag`.
    3. Put `sensor_meta_nu.pt` file under `/apollo/module/perception/production/data/perception/common`.
    4. Put `perception_common_nu.pt` file under `/apollo/module/perception/production/conf/perception`.
    5. Put `nu_front_radar_component_conf.pb.txt`, `nu_front_left_radar_component_conf.pb.txt`, `nu_front_right_radar_component_conf.pb.txt`,`nu_rear_right_radar_component_conf.pb.txt`,`nu_rear_left_radar_component_conf.pb.txt` file under `/apollo/module/perception/production/conf/perception/radar`.
    6. Put `nu_recognition_conf.pb.txt`, `nu_velodyne32_detection_conf.pb.txt` file under `/apollo/module/perception/production/conf/perception/lidar`.
    7. Put `fusion_component_conf_nuscenes.pb.txt` file under `/apollo/module/perception/production/conf/perception/fusion`.
    8. Put `lidar_tracking_pipeline_nu.pb.txt`, and `multi_sensor_fusion_pipeline_nu.txt` file under `/apollo/module/perception/pipeline/config`.
    9. Put  `nu_front_left_radar_extrinsics.yaml`, `nu_front_radar_extrinsics.yaml`, `nu_front_right_radar_extrinsics.yaml`, `nu_rear_left_radar_extrinsics.yaml`,`nu_rear_right_radar_extrinsics.yaml` file under `/apollo/module/perception/data/params`.
* Put `nu_velodyne32_novatel_extrinsics.yaml` file under `apollo/modules/drivers/lidar/velodyne/params`.