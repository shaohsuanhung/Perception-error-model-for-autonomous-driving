# !/bin/env/python3
# Author: Shao-Hsuan Hung
# This file is to parse the perceived object from Apollo perception module in corporate with nuScenes dataset.
# In this file, implement (1) Analyze the number of samples of each sensors 
#                         (2) Clean up the unneccessary data in record file,\
#                             and put the data in the following format: \
#                             time_stamp{
#                                    measurement:{
#                                       sneor_id: " " \
#                                        ...
#                                    }
#                             }
from dataclasses import dataclass
import matplotlib.pyplot as plt
import glob
# Dataset analyzer
# Output number of samples:
# lidar:
# radars:
def obj_map_dataset_analyzer(dataset):
    # Check
    obj_map_count_lidar32 = 0
    obj_map_count_radar_front = 0
    obj_map_count_radar_front_left = 0
    obj_map_count_radar_front_right = 0
    obj_map_count_radar_rear_left = 0
    obj_map_count_radar_rear_right = 0
    for objs_each_frame in dataset.object_map:
        for detection_list_frame in objs_each_frame[1]:
            # for each_measurment in list(detection_list_frame):
            #     if(each_measurment.sensor_id == 'velodyne32'):
            #         obj_map_count_lidar32+=1
            #     elif(each_measurment.sensor_id == 'radar_front'):
            #         obj_map_count_radar_front+=1

            #     elif(each_measurment.sensor_id == 'radar_front_left'):
            #         obj_map_count_radar_front_left+=1

            #     elif(each_measurment.sensor_id == 'radar_front_right'):
            #         obj_map_count_radar_front_right+=1

            #     elif(each_measurment.sensor_id == 'radar_rear_left'):
            #         obj_map_count_radar_rear_left+=1

            #     elif(each_measurment.sensor_id == 'radar_rear_right'):
            #         obj_map_count_radar_rear_right+=1
            if(detection_list_frame.sensor_id == 'velodyne32'):
                obj_map_count_lidar32+=1

            elif(detection_list_frame.sensor_id == 'radar_front'):
                obj_map_count_radar_front+=1

            elif(detection_list_frame.sensor_id == 'radar_front_left'):
                obj_map_count_radar_front_left+=1

            elif(detection_list_frame.sensor_id == 'radar_front_right'):
                obj_map_count_radar_front_right+=1

            elif(detection_list_frame.sensor_id == 'radar_rear_left'):
                obj_map_count_radar_rear_left+=1

            elif(detection_list_frame.sensor_id == 'radar_rear_right'):
                obj_map_count_radar_rear_right+=1
            # print("-------------- Number of sample in object map for that frame {}---------------".format(detection_list_frame.time_stamp))
            # print("Scenes token:{}".format(dataset.scene_token))
            # print("Lidar:{}\nRadar_front:{}\nRadar_front_left:{}\nRadar_front_right:{}\nRadar_rear_left:{}\nRadar_rear_right:{}".\
            #     format(obj_map_count_lidar32,obj_map_count_radar_front,obj_map_count_radar_front_left,\
            #     obj_map_count_radar_front_right,obj_map_count_radar_rear_left,obj_map_count_radar_rear_right))
            # print("-------------------------------------------------------------")
    print("-------------- Number of sample in object map ---------------")
    print("Scenes token:{}".format(dataset.scene_token))
    print("Number of ground truth samples:{}".format(dataset.nbr_gt_samples))
    print("Lidar:{}\nRadar_front:{}\nRadar_front_left:{}\nRadar_front_right:{}\nRadar_rear_left:{}\nRadar_rear_right:{}".\
          format(obj_map_count_lidar32,obj_map_count_radar_front,obj_map_count_radar_front_left,\
          obj_map_count_radar_front_right,obj_map_count_radar_rear_left,obj_map_count_radar_rear_right))
    print("-------------------------------------------------------------")

def data_analyzer_plot(dataset):
    # Plot the sample
    rain_lidar32 = 7716
    rain_radar_front = 174
    rain_radar_front_left = 67 
    rain_radar_front_right = 61
    rain_radar_rear_left = 748
    rain_radar_rear_right =  816
    rain_gt_sample = 1762
    #
    sun_lidar32 = 6354 + 8247 + 3893 + 8842 + 3342 + 6187 +10182
    sun_radar_front = 89 + 1345 + 214 + 628 + 32 + 214 + 554
    sun_radar_front_left = 210 + 108 + 0 + 334 + 0 + 578 + 249
    sun_radar_front_right = 17 + 318 + 4 + 68 + 17 + 71 + 1
    sun_radar_rear_left = 686 + 1022 + 220 +1754 + 269 + 521 + 1432
    sun_radar_rear_right = 600 + 1204 + 210 + 1861 + 325 + 609 + 1274
    sun_gt_sample = 2089 + 1957 + 1291 + 2106 + 704 + 935 +2369
    #
    night_lidar32 = 4317+7716
    night_radar_front = 99+174
    night_radar_front_left = 62+67 
    night_radar_front_right = 44+61
    night_radar_rear_left = 263+748
    night_radar_rear_right =  224+816
    night_gt = 748+1762
    #
    sensors = ['lidar','Radar front','Radar front left','Radar front right','Radar rear left','Radar rear right']
    h1 = plt.hist(sensors,weights=[sun_lidar32,sun_radar_front,sun_radar_front_left,sun_radar_front_right,sun_radar_rear_left,sun_radar_rear_right])

class dataset_list(object):
    def __init__(self,DTfile,GTfile,Egofile):
        # Read the file
        with open(DTfile) as f:
            dt_content = f.read()
        with open(GTfile) as f:
            gt_content = f.read()
        with open(Egofile) as f:
            ego_content = f.read()
        # Get the scene token
        self.scene_token = DTfile.split('/')[-1].split('_')[0]
        self.road_condition   = DTfile.split('detection_')[-1].split('.txt')[0]
        self.sensor_data_list = []
        self.groudtruth_list = []
        # Dim: [[detection_list_at_t1,gt_list_at_t1],[detection_list_at_t2,gt_list_at_t2].....\
        #                     [detection_list_at_tn, gt_list_at_tn]]
        # [[element[1]] for element in dataset.object_map] to take the frame by frame 
        self.object_map = []
        self.ego_vehicle_list = []
        self.count_lidar32 = 0
        self.count_radar_front = 0
        self.count_radar_front_left = 0
        self.count_radar_front_right = 0
        self.count_radar_rear_left = 0
        self.count_radar_rear_right = 0
        # Number of gt samples
        self.nbr_gt_samples =0
        # Read in the data
        self._make_detection(dt_content)
        self._make_gt(gt_content,ego_content)
        self._make_localization(ego_content)
        self._generate_object_map()


    def append_sensor_data(self,measurement):
        # Check the number of sample of data
        self.sensor_data_list.append(measurement)
        if(measurement.sensor_id == 'velodyne32'):
            self.count_lidar32+=1
        elif(measurement.sensor_id == 'radar_front'):
            self.count_radar_front+=1

        elif(measurement.sensor_id == 'radar_front_left'):
            self.count_radar_front_left+=1

        elif(measurement.sensor_id == 'radar_front_right'):
            self.count_radar_front_right+=1

        elif(measurement.sensor_id == 'radar_rear_left'):
            self.count_radar_rear_left+=1

        elif(measurement.sensor_id == 'radar_rear_right'):
            self.count_radar_rear_right+=1

    def _make_detection(self,file_content):
        # Use regular expressions to split the file_contents based on "Separator:" lines
        parts = file_content.split('perception_obstacle {')
        # parts = [part.strip() for part in parts]
        filetered_parts = [item for item in parts if not(item.startswith('header'))]
        for part in filetered_parts:
            measurements = part.split('measurements {')
            for multi_sensor_measurement in measurements:
                if multi_sensor_measurement.startswith('\n    sensor_id: '):
                    sensor_id  = multi_sensor_measurement.split('"')[1].split('"')[0]
                    id         =   int(multi_sensor_measurement.split(' id: ')[1].split('\n')[0])
                    position_x = float(multi_sensor_measurement.split('\n      x: ')[1].split('\n      y:')[0])
                    position_y = float(multi_sensor_measurement.split('\n      y: ')[1].split('\n      z:')[0])
                    position_z = float(multi_sensor_measurement.split('\n      z: ')[1].split('\n    }\n')[0])
                    theta      = float(multi_sensor_measurement.split('\n    theta: ')[1].split('\n    length:')[0])
                    length     = float(multi_sensor_measurement.split('\n    length: ')[1].split('\n    width:')[0])
                    width      = float(multi_sensor_measurement.split('\n    width: ')[1].split('\n    height:')[0])
                    height     = float(multi_sensor_measurement.split('\n    height: ')[1].split('\n    velocity {')[0])
                    velocity_x = float(multi_sensor_measurement.split('\n      x: ')[1].split('\n      y: ')[0])
                    velocity_y = float(multi_sensor_measurement.split('\n      y: ')[1].split('\n      z: ')[0])
                    velocity_z = float(multi_sensor_measurement.split('\n      z: ')[1].split('\n    }\n')[0])
                    type       = multi_sensor_measurement.split('\n    type: ')[1].split('\n    timestamp: ')[0]
                    time_stamp = float(multi_sensor_measurement.split('\n    timestamp: ')[1].split('\n    box {')[0])
                    xmin       = float(multi_sensor_measurement.split('\n      xmin: ')[1].split('\n      ymin: ')[0])
                    ymin       = float(multi_sensor_measurement.split('\n      ymin: ')[1].split('\n      xmax: ')[0])
                    xmax       = float(multi_sensor_measurement.split('\n      xmax: ')[1].split('\n      ymax: ')[0])
                    ymax       = float(multi_sensor_measurement.split('\n      ymax: ')[1].split('\n    }\n')[0])
                    self.append_sensor_data(sensor_data(sensor_id, id,position_x, position_y, position_z, \
                                                        theta, length, width,height, velocity_x, velocity_y, \
                                                        velocity_z, type,time_stamp,xmin,ymin,xmax,ymax))
        print("-------------- Number of sample ---------------")
        print("Scenes token:{}".format(self.scene_token))
        print("Lidar:{}\nRadar_front:{}\nRadar_front_left:{}\nRadar_front_right:{}\nRadar_rear_left:{}\nRadar_rear_right:{}".\
          format(self.count_lidar32,self.count_radar_front,self.count_radar_front_left,\
          self.count_radar_front_right,self.count_radar_rear_left,self.count_radar_rear_right))
        

    def _make_gt(self,gt_content,ego_content):
        parts = gt_content.split('perception_obstacle {')
        parts = parts[1:]
        for part in parts:
            id            = int(part.split('\n  id: ')[1].split('\n  position {')[0])
            position_x    = float(part.split('\n    x: ')[1].split('\n    y:')[0])
            position_y    = float(part.split('\n    y: ')[1].split('\n    z:')[0])
            position_z    = float(part.split('\n    z: ')[1].split('\n  }\n')[0])
            theta         = float(part.split('\n  theta: ')[1].split('\n  length: ')[0])
            length        = float(part.split('\n  length: ')[1].split('\n  width: ')[0])
            width         = float(part.split('\n  width: ')[1].split('\n  height: ')[0])
            height        = float(part.split('\n  height: ')[1].split('\n  tracking_time: ')[0])
            tracking_time = float(part.split('\n  tracking_time: ')[1].split('\n  type: ')[0])
            sensor_modal  =       part.split('\n  type: ')[1].split('\n  timestamp: ')[0]
            time_stamp    =       part.split('\n  timestamp: ')[1].split('\n}\n')[0]
            # Find the ego-pose base on time_stamp, remember to whether they are 100% match
            # If the time stamp not match, then split ...[1] will raise the out of range error
            # ego_info      = ego_content.split('header {\n  timestamp_sec: '+time_stamp)[1].split('measurement_time: '+time_stamp)[0]
            # ego_pose_x    = float(ego_info.split('position {\n    x: ')[1].split('\n    y: ')[0])
            # ego_pose_y    = float(ego_info.split('\n    y: ')[1].split('\n    z: ')[0])
            # ego_pose_z    = float(ego_info.split('\n    z: ')[1].split('\n  }\n  orientation')[0])
            # ori_qx        = float(ego_info.split('orientation {\n    qx: ')[1].split('\n    qy: ')[0])
            # ori_qy        = float(ego_info.split('\n    qy: ')[1].split('\n    qz: ')[0])
            # ori_qz        = float(ego_info.split('\n    qz: ')[1].split('\n    qw: ')[0])
            # ori_qw        = float(ego_info.split('\n    qw: ')[1].split('\n  }\n')[0])
            self.groudtruth_list.append(gt_data(id,position_x,position_y,position_z,theta,length,width,height,\
                                                int(tracking_time),sensor_modal,float(time_stamp)))
            self.nbr_gt_samples += 1
        print('Number of gt sampples:{}'.format(self.nbr_gt_samples))

    def _generate_object_map(self):
        # Temperoal matching the ground and detections
        # Generate (w(t), w'(t))
        # Iteratively go throught timestamp in ground truth.
        # Pull out the objects from perception module and gt at same timestamp
        # return list of (sensor_data, gt_data)
        # 
        # Get the time stamp list
        time_stamp_list = sorted(list(set([localization_ele.time_stamp for localization_ele in self.ego_vehicle_list])))
        for (idx,frame) in enumerate(time_stamp_list):
            if idx > 0:
                prev_frame = time_stamp_list[idx-1]
            else:
                prev_frame = 0.0
            frame_detection_list = [] # Reuse-able list to make obj_list
            frame_gt_list        = [] # Reuse-able list to make obj_list
            for gt_element in self.groudtruth_list:
                if((gt_element.time_stamp <= frame and gt_element.time_stamp > prev_frame)): 
                    frame_gt_list.append(gt_element)

            for detections_element in self.sensor_data_list:
                if((detections_element.time_stamp <= frame and detections_element.time_stamp > prev_frame)): 
                    frame_detection_list.append(detections_element)
                # if((detections_element.time_stamp < frame+0.01 )and (detections_element.time_stamp > frame-0.01 )): 
                #     frame_detection_list.append(detections_element)

            self.object_map.append([frame,frame_detection_list,frame_gt_list])

    def _make_localization(self,ego_content):
        parts = ego_content.split('pose {')
        filetered_parts = [item for item in parts if not(item.startswith('header'))]
        for part in filetered_parts:
            ego_pose_x = float(part.split('position {\n    x: ')[1].split('\n    y: ')[0])
            ego_pose_y = float(part.split('\n    y: ')[1].split('\n    z: ')[0])
            ego_pose_z = float(part.split('\n    z: ')[1].split('\n  }\n  orientation')[0])
            ori_qx     = float(part.split('orientation {\n    qx: ')[1].split('\n    qy: ')[0])
            ori_qy     = float(part.split('\n    qy: ')[1].split('\n    qz: ')[0])
            ori_qz     = float(part.split('\n    qz: ')[1].split('\n    qw: ')[0])
            ori_qw     = float(part.split('\n    qw: ')[1].split('\n  }\n')[0])
            time_stamp = float(part.split('\nmeasurement_time: ')[1].split('\n')[0])
            self.ego_vehicle_list.append(ego_pose_data(time_stamp,ego_pose_x,ego_pose_y,ego_pose_z,\
                                                       ori_qx,ori_qy,ori_qz,ori_qw))
@dataclass
class sensor_data:
    sensor_id : int
    id        : int
    position_x: float 
    position_y: float 
    position_z: float
    theta     : float
    width     : float
    length    : float 
    height    : float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    type      : str       # Can be use for selecting specific type of sensor to discuss
    time_stamp: float
    box_xmin  : float 
    box_ymin  : float
    box_xmax  : float
    box_ymax  : float

@dataclass
class ego_pose_data:
# Msg. in 'apollo/localization/pose' channel
    time_stamp    : float  # highest time frame ratio
    ego_position_x: float
    ego_position_y: float
    ego_position_z: float
    orientation_qx: float
    orientation_qy: float
    orientation_qz: float
    orientation_qw: float


@dataclass
class gt_data:
    id: int
    position_x : float
    position_y : float
    position_z : float
    theta      : float
    length     : float
    width      : float
    height     : float
    tracking_time: int # This is the bisibility of the obj. in the scene
    type       : str
    time_stamp : float
    # Find same time stamp and declar the ego pose info.
    # base on the data struct of pose, are going to find the info. by:
    # pose {
    #     // Content we want 
    # }
    # measurement_time: xxxxx 
    # ego_position_x: float
    # ego_position_y: float
    # ego_position_z: float
    # orientation_qx: float
    # orientation_qy: float
    # orientation_qz: float
    # orientation_qw: float


if __name__ == '__main__':
    DTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/detection/*_detection*.txt")
    GTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/*_gt*.txt")
    Egofile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/ego/*_gt*_ego-pose.txt")
    scene_list = []
    DTfile_list.sort();GTfile_list.sort();Egofile_list.sort()
    for DT, GT, Ego in zip(DTfile_list,GTfile_list,Egofile_list):
        dataset = dataset_list(DT,GT,Ego)
        scene_list.append(dataset)
        print('Get the dataset of scene token:{}'.format(dataset.scene_token))

    print('FOO')
# Planning:
# for loop, 
# Read all the token name in the directory, 
# in the dataset_list, in the __init___: read in the ego, dt, gt file
#                                        and make_detection, gt, localzation, generate_object_map_object_map
# 