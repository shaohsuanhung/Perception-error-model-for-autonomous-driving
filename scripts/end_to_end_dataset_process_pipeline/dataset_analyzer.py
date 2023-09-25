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
    def __init__(self,scene_token):
        self.scene_token = scene_token
        self.sensor_data_list = []
        self.groudtruth_list = []
        # Dim: [[detection_list_at_t1,gt_list_at_t1],[detection_list_at_t2,gt_list_at_t2].....\
        #                     [detection_list_at_tn, gt_list_at_tn]]
        self.object_map = []
        self.count_lidar32 = 0
        self.count_radar_front = 0
        self.count_radar_front_left = 0
        self.count_radar_front_right = 0
        self.count_radar_rear_left = 0
        self.count_radar_rear_right = 0
        # Number of gt samples
        self.nbr_gt_samples =0

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

    def make_detection(self,file_content):
        # Use regular expressions to split the file_contents based on "Separator:" lines
        parts = file_content.split('perception_obstacle {')
        # parts = [part.strip() for part in parts]
        filetered_parts = [item for item in parts if not(item.startswith('header'))]
        for part in filetered_parts:
            measurements = part.split('measurements {')
            for multi_sensor_measurement in measurements:
                if multi_sensor_measurement.startswith('\n    sensor_id: '):
                    sensor_id = multi_sensor_measurement.split('"')[1].split('"')[0]
                    id = multi_sensor_measurement.split(' id: ')[1].split('\n')[0]
                    position_x = multi_sensor_measurement.split('\n      x: ')[1].split('\n      y:')[0]
                    position_y = multi_sensor_measurement.split('\n      y: ')[1].split('\n      z:')[0]
                    position_z = multi_sensor_measurement.split('\n      z: ')[1].split('\n    }\n')[0]
                    theta      = multi_sensor_measurement.split('\n    theta: ')[1].split('\n    length:')[0]
                    length     = multi_sensor_measurement.split('\n    length: ')[1].split('\n    width:')[0]
                    width      = multi_sensor_measurement.split('\n    width: ')[1].split('\n    height:')[0]
                    height     = multi_sensor_measurement.split('\n    height: ')[1].split('\n    velocity {')[0]
                    velocity_x = multi_sensor_measurement.split('\n      x: ')[1].split('\n      y: ')[0]
                    velocity_y = multi_sensor_measurement.split('\n      y: ')[1].split('\n      z: ')[0]
                    velocity_z = multi_sensor_measurement.split('\n      z: ')[1].split('\n    }\n')[0]
                    type       = multi_sensor_measurement.split('\n    type: ')[1].split('\n    timestamp: ')[0]
                    time_stamp = multi_sensor_measurement.split('\n    timestamp: ')[1].split('\n    box {')[0]
                    xmin       = multi_sensor_measurement.split('\n      xmin: ')[1].split('\n      ymin: ')[0] 
                    ymin       = multi_sensor_measurement.split('\n      ymin: ')[1].split('\n      xmax: ')[0] 
                    xmax       = multi_sensor_measurement.split('\n      xmax: ')[1].split('\n      ymax: ')[0]
                    ymax       = multi_sensor_measurement.split('\n      ymax: ')[1].split('\n    }\n')[0]
                    self.append_sensor_data(sensor_data(sensor_id, id,position_x, position_y, position_z, \
                                                        theta, length, width,height, velocity_x, velocity_y, \
                                                        velocity_z, type,float(time_stamp),xmin,ymin,xmax,ymax))
        print("-------------- Number of sample ---------------")
        print("Scenes token:{}".format(self.scene_token))
        print("Lidar:{}\nRadar_front:{}\nRadar_front_left:{}\nRadar_front_right:{}\nRadar_rear_left:{}\nRadar_rear_right:{}".\
          format(self.count_lidar32,self.count_radar_front,self.count_radar_front_left,\
          self.count_radar_front_right,self.count_radar_rear_left,self.count_radar_rear_right))
        

    def make_gt(self,gt_content,ego_content):
        parts = gt_content.split('perception_obstacle {')
        parts = parts[1:]
        for part in parts:
            id = part.split('\n  id: ')[1].split('\n  position {')[0]
            position_x = part.split('\n    x: ')[1].split('\n    y:')[0]
            position_y = part.split('\n    y: ')[1].split('\n    z:')[0]
            position_z = part.split('\n    z: ')[1].split('\n  }\n')[0]
            theta      = part.split('\n  theta: ')[1].split('\n  length: ')[0]
            length     = part.split('\n  length: ')[1].split('\n  width: ')[0]
            width      = part.split('\n  width: ')[1].split('\n  height: ')[0]
            height     = part.split('\n  height: ')[1].split('\n  tracking_time: ')[0]
            tracking_time = part.split('\n  tracking_time: ')[1].split('\n  type: ')[0]
            type       = part.split('\n  type: ')[1].split('\n  timestamp: ')[0]
            time_stamp = part.split('\n  timestamp: ')[1].split('\n}\n')[0]
            # Find the ego-pose base on time_stamp, remember to whether they are 100% match
            # If the time stamp not match, then split ...[1] will raise the out of range error
            ego_info = ego_content.split('header {\n  timestamp_sec: '+time_stamp)[1].split('measurement_time: '+time_stamp)[0]
            ego_pose_x = ego_info.split('position {\n    x: ')[1].split('\n    y: ')[0]
            ego_pose_y = ego_info.split('\n    y: ')[1].split('\n    z: ')[0]
            ego_pose_z = ego_info.split('\n    z: ')[1].split('\n  }\n  orientation')[0]
            ori_qx     = ego_info.split('orientation {\n    qx: ')[1].split('\n    qy: ')[0]
            ori_qy     = ego_info.split('\n    qy: ')[1].split('\n    qz: ')[0]
            ori_qz     = ego_info.split('\n    qz: ')[1].split('\n    qw: ')[0]
            ori_qw     = ego_info.split('\n    qw: ')[1].split('\n  }\n')[0]
            self.groudtruth_list.append(gt_data(id,position_x,position_y,position_z,theta,length,width,height\
                                   ,tracking_time,type,float(time_stamp),ego_pose_x,ego_pose_y,ego_pose_z,\
                                    ori_qx,ori_qy,ori_qz,ori_qw))
            self.nbr_gt_samples += 1
        print('Number of gt sampples:{}'.format(self.nbr_gt_samples))

    def generate_object_map(self):
        # Temperoal matching the ground and detections
        # Generate (w(t), w'(t))
        # Iteratively go throught timestamp in ground truth.
        # Pull out the objects from perception module and gt at same timestamp
        # return list of (sensor_data, gt_data)
        # 
        # Get the time stamp list
        time_stamp_list = sorted(list(set([gt_element.time_stamp for gt_element in self.groudtruth_list])))
        for frame in time_stamp_list:
            frame_detection_list = [] # Reuse-able list to make obj_list
            frame_gt_list        = [] # Reuse-able list to make obj_list
            for gt_element in self.groudtruth_list:
                if(gt_element.time_stamp == frame): 
                    frame_gt_list.append(gt_element)

            for detections_element in self.sensor_data_list:
                if(detections_element.time_stamp == frame): 
                    frame_detection_list.append(detections_element)
                # if((detections_element.time_stamp < frame+0.01 )and (detections_element.time_stamp > frame-0.01 )): 
                #     frame_detection_list.append(detections_element)

            self.object_map.append([frame,frame_detection_list,frame_gt_list])

    def make_localization(self,ego_conten):
        parts = file_content.split('pose {')
        filetered_parts = [item for item in parts if not(item.startswith('header'))]
        for part in filetered_parts:
            ego_pose_x = part.split('position {\n    x: ')[1].split('\n    y: ')[0]
            ego_pose_y = part.split('\n    y: ')[1].split('\n    z: ')[0]
            ego_pose_z = part.split('\n    z: ')[1].split('\n  }\n  orientation')[0]
            ori_qx     = part.split('orientation {\n    qx: ')[1].split('\n    qy: ')[0]
            ori_qy     = part.split('\n    qy: ')[1].split('\n    qz: ')[0]
            ori_qz     = part.split('\n    qz: ')[1].split('\n    qw: ')[0]
            ori_qw     = part.split('\n    qw: ')[1].split('\n  }\n')[0]
            self.ego_vehicle_list.append(ego_pose_data(ego_pose_x,ego_pose_y,ego_pose_z,\
                                                       ori_qx,ori_qy,ori_qz,ori_qw))
@dataclass
class sensor_data:
    sensor_id : str
    id        : str
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
    type      : str
    time_stamp: float
    box_xmin  : float 
    box_ymin  : float
    box_xmax  : float
    box_ymax  : float

@dataclass
class ego_pose_data:
# Msg. in 'apollo/localization/pose' channel
    ego_position_x: float
    ego_position_y: float
    ego_position_z: float
    orientation_qx: float
    orientation_qy: float
    orientation_qz: float
    orientation_qw: float


@dataclass
class gt_data:
    id: str
    position_x : float
    position_y : float
    position_z : float
    theta      : float
    length     : float
    width      : float
    height     : float
    tracking_time: float
    type       : str
    time_stamp : float
    # Find same time stamp and declar the ego pose info.
    # base on the data struct of pose, are going to find the info. by:
    # pose {
    #     // Content we want 
    # }
    # measurement_time: xxxxx 
    ego_position_x: float
    ego_position_y: float
    ego_position_z: float
    orientation_qx: float
    orientation_qy: float
    orientation_qz: float
    orientation_qw: float


if __name__ == '__main__':
    ###########################################################################
    # 1
    cc8_detection_filename = './text_dataset/detection/cc8c0bf57f984915a77078b10eb33198_detection.txt'
    cc8_gt_filename = './text_dataset/ground_truth/cc8c0bf57f984915a77078b10eb33198_gt.txt'
    cc8_gt_ego =  './text_dataset/ground_truth/cc8c0bf57f984915a77078b10eb33198_gt_ego-pose.txt'
    cc8_scene_token = cc8_gt_filename[-39:-7]
    cc8_dataset = dataset_list(cc8_scene_token)

    # 2
    fc_detection_filename = './text_dataset/detection/fcbccedd61424f1b85dcbf8f897f9754_detection.txt'
    fc_gt_filename = './text_dataset/ground_truth/fcbccedd61424f1b85dcbf8f897f9754_gt.txt'
    fc_gt_ego =  './text_dataset/ground_truth/fcbccedd61424f1b85dcbf8f897f9754_gt_ego-pose.txt'
    fc_scene_token = fc_gt_filename[-39:-7]
    fc_dataset = dataset_list(fc_scene_token)
    # 3
    f6_detection_filename = './text_dataset/detection/6f83169d067343658251f72e1dd17dbc_detection.txt'
    f6_gt_filename = './text_dataset/ground_truth/6f83169d067343658251f72e1dd17dbc_gt.txt'
    f6_gt_ego =  './text_dataset/ground_truth/6f83169d067343658251f72e1dd17dbc_gt_ego-pose.txt'
    f6_scene_token = f6_gt_filename[-39:-7]
    f6_dataset = dataset_list(f6_scene_token)
    
    # 4
    fc375_detection_filename = './text_dataset/detection/2fc3753772e241f2ab2cd16a784cc680_detection.txt'
    fc375_gt_filename = './text_dataset/ground_truth/2fc3753772e241f2ab2cd16a784cc680_gt.txt'
    fc375_gt_ego =  './text_dataset/ground_truth/2fc3753772e241f2ab2cd16a784cc680_gt_ego-pose.txt'
    fc375_scene_token = fc375_gt_filename[-39:-7]
    fc375_dataset = dataset_list(fc375_scene_token)

    # 5
    cef682_detection_filename = './text_dataset/detection/325cef682f064c55a255f2625c533b75_detection.txt'
    cef682_gt_filename = './text_dataset/ground_truth/325cef682f064c55a255f2625c533b75_gt.txt'
    cef682_gt_ego =  './text_dataset/ground_truth/325cef682f064c55a255f2625c533b75_gt_ego-pose.txt'
    cef682_scene_token = cef682_gt_filename[-39:-7]
    cef682_dataset = dataset_list(cef682_scene_token)

    # 6
    bebf_detection_filename = './text_dataset/detection/bebf5f5b2a674631ab5c88fd1aa9e87a_detection.txt'
    bebf_gt_filename = './text_dataset/ground_truth/bebf5f5b2a674631ab5c88fd1aa9e87a_gt.txt'
    bebf_gt_ego =  './text_dataset/ground_truth/bebf5f5b2a674631ab5c88fd1aa9e87a_gt_ego-pose.txt'
    bebf_scene_token = bebf_gt_filename[-39:-7]
    bebf_dataset = dataset_list(bebf_scene_token)

    # 7
    c5224_detection_filename = './text_dataset/detection/c5224b9b454b4ded9b5d2d2634bbda8a_detection.txt'
    c5224_gt_filename = './text_dataset/ground_truth/c5224b9b454b4ded9b5d2d2634bbda8a_gt.txt'
    c5224_gt_ego =  './text_dataset/ground_truth/c5224b9b454b4ded9b5d2d2634bbda8a_gt_ego-pose.txt'
    c5224_scene_token = c5224_gt_filename[-39:-7]
    c5224_dataset = dataset_list(c5224_scene_token)

    # 8
    d257_detection_filename = './text_dataset/detection/d25718445d89453381c659b9c8734939_detection.txt'
    d257_gt_filename = './text_dataset/ground_truth/d25718445d89453381c659b9c8734939_gt.txt'
    d257_gt_ego =  './text_dataset/ground_truth/d25718445d89453381c659b9c8734939_gt_ego-pose.txt'
    d257_scene_token = d257_gt_filename[-39:-7]
    d257_dataset = dataset_list(d257_scene_token)

    # 9
    de7_detection_filename = './text_dataset/detection/de7d80a1f5fb4c3e82ce8a4f213b450a_detection.txt'
    de7_gt_filename = './text_dataset/ground_truth/de7d80a1f5fb4c3e82ce8a4f213b450a_gt.txt'
    de7_gt_ego =  './text_dataset/ground_truth/de7d80a1f5fb4c3e82ce8a4f213b450a_gt_ego-pose.txt'
    de7_scene_token = de7_gt_filename[-39:-7]
    de7_dataset = dataset_list(de7_scene_token) 

    # 10
    e233_detection_filename = './text_dataset/detection/e233467e827140efa4b42d2b4c435855_detection.txt'
    e233_gt_filename = './text_dataset/ground_truth/e233467e827140efa4b42d2b4c435855_gt.txt'
    e233_gt_ego =  './text_dataset/ground_truth/e233467e827140efa4b42d2b4c435855_gt_ego-pose.txt'
    e233_scene_token = e233_detection_filename[-39:-7]
    e233_dataset = dataset_list(e233_scene_token)
    # make them to list 
    datasets = [[cc8_detection_filename,cc8_gt_filename,cc8_gt_ego,cc8_dataset]\
                ,[fc_detection_filename,fc_gt_filename,fc_gt_ego,fc_dataset]\
                ,[f6_detection_filename,f6_gt_filename,f6_gt_ego,f6_dataset]\
                ,[fc375_detection_filename,fc375_gt_filename,fc375_gt_ego,fc375_dataset]\
                ,[cef682_detection_filename,cef682_gt_filename,cef682_gt_ego,cef682_dataset]\
                ,[bebf_detection_filename,bebf_gt_filename,bebf_gt_ego,bebf_dataset]\
                ,[c5224_detection_filename,c5224_gt_filename,c5224_gt_ego,c5224_dataset]\
                ,[d257_detection_filename,d257_gt_filename,d257_gt_ego,d257_dataset]\
                ,[de7_detection_filename,de7_gt_filename,de7_gt_ego,de7_dataset]\
                ,[e233_detection_filename,e233_gt_filename,e233_gt_ego,e233_dataset]]
    #############################################################################
    for (detection_filename, gt_filename,gt_ego_pose, gt_dataset) in datasets:
        with open(detection_filename, 'r') as file:
            file_content = file.read()

        with open(gt_filename,'r') as file:
            gt_content = file.read()

        with open(gt_ego_pose,'r') as file:
            gt_ego_content = file.read()

        gt_dataset.make_detection(file_content)
        gt_dataset.make_gt(gt_content,gt_ego_content)
        gt_dataset.generate_object_map()
        gt_dataset.make_localization(gt_ego_content)
        print("-----------------------------------------------")
        # dataset_analyzer(gt_dataset)
        
        