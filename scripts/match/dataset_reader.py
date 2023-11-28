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
# Important things to mention: In my case, the tab in the text file is 2 spaces, not 4 spaces
# Please check the tab in the text file before running this script
from dataclasses import dataclass
import matplotlib.pyplot as plt
import glob
import numpy as np
import math
from matplotlib.patches import Rectangle as Rec
# Dataset analyzer
# Output number of samples:
# lidar:
# radars:
import dataclasses, json

class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)



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
        self.groundtruth_list = []
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
        self._make_localization(ego_content)
        ########## Select either line here if want to use fusion mode ##########
        self._make_detection(dt_content)
        # self._make_fusion_detection(dt_content)
        ############################################################
        self._make_gt(gt_content,ego_content)
        # Interpolated the ego pose
        self._interpolated_ego_pose()
        self._filter_modality('radar')
        # self._filter_modality('velodyne32')
        self._rotate_detection()
        self._generate_object_map()
        ###################################
        ###################################

    def _filter_modality(self,modality):
        ''' Given the str of modality, get that modality from the sensor_data_list'''
        # Filter out the modality
        print("Before filtering:{}".format(len(self.sensor_data_list)))
        self.sensor_data_list = [obj for obj in self.sensor_data_list if obj.sensor_id.startswith(modality)]
        print("After filtering:{},sensor modal:{}".format(len(self.sensor_data_list),self.sensor_data_list[0].sensor_id))
        # print("-------------- Number of sample after filter {} only---------------".format(modality))    
        print("Scenes token:{}".format(self.scene_token))
        print("Radar_front:{}\nRadar_front_left:{}\nRadar_front_right:{}\nRadar_rear_left:{}\nRadar_rear_right:{}".\
          format(self.count_radar_front,self.count_radar_front_left,\
          self.count_radar_front_right,self.count_radar_rear_left,self.count_radar_rear_right))

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

        else:
            # print("Fusion mode")
            pass
 
    def _interpolated_ego_pose(self):
        # Some timestamp in the sensor is not in the ego pose, so interpolated the ego pose
        i = 0
        for detection in self.sensor_data_list:
            idxCount = 0
            
            Ego_pose = [ego for ego in self.ego_vehicle_list if ego.time_stamp == detection.time_stamp]
            if Ego_pose != []:
                continue
            else: # The time stamp is not in the ego pose list, need to interpolated
                i+=1
                while self.ego_vehicle_list[idxCount].time_stamp < detection.time_stamp:
                    ego_prev_idx = idxCount
                    idxCount+=1
                ego_next_idx = idxCount
                # Interpolate the ego pose
                ratio = (detection.time_stamp-self.ego_vehicle_list[ego_prev_idx].time_stamp)/(self.ego_vehicle_list[ego_next_idx].time_stamp-self.ego_vehicle_list[ego_prev_idx].time_stamp)
                timestamp = detection.time_stamp
                # timestamp = self.ego_vehicle_list[ego_prev_idx].time_stamp * \
                #     (1-ratio)+ratio*self.ego_vehicle_list[ego_next_idx].time_stamp
                position_x = self.ego_vehicle_list[ego_prev_idx].ego_position_x * \
                    (1-ratio)+ratio*self.ego_vehicle_list[ego_next_idx].ego_position_x
                position_y = self.ego_vehicle_list[ego_prev_idx].ego_position_y * \
                    (1-ratio)+ratio*self.ego_vehicle_list[ego_next_idx].ego_position_y
                position_z = self.ego_vehicle_list[ego_prev_idx].ego_position_z * \
                    (1-ratio)+ratio*self.ego_vehicle_list[ego_next_idx].ego_position_z
                orientation_qx = self.ego_vehicle_list[ego_prev_idx].orientation_qx * \
                    (1-ratio)+ratio*self.ego_vehicle_list[ego_next_idx].orientation_qx
                orientation_qy = self.ego_vehicle_list[ego_prev_idx].orientation_qy * \
                    (1-ratio)+ratio*self.ego_vehicle_list[ego_next_idx].orientation_qy
                orientation_qz = self.ego_vehicle_list[ego_prev_idx].orientation_qz * \
                    (1-ratio)+ratio*self.ego_vehicle_list[ego_next_idx].orientation_qz
                orientation_qw = self.ego_vehicle_list[ego_prev_idx].orientation_qw * \
                    (1-ratio)+ratio*self.ego_vehicle_list[ego_next_idx].orientation_qw
                # Insert the interpolated ego pose in the original ego pose list 
                self.ego_vehicle_list.insert(ego_next_idx,ego_pose_data(timestamp,position_x,position_y,position_z,\
                                                        orientation_qx,orientation_qy,orientation_qz,orientation_qw))
                # print("Inserted at timestamp:{}".format(timestamp))
        print("Interpolated {} times".format(i))
    def _make_detection(self,file_content):
        # Make detection from different sensor
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
        
    def _make_fusion_detection(self,file_content):
        # Use regular expressions to split the file_contents based on "Separator:" lines
        parts = file_content.split('perception_obstacle {')
        # parts = [part.strip() for part in parts]
        filetered_parts = [item for item in parts if not(item.startswith('header'))]
        for part in filetered_parts:
            measurements = part.split('measurements {')
            modality = 'fusion_'
            for multi_sensor_measurement in measurements:
                if multi_sensor_measurement.startswith('\n    sensor_id: '):
                    sensor_id  = multi_sensor_measurement.split('"')[1].split('"')[0]
                    modality += sensor_id
        for part in filetered_parts:
            id            = int(part.split('\n  id: ')[1].split('\n  position {')[0])
            position_x    = float(part.split('\n    x: ')[1].split('\n    y:')[0])
            position_y    = float(part.split('\n    y: ')[1].split('\n    z:')[0])
            position_z    = float(part.split('\n    z: ')[1].split('\n  }\n')[0])
            theta         = float(part.split('\n  theta: ')[1].split('\n  velocity {')[0])
            # theta         = float(part.split('\n  theta: ')[1].split('\n')[0])
            length        = float(part.split('\n  length: ')[1].split('\n  width: ')[0])
            width         = float(part.split('\n  width: ')[1].split('\n  height: ')[0])
            height        = float(part.split('\n  height: ')[1].split('\n  polygon_point {')[0])
            # height        = float(part.split('\n  height: ')[1].split('\n')[0])
            tracking_time = float(part.split('\n  tracking_time: ')[1].split('\n  type: ')[0])
            type          =       part.split('\n  type: ')[1].split('\n  timestamp: ')[0]
            time_stamp    =       part.split('\n  timestamp: ')[1].split('\n')[0]
            self.append_sensor_data(sensor_data(modality,id,position_x,position_y,position_z,theta,length,width,height,\
                                                0,0,0,type,float(time_stamp),0,0,0,0))

        
    def _rotate_detection(self):
        # Rotate the detection by 90 degree c.c.w. w.r.t the ego vehicle(90 deg on the polar coordinate)
        # 1. Convert to the local coordinate w.r.t ego vehicle
        # 2. Rotate
        # 3. Convert back to the gloal cooridnate    
        ROTATE90 = np.array([[ 0.0000000, -1.0000000, 0.0000000],
                             [ 1.0000000,  0.0000000, 0.0000000],
                             [ 0.0000000,  0.0000000, 1.0000000]])
        # ROTATE90 = np.array([[ 0.0000000, 1.0000000, 0.0000000],
        #                      [ -1.0000000,  0.0000000, 0.0000000],
        #                      [ 0.0000000,  0.0000000, 1.0000000]])
        
        for detection in self.sensor_data_list:
            DT_position = np.array([detection.position_x,detection.position_y,detection.position_z])
            Ego_pose = [ego for ego in self.ego_vehicle_list if ego.time_stamp == detection.time_stamp]
            Ego_pose = Ego_pose[0]
            Ego_position = np.array([Ego_pose.ego_position_x,Ego_pose.ego_position_y,Ego_pose.ego_position_z])
            Ego_theta = Quaternion(Ego_pose.orientation_qw,Ego_pose.orientation_qx,Ego_pose.orientation_qy,Ego_pose.orientation_qz).to_euler().yaw
            DTLocalPose = np.array([DT_position[0]-Ego_position[0],DT_position[1]-Ego_position[1],DT_position[2]-Ego_position[2]])
            ##  ----------------------------- To  verifity --------------------------------------------------
            # print("Before rotation")
            # print("x:{},y:{},z:{}\n".format(detection.position_x, detection.position_y, detection.position_z))
            # print("Vx:{},Vy:{},Vz:{}\n".format(detection.velocity_x, detection.velocity_y, detection.velocity_z))
            # print("theta:{}".format(detection.theta))
            ##  ----------------------------- End of verifity --------------------------------------------------
            RotatedDTLocalPose= np.matmul(ROTATE90, DTLocalPose)
            [detection.position_x,detection.position_y,detection.position_z] = [RotatedDTLocalPose[0]+Ego_position[0],RotatedDTLocalPose[1]+Ego_position[1],RotatedDTLocalPose[2]+Ego_position[2]]
            detection.theta = detection.theta + math.pi/2
            # detection.theta = detection.theta - math.pi/2
            # detection.theta = 2*Ego_theta - detection.theta - math.pi/2 # Looks match bettewe
            # detection.theta = detection.theta + (detection.theta - Ego_theta) - math.pi/2
            ##  ----------------------------- To  verifity --------------------------------------------------
            # print("After rotation")
            # print("x:{},y:{},z:{}\n".format(detection.position_x, detection.position_y, detection.position_z))
            # print("Vx:{},Vy:{},Vz:{}\n".format(detection.velocity_x, detection.velocity_y, detection.velocity_z))
            # print("theta:{}".format(detection.theta))
            ##  ----------------------------- End of verifity --------------------------------------------------
    def _make_gt(self,gt_content,ego_content):
        parts = gt_content.split('perception_obstacle {')
        parts = parts[1:]
        for part in parts:
            id            = int(part.split('\n  id: ')[1].split('\n  position {')[0])
            position_x    = float(part.split('\n    x: ')[1].split('\n    y:')[0])
            position_y    = float(part.split('\n    y: ')[1].split('\n    z:')[0])
            position_z    = float(part.split('\n    z: ')[1].split('\n  }\n')[0])
            theta         = float(part.split('\n  theta: ')[1].split('\n  length: ')[0])
            # theta         = float(part.split('\n  theta: ')[1].split('\n')[0])
            length        = float(part.split('\n  length: ')[1].split('\n  width: ')[0])
            width         = float(part.split('\n  width: ')[1].split('\n  height: ')[0])
            height        = float(part.split('\n  height: ')[1].split('\n  tracking_time: ')[0])
            # height        = float(part.split('\n  height: ')[1].split('\n')[0])
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
            self.groundtruth_list.append(gt_data(id,position_x,position_y,position_z,theta,length,width,height,\
                                                int(tracking_time),sensor_modal,float(time_stamp)))
            self.nbr_gt_samples += 1
        print('Number of gt samples:{}'.format(self.nbr_gt_samples))

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
                prev_frame = 0.0 # the first frame
            frame_detection_list = [] # Reuse-able list to make obj_list
            frame_gt_list        = [] # Reuse-able list to make obj_list
            for gt_element in self.groundtruth_list:
                if((gt_element.time_stamp < frame and gt_element.time_stamp >= prev_frame)): 
                    frame_gt_list.append(gt_element)

            for detections_element in self.sensor_data_list:
                if((detections_element.time_stamp < frame and detections_element.time_stamp >= prev_frame)): 
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
            
    def render_object_map(self):
        i = 0
        # For debug use, to visualize the DT,GT,Ego w.r.t time stamp
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for (timestamp,DT_list,GT_list) in self.object_map:
           
            # Plot only when GT exist
            if GT_list!=[]:
                for (idx,obj) in enumerate(GT_list):
                        # print("HIT")
                        plt.plot(obj.position_x, obj.position_y, 'bo',label='GT' if idx == 0 else "",markersize=5) 
                        ax.add_patch(Rec(xy=(obj.position_x-(obj.width/2),obj.position_y-(obj.length/2)),width=obj.width,height=obj.length,angle=obj.theta*180/math.pi,color='blue',rotation_point='center',fill=False))
                        plt.arrow(obj.position_x,obj.position_y,5*math.cos(obj.theta),5*math.sin(obj.theta),width=0.1,color='blue')
            else:
                continue   
            # Plot DT
            if DT_list!=[]:
                for (idx,obj) in enumerate(DT_list):
                    plt.plot(obj.position_x, obj.position_y, 'ro',label = 'DT' if idx == 0 else "",markersize=5)
                    ax.add_patch(Rec(xy=(obj.position_x-(obj.width/2),obj.position_y-(obj.length/2)),width=obj.width,height=obj.length,angle=obj.theta*180/math.pi,color='red',rotation_point='center',fill=False))
                    plt.arrow(obj.position_x,obj.position_y,5*math.cos(obj.theta),5*math.sin(obj.theta),width=0.1,color='red')
            
            egoPose = [ego for ego in self.ego_vehicle_list if ego.time_stamp == timestamp]
            egoPose = egoPose[0]           
            # Plot eog_pose
            plt.plot(egoPose.ego_position_x, egoPose.ego_position_y, 'k*',label='Ego Pose',markersize=5)
            Angle = Quaternion(egoPose.orientation_qw,egoPose.orientation_qx,egoPose.orientation_qy,egoPose.orientation_qz).to_euler().yaw
            plt.arrow(egoPose.ego_position_x,egoPose.ego_position_y,5*math.cos(Angle),5*math.sin(Angle),width=0.1,color='blue')
            plt.legend()
            plt.title('Scene token:{}\nTime:{}'.format(self.scene_token,timestamp))
            # plt.xlim(330,440)
            # plt.ylim(1080,1220)
            plt.savefig('./render_frame/in_reader/'+str(i))
            i+=1
            # if i==38:
            #     print("fpp")
            plt.close()

    def render_DT(self):
        i = 0
        # For debug use, to visualize the DT,GT,Ego w.r.t time stamp
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for egoPose in self.ego_vehicle_list:
            DT = [obj for obj in self.sensor_data_list if obj.time_stamp == egoPose.time_stamp]
            if DT!=[]:
                for (idx,obj) in enumerate(DT):
                    plt.plot(obj.position_x, obj.position_y, 'ro',label = 'DT' if idx == 0 else "",markersize=5)
                    ax.add_patch(Rec(xy=(obj.position_x-(obj.width/2),obj.position_y-(obj.length/2)),width=obj.width*10,height=obj.length*10,angle=obj.theta*180/math.pi,color='red',rotation_point='center',fill=False))
                    plt.arrow(obj.position_x,obj.position_y,5*math.cos(obj.theta),5*math.sin(obj.theta),width=0.1,color='red')
                
                plt.plot(egoPose.ego_position_x, egoPose.ego_position_y, 'k*',label='Ego Pose',markersize=5)
                Angle = Quaternion(egoPose.orientation_qw,egoPose.orientation_qx,egoPose.orientation_qy,egoPose.orientation_qz).to_euler().yaw
                plt.arrow(egoPose.ego_position_x,egoPose.ego_position_y,5*math.cos(Angle),5*math.sin(Angle),width=0.1,color='blue')
            
            else:
                continue        
            # plt.xlim(600,700)
            # plt.ylim(1580,1680)
            plt.savefig('./render_frame/in_reader/DT/'+str(i))
            i+=1
            plt.close()

    def rendering_one_obj_tracking(self,assigned_id=19):
        # Track the GT obj with assigned id
        fig = plt.figure()
        ax = fig.add_subplot(111)
        i = 0 
        for obj in self.groundtruth_list:
            if obj.id == assigned_id:
                # print("-"*50)
                # print("Object id:{}".format(obj.id))
                # print("Object type:{}".format(obj.type))
                # print("Object time stamp:{}".format(obj.time_stamp))
                # print("Object position_x:{}".format(obj.position_x))
                # print("Object position_y:{}".format(obj.position_y))
                # print("Object position_z:{}".format(obj.position_z))
                # print("Object theta:{}".format(obj.theta))
                # print("-"*50)
                plt.plot(obj.position_x, obj.position_y, 'bo')
                ax.add_patch(Rec(xy=(obj.position_x-(obj.width/2),obj.position_y-(obj.length/2)),width=obj.width,height=obj.length,angle=obj.theta*180/math.pi,color='blue',rotation_point='center',fill=False))
                plt.arrow(obj.position_x,obj.position_y,5*math.cos(obj.theta),5*math.sin(obj.theta),width=0.1,color='blue')
                egoPose = [ego for ego in self.ego_vehicle_list if ego.time_stamp == obj.time_stamp]
                egoPose = egoPose[0] 
                plt.plot(egoPose.ego_position_x, egoPose.ego_position_y, 'k*',label='Ego Pose',markersize=5)
                Angle = Quaternion(egoPose.orientation_qw,egoPose.orientation_qx,egoPose.orientation_qy,egoPose.orientation_qz).to_euler().yaw
                plt.arrow(egoPose.ego_position_x,egoPose.ego_position_y,5*math.cos(Angle),5*math.sin(Angle),width=0.1,color='black')
                plt.xlim(330,470)
                plt.ylim(1040,1250)
                plt.savefig('./render_frame/in_reader/'+str(i))
                i+=1
                plt.close()

class Euler(object):
  def __init__(self, roll, pitch, yaw) -> None:
    self.roll = roll
    self.pitch = pitch
    self.yaw = yaw

  def to_quaternion(self):
    cr = math.cos(self.roll * 0.5)
    sr = math.sin(self.roll * 0.5)
    cp = math.cos(self.pitch * 0.5)
    sp = math.sin(self.pitch * 0.5)
    cy = math.cos(self.yaw * 0.5)
    sy = math.sin(self.yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return Quaternion(qw, qx, qy, qz)
  
class Quaternion(object):
  def __init__(self, w, x, y, z) -> None:
    self.w = w
    self.x = x
    self.y = y
    self.z = z

  def to_euler(self):
    t0 = 2 * (self.w * self.x + self.y * self.z)
    t1 = 1 - 2 * (self.x * self.x + self.y * self.y)
    roll = math.atan2(t0, t1)

    t2 = 2 * (self.w * self.y - self.z * self.x)
    pitch = math.asin(t2)

    t3 = 2 * (self.w * self.z + self.x * self.y)
    t4 = 1 - 2 * (self.y * self.y + self.z * self.z)
    yaw = math.atan2(t3, t4)
    return Euler(roll, pitch, yaw)

  def __mul__(self, other):
    w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
    x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
    y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
    z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
    return Quaternion(w, x, y, z)

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
    # DTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/detection/*_detection*.txt")
    # GTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/*_gt*.txt")
    # Egofile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/ego/*_gt*_ego-pose.txt")
    token = '0ced08ea43754420a23b2fbec667a763'
    weather = 'rain'
    DTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/detection/{}_detection_{}.txt".format(token,weather))
    GTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/{}_gt_{}.txt".format(token,weather))
    Egofile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/ego/{}_gt_{}_ego-pose.txt".format(token,weather))
    
    scene_list = []
    DTfile_list.sort();GTfile_list.sort();Egofile_list.sort()
    for DT, GT, Ego in zip(DTfile_list,GTfile_list,Egofile_list):
        dataset = dataset_list(DT,GT,Ego)
        scene_list.append(dataset)
        # dataset.rendering_one_obj_tracking(42)
        # cc8 ID 42: the vehcile in front of ego
        # ID: 72, 74, 78, 
        # dataset.render_DT()
        # dataset.render_object_map()
        print('Get the dataset of scene token:{}'.format(dataset.scene_token))

# Planning:
# for loop, 
# Read all the token name in the directory, 
# in the dataset_list, in the __init___: read in the ego, dt, gt file
#                                        and make_detection, gt, localzation, generate_object_map_object_map
# 