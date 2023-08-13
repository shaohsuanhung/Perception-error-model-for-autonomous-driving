#!/usr/bin/env python3
###############################################################################
#  Authors: HungFrancis. All Rights Reserved
###############################################################################
'''Generate apollo record file by nuscenes raw sensor data.'''
import os
import logging
import numpy as np

from cyber_record.record import Record
from record_msg.builder import (
  ImageBuilder,
  PointCloudBuilder,
  LocalizationBuilder,
  TransformBuilder)
from nuscenes_lib import NuScenesSchema, NuScenesHelper, NuScenes
from geometry import Quaternion
#
from record_msg import pypcd
import time
from modules.drivers.proto import sensor_image_pb2, pointcloud_pb2, conti_radar_pb2
from modules.localization.proto import localization_pb2
from modules.transform.proto import transform_pb2
from nuscenes.utils.data_classes import RadarPointCloud
from protoLib.conti_radar_pb2 import ContiRadar, ContiRadarObs 
from protoLib.perception_obstacle_pb2 import PerceptionObstacles, PerceptionObstacle
import math
from nuscenes.nuscenes import NuScenes as Nuscenes_dataset
#
DATAVERSION = 'v1.0-mini'
DATAROOT = '/home/francis/Desktop/internship/nuScenes_dataset'
LOCALIZATION_TOPIC = '/apollo/localization/pose'
TF_TOPIC= '/tf'
GT = '/apollo/perception/obstacles'
nusc = Nuscenes_dataset(version=DATAVERSION, dataroot=DATAROOT, verbose=True)
pedestrian = ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                      'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
                      'human.pedestrian.wheelchair']
vehicle = ['vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction',
                   'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.motorcycle',
                   'vehicle.trailer', 'vehicle.truck'] 
#

# Need to convert to apollo coordinate system, for nuscenes is 90 degrees
LIDAR_TRANSFORM = np.array([[ 0.0020333, 0.9997041, 0.0242417, 0.9437130],
                            [-0.9999805, 0.0021757,-0.0058486, 0.0000000],
                            [-0.0058997,-0.0242294, 0.9996890, 1.8402300],
                            [ 0.0000000, 0.0000000, 0.0000000, 1.0000000]])

def dataset_to_record(nuscenes, record_root_path):
  """Construct record message and save it as record

  Args:
      nuscenes (_type_): nuscenes(one scene)
      record_root_path (str): record file saved path
  """
  image_builder = ImageBuilder()
  pc_builder = PointCloudBuilder(dim=5)
  pc_builder_extra = PointCloudBuilder_extra(dim=5)
  localization_builder = LocalizationBuilder()
  transform_builder = TransformBuilder()

  record_file_name = "{}.record".format(nuscenes.scene_token)
  record_file_path = os.path.join(record_root_path, record_file_name)

  with Record(record_file_path, mode='w') as record:
    j = 0
    for c, f, ego_pose, calibrated_sensor, t in nuscenes:
      logging.debug("{}, {}, {}, {}".format(c, f, ego_pose, t))
      pb_msg = None
      ##################### Sensor msg #####################
      if c.startswith('CAM'):
        pb_msg = image_builder.build(f, 'camera', 'rgb8', t/1e6)
        channel_name = "/apollo/sensor/camera/{}/image".format(c)
      elif c.startswith('LIDAR'):
        pb_msg = pc_builder.build_nuscenes(f, 'velodyne', t/1e6, LIDAR_TRANSFORM)
        channel_name = "/apollo/sensor/{}/compensator/PointCloud2".format(c)
    
      elif c.startswith('RADAR'):
        # print('Processing Radar')
        pb_msg = pc_builder_extra.build_nuscenes_radar(f,'radar',j,t/1e6)
        channel_name = "/apollo/sensor/{}".format(c)

      if pb_msg:
        record.write(channel_name, pb_msg, t*1000)

      ##################### Posotion msg #######################
      rotation = ego_pose['rotation']
      quat = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
      heading = quat.to_euler().yaw

      ego_pose_t = ego_pose['timestamp']
      pb_msg = localization_builder.build(
        ego_pose['translation'], ego_pose['rotation'], heading, ego_pose_t/1e6)
      if pb_msg:
        record.write(LOCALIZATION_TOPIC, pb_msg, ego_pose_t*1000)

      ##################### location msg #######################
      pb_msg = transform_builder.build('world', 'localization',
        ego_pose['translation'], ego_pose['rotation'], ego_pose_t/1e6)
      if pb_msg:
        record.write(TF_TOPIC, pb_msg, ego_pose_t*1000)
      #
      ##################### Ground Truth #######################
      samples = nuscenes.nuscenes_helper.get_sample_by_scene(nuscenes.scene_token)
      for sample in samples:
        pb_msg = pc_builder_extra.build_nuscenes_gt(nusc.get('sample',sample['token']),'gt',j,t/1e6)
        if pb_msg:
          record.write(GT,pb_msg,t*1000)
      
      j+=1

def convert_dataset(dataset_path, record_path):
  """Generate apollo record file by nuscenes dataset

  Args:
      dataset_path (str): nuscenes dataset path
      record_path (str): record file saved path
  """
  nuscenes_schema = NuScenesSchema(dataroot=dataset_path)
  n_helper = NuScenesHelper(nuscenes_schema)
  # i = 0
  for scene_token in nuscenes_schema.scene.keys():
    # if(i<2): # For testing
    print("Start to convert scene: {}, Pls wait!".format(scene_token))
    nuscenes = NuScenes(n_helper, scene_token)
    dataset_to_record(nuscenes, record_path)
    # i+=1
    # else:
      #  break
  print("Success! Records saved in '{}'".format(record_path))

################ Add Feature, also convert radar information ###############
class Builder_extra(object):
  def __init__(self) -> None:
    self._sequence_num = 0

  def _build_header(self, header,
      t=None, module_name=None, version=None, frame_id=None):
    header.sequence_num = self._sequence_num
    if t:
      header.timestamp_sec = t
      # todo(zero): no need to add?
      # header.camera_timestamp = int(t * 1e9)
      # header.lidar_timestamp = int(t * 1e9)
    if module_name:
      header.module_name = module_name
    if version:
      header.version = version
    if frame_id:
      header.frame_id = frame_id

class PointCloudBuilder_extra(Builder_extra):
  def __init__(self, dim=4) -> None:
    super().__init__()
    self._dim = dim

  def build(self, file_name, frame_id, t=None):
    pb_point_cloud = pointcloud_pb2.PointCloud()

    if t is None:
      t = time.time()

    self._build_header(pb_point_cloud.header, t=t, frame_id=frame_id)
    pb_point_cloud.frame_id = frame_id
    # pb_point_cloud.is_dense = False
    pb_point_cloud.measurement_time = t

    point_cloud = pypcd.point_cloud_from_path(file_name)

    pb_point_cloud.width = point_cloud.width
    pb_point_cloud.height = point_cloud.height

    for data in point_cloud.pc_data:
      point = pb_point_cloud.point.add()
      point.x, point.y, point.z, point.intensity, timestamp = data
      point.timestamp = int(timestamp * 1e9)

    self._sequence_num += 1
    return pb_point_cloud

  def build_nuscenes_radar(self, file_name, frame_id, seq_num ,t=None):
    """
    Example of the header fields:
        # .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
        TYPE F F F I I F F F F F I I I I I I I I
        COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        WIDTH 125
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS 125
        DATA binary
    """
    # pb_point_cloud = pointcloud_pb2.PointCloud()
    msg = conti_radar_pb2.ContiRadar()

    if t is None:
      t = time.time()

    self._build_header(msg.header, t=t, frame_id=frame_id)
    msg.header.frame_id = frame_id
    msg.header.sequence_num = seq_num
    # pb_point_cloud.is_dense = False
    ########################
    # msg.header.sequence_num = t
    #FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    #       0 1 2    3      4  5   6  7    8      9           10            11          12   13          14     15    16     17 
    fradar = RadarPointCloud.from_file(file_name)
    # logging.debug(fradar[:100])
    pointList = list()
    for rows in fradar.points.T:
            #print(rows)
            newPoint = ContiRadarObs()
            newPoint.header.timestamp_sec= msg.header.timestamp_sec    
            newPoint.header.frame_id = "radar"
            newPoint.header.sequence_num = int(t)
            newPoint.clusterortrack = 1
            newPoint.obstacle_id = int(rows[4]) 
            newPoint.longitude_dist = rows[0]
            newPoint.lateral_dist = rows[1]
            newPoint.longitude_vel = rows[6]
            newPoint.lateral_vel = rows[7]
            newPoint.rcs = rows[5]
            newPoint.dynprop = int(rows[3]) # 0 = moving, 1 = stationary, 2 = oncoming, 3 = stationary candidate, 4 = unknown, 5 = crossing stationary, 6 = crossing moving, 7 = stopped TODO use 2-7
            newPoint.longitude_dist_rms = rows[12]
            newPoint.lateral_dist_rms = rows[13]
            newPoint.longitude_vel_rms = rows[16]
            newPoint.lateral_vel_rms = rows[17]
            newPoint.probexist = 1.0 #prob confidence
            pointList.append(newPoint)
    msg.contiobs.extend(pointList)
    return msg
  def build_nuscenes_gt(self,sample,frame_id, sequ_num, t= None):
    msg= PerceptionObstacles()
    # msg.header.timestamp_sec = sample['timestamp']
    obstacleList = list()
    instanceDict = dict()
    IDcount = 0
    #print("----------------------------------------------------------", j )
    # sample = nusc.get('sample_data',sam)
    for ann in sample['anns']:
        sampleAnn = nusc.get('sample_annotation', ann)
        #print(sampleAnn)
        category = 0
        if sampleAnn['category_name'] in pedestrian:
            category = 3
        if sampleAnn['category_name'] in vehicle :
            category = 5
        if category in [3,5]:
            newObs = PerceptionObstacle()
                
            if sampleAnn["instance_token"] not in instanceDict:
                instanceDict[sampleAnn["instance_token"]] = IDcount
                IDcount = IDcount+1
            newObs.id = instanceDict[sampleAnn["instance_token"]]
            #print(sampleAnn["instance_token"], self.instanceDict[sampleAnn["instance_token"]])
            x, y, z = sampleAnn["translation"]
            newObs.position.x,newObs.position.y, newObs.position.z = y,-x,z
            X, Y, Z = quaternion_to_euler(sampleAnn["rotation"])
            newObs.tracking_time = float(sampleAnn["visibility_token"])
            newObs.theta =math.radians(-X+90)
            x, y, z = sampleAnn["size"]
            newObs.length =y
            newObs.width =z
            newObs.height= z
            #newObs.velocity
                
            newObs.type = category
            newObs.timestamp =0 
            #i=i+1
            obstacleList.append(newObs)
    msg.perception_obstacle.extend(obstacleList)
    return msg



def quaternion_to_euler(quat):
        x, y, z, w = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z