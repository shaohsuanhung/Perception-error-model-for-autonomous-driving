#!/usr/bin/env python3
###############################################################################
#  Authors: HungFrancis. All Rights Reserved
###############################################################################
'''NuScenes pcd file to pcl pcd file converter.'''

import logging
import numpy as np

from record_msg import pypcd

def convert_pcd(input_file, output_file):
  # Loads LIDAR data from binary numpy format.
  # Data is stored as (x, y, z, intensity, ring index).
  scan = np.fromfile(input_file, dtype=np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('intensity', np.float32),
    ('ring_index', np.float32)]))

  logging.debug("points: {},{}".format(np.shape(scan), scan.dtype))
  point_cloud = pypcd.PointCloud.from_array(scan)
  point_cloud.save(output_file)
  print("Success! Pcd file saved to '{}'".format(output_file))
