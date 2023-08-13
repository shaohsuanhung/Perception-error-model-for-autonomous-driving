#!/usr/bin/env python3
###############################################################################
#  Authors: HungFrancis. All Rights Reserved
###############################################################################

import argparse
import os
import sys
import logging

from calibration_converter import convert_calibration
from dataset_converter import convert_dataset
from pcd_converter import convert_pcd


def main(args=sys.argv):
  parser = argparse.ArgumentParser(
    description="nuScenes dataset convert to record tool.",
    prog="main.py")

  parser.add_argument(
    "-i", "--input", action="store", type=str, required=True,
    help="Input file or directory.")
  parser.add_argument(
    "-o", "--output", action="store", type=str, required=False,
    help="Output file or directory.")
  parser.add_argument(
    "-t", "--type", action="store", type=str, required=False,
    default="rcd", choices=['rcd', 'cal', 'pcd'],
    help="Conversion type. rcd:record, cal:calibration, pcd:pointcloud")

  args = parser.parse_args(args[1:])
  logging.debug(args)

  if args.type == 'rcd':
    if os.path.isdir(args.input):
      if args.output is None:
        args.output = '.'
      convert_dataset(args.input, args.output)
    else:
      logging.error("Pls enter directory! Not '{}'".format(args.input))
  elif args.type == 'cal':
    if os.path.isdir(args.input):
      if args.output is None:
        args.output = '.'
      convert_calibration(args.input, args.output)
    else:
      logging.error("Pls enter directory! Not '{}'".format(args.input))
  elif args.type == 'pcd':
    if os.path.isfile(args.input):
      if args.output is None:
        args.output = 'result.pcd'
      convert_pcd(args.input, args.output)
    else:
      logging.error("Pls enter file! Not '{}'".format(args.input))
  else:
    logging.error("Input error! '{}'".format(args.input))


if __name__ == '__main__':
  main()
