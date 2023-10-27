# Week 12 (From Oct.9 to Oct. 13)
**Summary of the week**  
1. Find the orientation alignment issue in the Apollo. 
2. Fix the orientation alignment issue.
3. Implement the features that render the result of matching. Visaulize the object trajectory of GT and DT.
## Weekly outcome
- To fix the oriendtation alignment issue, I rewrite the [nuscenes_calibration_converter](../../scripts/nuscenes_converter/calibration_converter.py), to add a new sensor transform tree (dag files, launcg files and pb.txt files) in the corresponding directory (namely, apollo/modules/transform/) even though Andrea said the difference of orientation among different scenes are pretty small.
- Also, in the [play_record.sh](../../scripts/end_to_end_dataset_process_pipeline/play_record.sh), parse the scene token and launch the corresponding static transform module,
- Fix the bug that incorrectly assign the id in the record file. In the [data_converter.py](../../scripts/nuscenes_converter/dataset_converter.py), should initialize the instance dictionary in the constructor. Otherwise the same obj won’t have same id.
- Fix the bug on orientation of detection. This is raise due to the coordinate setup is different between Apollo and Nuscenes. Therefore, it would be 90 deg. between detection and the GT. The best strategy for my case is rotate the detection after get the detection data from the Apollo, change the oridentation by 90 in my python script.
- Identify the issue that interpolated GT data. When interpolating the orientation, for the case that theta change from 359 deg. to 0 deg. or from 0 deg to 359 deg., the object would rotate counter clock wise (in fact should be clock wise rotation). This is the disadvantage of interpolation of angular data. To fix the problem, we should deal with the case that the angle change from Quadrant 4 to Quadrant 1 or Quadrant 1 to Quadrant (or every rotation that cross the 0 deg line) since the ‘value-inconsistent’. So we should add the 2*pi to the original angle.


## Next week task
### Normal
- Considering will use more data to train the HMM, the matching part will take too much time, so are going to implement seralization of the matching result (to json format)