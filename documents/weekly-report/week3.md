
# Week 3 (Aug. 7 ~ Aug. 11)
## Progress
1. Finish setting up the github page.
2. Finish the nuscenes dataset converter. scripts are under the ```./scripts/nuscenes_converter```.
3. The perception module is failed to launch. Will report the issue on the Apollo github page. Already tried the following method to fix the problem: (The further details error message please see the below section.)
    * Searched online on Github, Stackoverflow.
    * Re-installed the Apollo docker, GPU driver, CUDA Devkit.
    * Co-work with PhD.   
## Challenging task of the week:
- **The perception module is failed to launch.**  
    - **Expected action**: The sensor module (camera, radar, LiDAR) is sucessfully launch ,the detected bounding box is shown in the Dreamview.
    - **Actual action**: The node and message of nuscenes dataset is working fine. But the nothing shows in the Dreamview, and the terminal shows some error message.
    - **Step to reproduce the issue**: 
    Follow the [documents](https://github.com/ApolloAuto/apollo/blob/master/docs/06_Perception/how_to_run_perception_module_on_your_local_computer.md) at apollo/docs/06_Perception/how_to_run_perception_module_on_your_local_computer.md.
1. Run bootstrap.sh
```
$ bootstrap.sh
```
2. Launch static transform
```
$ cyber_launch start /apollo/modules/transform/launch/static_transform.launch
```
3. Launch the perception modules
```
$ cyber_launch start /apollo/modules/perception/production/launch/perception.launch
```
On the Dreamview, It can be seen that the button of perception is launched, but then close automatically after few seconds.
## Next week task

### Urgent
- (From week 1) Internship research proposal (including problem statement, goal)
- (From week 1) Gantt chart of the project.
- Launch the perception module.
### Normal
- Run the Apollo perception module, and see how the system perceived the nuscenes dataset.