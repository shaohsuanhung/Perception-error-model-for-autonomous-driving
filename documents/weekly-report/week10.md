# Week 10 (From Sept. 25 to Sept. 29)
**Summary of the week**  
- Implement the matching algorithm by myself base on the [paper](https://jivp-eurasipjournals.springeropen.com/articles/10.1155/2008/246309):  "Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics"
- The implementation script is uploaded in [/scripts/match](../../scripts/match/).
- Literature review on HMM based that modeling time-series sensor error. Plan on the next step experiement. 
## Weekly outcome
- Details of the matching algorithm: the overall procedure is described as following steps:  
(1) Read in the text file.   
(2) Parse the text and build the corresponding class object (detected object (DT), ground truth (GT) object, ego pose) that implemented in the [week9](./week9.md).   
(3) Parse the DT, GT, ego pose frame by frame. If the timestamp of DT have no corresponding GT and ego pose, use the closest frame to do the interpolation. After this step, one will get the pairs of DT and GT w.r.t. timestamp, i.e. (o(t), o'(t)).  
(4) Use the [motmetrics](https://github.com/cheind/py-motmetrics), a library a Python implementation of metrics for benchmarking multiple object trackers, to calculate the matching distance between each DT and GT. Extract the "MATCH" and "SWITCH" column in the `MOTFAcculator`, as the matching result. 
## Next week task
- Comeing up how which model to use for modeling the sensor error.