## Introduction to the end to end dataset process pipeline
### Convert record files to text files
Run the `record2txt.sh` to convert record files under the assigned directory. For the ground truth record files, they are name after `{scene token}_gt.record`, read `/apollo/perception/obstacles` and `/apollo/localization/pose` channels in the files unber same path as shell scripts and convert to the text files, `/text_dataset/ground_truth/{scene token}.txt`, and `/text_dataset/ground_truth/{scene token}_eog-pose.txt`. For the perceived record files, they are name after `{scene_token}_detection.record.0000`, read `/apollo/perception/obstacles`, and convert to the text files, `/text_dataset/detection/{scene token}.txt`.

### Automatically launch the Apollo modules and record assigned channel
Before run the shell script, user must install the [tmux](https://github.com/tmux/tmux/wiki) in the Apollo docker using the following command:
```
sudo apt-get update
sudo apt-get install tmux
```
After that, the shell script would use the tmux to open 4 terminal and launch the transform module, perception module and cyber recorder to play and record the `_detection.record`.

### Auto-terminal setup
Using the [gnome-terminal](https://help.gnome.org/users/gnome-terminal/stable/) to open multiple terminal and launch the docker at the same time. 
```
$ gnome-terminal --load-config=workplace.txt
```
### Dataset Analyzer
Read the text file and sort out the data of perceptive data and ground truth. Build up the object list frame by frame.