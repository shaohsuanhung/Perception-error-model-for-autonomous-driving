# Week 2 (July 31 ~ Aug. 4)

**Summary of the week:**  
1. Learning API of Cyber RT and nuScenes dataset. 
2. Work on python that converting the nuscenes dataset to .record files.

## Progress
- Study the cyber_rt protocol and know the whole message passing system to covert the dataset.
- The pratice scripts of the nuscenes dataset is under ```./scripts/nuscenes_dataset_pratice/```.
- Writing python scripts that convert the nuscenes dataset to .record file, placed at ```./scripts/nuscenes_converter```.

## Challenging task of the week:

- **How to convert radar data to Apollo**.  
Find scripts that convert the dataset had provided by the Apollo, but they don’t support the radar dataset convertion. So I had to modify the scripts by myself. Before starting programming, I need to look into several API such as nuscenes dataset Dev-kit API, Cyber RT API, and messages that published on the runtime framework.

## Next week task

### Urgent
- Convert the nuscenses dataset to .record file.
- (From week 1) Internship research proposal (including problem statement, goal)
- (From week 1) Set up github page, write research description, deliverable on main page.
- (From week 1) Gantt chart of the project.
### Normal
- Run the Apollo perception module, and see how the system perceived the nuscenes dataset.