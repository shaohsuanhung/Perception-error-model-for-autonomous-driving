# !/bin/env/python3
# 1. Define the class for perception objects
# 2. Implement the interpolation function for parameter I need to train / analysis
# 3. Load data in SceneFrames obj. and parseScenes to create list of AlignedFrames
# !!. Make sure every object is in the same frame of ref.
import numpy as np
import motmetrics as mm
import math
from dataset_reader import dataset_list, sensor_data, gt_data, ego_pose_data
from dataclasses import dataclass
import glob

@dataclass
class interpolated_gt_data:
    id        : float   
    # Add a detDatastamp????
    # detDataTimestamp: float
    position_x: float
    position_y: float
    position_z: float
    theta     : float
    length    : float
    width     : float
    height    : float
    type      : str     # Can be a filter for selecting specific type of object  
    time_stamp: float


class Trajectory():
    # Record the trajectory of each object (specific by unique ID)
    # For ID switch case, it will be recordedin the obj_list dict
    #
    def __init__(self, ID):
        self.ID = ID
        self.obj_list = dict()
        self.timeline = dict()
        self.startFrame = math.inf
        self.endFrame = 0

    def addObject(self, obj):
        # Take the obj.id as the frame indices
        # for key: obj ID, record obj
        self.obj_list[obj.id] = obj
        self.startFrame = min(self.startFrame,obj.id)
        self.endFrame   = max(self.endFrame,obj.id)
        # self.obj_list[obj.time_stamp] = obj
        # self.startFrame = min(self.startFrame,obj.time_stamp)
        # self.endFrame   = max(self.endFrame,obj.time_stamp)

    def addPairing(self,detObj):
        self.timeline[detObj.id] = detObj
    
    def makeAttributes(self):
        '''
        '''
        for frameID in range(self.startFrame,self.endFrame+1):
            if frameID in self.obj_list.keys():
            #    self.objList[frameID].firstAppear = False #bool if this obj is first time in GT
                if frameID in self.timeline.keys():
                    print('frameID:',frameID)
                    self.obj_list[frameID].currentDet = True
                else:
                    self.obj_list[frameID].currentDet = False

                if frameID-1 in self.timeline.keys() and frameID-1 in self.obj_list.keys() :
                    print('frameID-1:',frameID-1)
                    self.obj_list[frameID].previousDet = True
                    self.obj_list[frameID].previousPair = self.obj_list[frameID-1].Pair
                else:
                    self.obj_list[frameID].previousDet = False #bool if this obj was detected previous frame
        
   

class AlignedFrame(object):
    lidar_num_pairing = 0
    radar_num_pairing = 0
    def __init__(self, frameID, detData):
        self.frameID          = frameID
        self.detDataTimestamp = detData[0].time_stamp  # needed to interpolate the data
        self.reverseGTID      = dict()
        self.reverseDTID      = dict()
        self.det              = self.loadDet(detData)
        self.matchingResults  = list()
        self.egoPose          = list()
        self.gt               = list()
        self.pairings         = None
        #(w,w') in the paper, None is not detected, GT doesnt have the corresponding DT
        # input of model : input the pairing, predictor is GT
        
    def interpolateData(self, gtDataPrev, gtDataNext, egoDataPrev, egoDataNext):
        '''
        gtDataPrev : gt object that is smaller closet GT data to the DT data
        gtDataNext : gt object  index that is the larger closest GT data to the DT data
        egoDataPrev: ego pose object  index that is the smaller closest ego pose data to the DT data
        egoDataNext: ego pose object  index that is the larger closest ego pose data to the DT data
        '''
        self.egoPose = self.loadEgo(egoDataPrev, egoDataNext)
        self.gt = self.loadGT(gtDataPrev, gtDataNext)

    def loadGT(self, gtDataPrev, gtDataNext):
        # Load GT file and return a list of objects
        prevTimestamp = gtDataPrev[0].time_stamp
        nextTimestamp = gtDataNext[0].time_stamp
        ratio = (self.detDataTimestamp-prevTimestamp) / \
            (nextTimestamp-prevTimestamp)
        interpolated_gt = []
        for obj in gtDataPrev:
            # Find the objs with same id in gtDataNext: 
            # Can garuantee only have exactly one (tracked) or None (lost)in the next frame
            next_frame_obj = [next_obj for next_obj in gtDataNext if next_obj.id == obj.id]
            if next_frame_obj != []: # interpolate only when match id is found
                # Matching here
                interpolated_position_x = obj.position_x * \
                    (1-ratio)+ratio*next_frame_obj[0].position_x
                interpolated_position_y = obj.position_y * \
                    (1-ratio)+ratio*next_frame_obj[0].position_y
                interpolated_position_z = obj.position_z * \
                    (1-ratio)+ratio*next_frame_obj[0].position_z
                interpolated_theta = obj.theta*(1-ratio)+ratio*next_frame_obj[0].theta
                interpolated_length = obj.length * \
                    (1-ratio)+ratio*next_frame_obj[0].length
                interpolated_width = obj.width*(1-ratio)+ratio*next_frame_obj[0].width
                interpolated_height = obj.height * \
                    (1-ratio)+ratio*next_frame_obj[0].height
                new_obj_type = obj.type
                interpolated_timestamp = obj.time_stamp * \
                    (1-ratio)+ratio*next_frame_obj[0].time_stamp
                # Append the new interpolated gt data object in the list
                interpolated_gt.append(interpolated_gt_data(int(obj.id),interpolated_position_x, interpolated_position_y, interpolated_position_z,
                                    interpolated_theta, interpolated_length, interpolated_width, interpolated_height, new_obj_type, interpolated_timestamp))

        # If next_frame_obj is null, then interpolate_gt is null also
        # So no GT in this timestamp of AlignedFrame
        self.reverseGTID = {gt_obj.id: idx for (
            idx, gt_obj) in enumerate(interpolated_gt)}
        return interpolated_gt

    def loadEgo(self, egoDataPrev, egoDataNext):
        # load localization file and interpolate the position accordingly
        prevTimestamp = egoDataPrev.time_stamp
        nextTimestamp = egoDataNext.time_stamp
        ratio = (self.detDataTimestamp-prevTimestamp) / \
            (nextTimestamp-prevTimestamp)
        timestamp = egoDataPrev.time_stamp * \
            (1-ratio)+ratio*egoDataNext.time_stamp
        position_x = egoDataPrev.ego_position_x * \
            (1-ratio)+ratio*egoDataNext.ego_position_x
        position_y = egoDataPrev.ego_position_y * \
            (1-ratio)+ratio*egoDataNext.ego_position_y
        position_z = egoDataPrev.ego_position_z * \
            (1-ratio)+ratio*egoDataNext.ego_position_z
        orientation_qx = egoDataPrev.orientation_qx * \
            (1-ratio)+ratio*egoDataNext.orientation_qx
        orientation_qy = egoDataPrev.orientation_qy * \
            (1-ratio)+ratio*egoDataNext.orientation_qy
        orientation_qz = egoDataPrev.orientation_qz * \
            (1-ratio)+ratio*egoDataNext.orientation_qz
        orientation_qw = egoDataPrev.orientation_qw * \
            (1-ratio)+ratio*egoDataNext.orientation_qw
        # package the three params to egoPose
        return ego_pose_data(float(timestamp), float(position_x), float(position_y), float(position_z),
                             float(orientation_qx), float(orientation_qy), float(orientation_qz), float(orientation_qw))

    def loadDet(self, detData):
        # Load the raw perception data in list of objects form
        # no need to interpolate
        # {key:value} = {id:index in the list}
        self.reverseDTID = {det_obj.id: idx for (
            idx, det_obj) in enumerate(detData)}
        return detData

    def getPairings(self):
        if self.pairings is not None:
            return self.pairings

        else:
            self.pairings = list()
            for pairIDs in self.matchingResults:
                a, b = pairIDs
                gt = self.gt[a] if a >= 0 else None
                det = self.det[b] if b >= 0 else None
                self.pairings.append((gt, det))
                if (gt is not None and det is not None):
                    if (det.sensor_id == 'velodyne32'):
                        AlignedFrame.lidar_num_pairing += 1
                    else:
                        AlignedFrame.radar_num_pairing += 1
            return self.pairings

    def makeMatchMatrix(self, maxV=True):
        # Calculate the matching distance b.t. each DT and GT
        # Build up the id list for DT and GT
        # Add the non-repeated id of DT/GT in the Trajectory
        # Update the MOTAccumulator
        self.matchMatrix = np.zeros((len(self.gt), len(self.det)))
        
        for i in range(0, len(self.gt)):
            for j in range(0, len(self.det)):
                if PercObj.compatibilityCheck(self.gt[i], self.det[j]):
                    dist = PercObj.matchingDist(
                        self.gt[i], self.det[j], maxV)
                else:
                    dist = np.nan

                self.matchMatrix[i][j] = dist

class PercObj():
    def __init__(self, data):
        pass

    @staticmethod
    def matchingDist(a, b, maxV):
        MAX_POSE_ERROR = 10
        position2d_a = np.array([a.position_x, a.position_y])
        position2d_b = np.array([b.position_x, b.position_y])
        dist = np.linalg.norm(position2d_a-position2d_b)
        # If the dist is larger than the threshold, then turn to nan to ignore
        if (maxV and (dist > MAX_POSE_ERROR)):
            dist = np.nan

        return dist

    @staticmethod
    def compatibilityCheck(a, b):
        # determine if object a and b are compatible based on your assumptions ( hard constraint on no matching allowed)
        return True


class SceneFrames(object):
    def __init__(self, id):
        # parseScene should be called before the matching function
        self.id         = id
        self.frameList  = list()
        self.gtTrajDict = dict()
        self.dtTrajDict = dict()

    def parseScene(self, dataset):
        # load detection file a list
        # Can be null in the beginning
        # detectionFrameList = [element[1] for element in dataset.object_map]
        GTFrameList = [element[2] for element in dataset.object_map]
        GTFrameList = list(filter(None,GTFrameList))
        detectionFrameList = [element[1] for element in dataset.object_map]
        detectionFrameList = list(filter(None,detectionFrameList))
        ###### If not going to extrapolate the ground truth :#########
        GTLastTimestamp = GTFrameList[-1][0].time_stamp
        detectionFrameList = [element for element in detectionFrameList if element[0].time_stamp < GTLastTimestamp]
        ##############################################################
        egoFrameList = dataset.ego_vehicle_list
        detCount = 0
        egoCount = 0
        for detectionFrame in detectionFrameList: 
            newFrame = AlignedFrame(detCount, detectionFrame)
            detDateTimestamp = newFrame.detDataTimestamp
            GTCount = 0  # index of closest GT frame compare to the DT
            egoCount = 0  # index

            # Find 2 closest GT frames, one before perception timestamp and \
            # one for after it
            # GTFrameList[idx][....] share same time_stamp
            while GTFrameList[GTCount][0].time_stamp < detDateTimestamp:
                gtDataPrev = GTCount
                GTCount += 1
            gtDataNext = GTCount

            while egoFrameList[egoCount].time_stamp < detDateTimestamp:
                egoDataPrev = egoCount
                egoCount = egoCount+1
            egoDataNext = egoCount

            newFrame.interpolateData(
                GTFrameList[gtDataPrev], GTFrameList[gtDataNext], egoFrameList[egoDataPrev], egoFrameList[egoDataNext])
            self.frameList.append(newFrame)
            detCount += 1

    def match(self):
        acc = mm.MOTAccumulator()
        # Keep track of the DT/GT trajectory for single ID 
        for frame in self.frameList:
            # 1. Calculate the matching distance b.t. each DT and GT
            frame.makeMatchMatrix()
            # 2. Build up the id list for DT and GT 
            gtIDList = [x.id for x in frame.gt]
            dtIDList = [x.id for x in frame.det]

            # 3. Add the non-repeated id of DT/GT in the Trajectory 
            for gtObj in frame.gt:
                # Check whether ID of gtObj in the gtTrajectory.
                if gtObj.id not in self.gtTrajDict:
                    self.gtTrajDict[gtObj.id] = Trajectory(gtObj.id)
                # else:
                #     print('Repeated ID{} in GT!'.format(gtObj.id))
                # If id is "NOT" in the list, add a new class Trajectory in
                # the GT Trajectory dictionary:
                # {key:value} = {gtobj.id : Trajectory class}
                #
                # Else, if id is already in the list, then not build new class Trajectory
                # Then addObject in that cooresponding key in the dictionary
                self.gtTrajDict[gtObj.id].addObject(gtObj)
            
            for dtObj in frame.det:
                if dtObj.id not in self.dtTrajDict:
                    self.dtTrajDict[dtObj.id] = Trajectory(dtObj.id)
                # else:
                #     print('Repeated ID{} in DT!'.format(dtObj.id))
                self.dtTrajDict[dtObj.id].addObject(dtObj)
            
            # 4. Update the MOTAccumulator
            acc.update(gtIDList,dtIDList,frame.matchMatrix,frame.frameID)
        # 5. Update the matching results for each frame
        for frame in self.frameList:
            if len(frame.gt):
                res = acc.mot_events.loc[frame.frameID]
                for idx , row in res.iterrows():
                    if row["Type"] in ["MATCH","SWITCH"]:
                        gt = frame.reverseGTID[int(row["OId"])]
                        dt = frame.reverseDTID[int(row["HId"])]
                        frame.matchingResults.append((gt,dt))
                        self.gtTrajDict[int(row["OId"])].addPairing(frame.det[dt])
                    if row["Type"] in ["MISS"]:
                        gt = frame.reverseGTID[int(row["OId"])]
                        frame.matchingResults.append((gt,-1))
                    if row["Type"] in ["FP"]:
                        dt = frame.reverseDTID[int(row["HId"])]
                        frame.matchingResults.append((-1,dt))

            frame.getPairings()

        for trj in self.gtTrajDict.keys():
            self.gtTrajDict[trj].makeAttributes()

if __name__ == '__main__':
    # cc8_detection_filename = '/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/detection/cc8c0bf57f984915a77078b10eb33198_detection.txt'
    # cc8_gt_filename = '/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/cc8c0bf57f984915a77078b10eb33198_gt.txt'
    # cc8_gt_ego = '/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/cc8c0bf57f984915a77078b10eb33198_gt_ego-pose.txt'
    # cc8_scene_token = cc8_gt_filename[-39:-7]
    # cc8_dataset = dataset_list(cc8_scene_token)
    # with open(cc8_detection_filename, 'r') as file:
    #     file_content = file.read()

    # with open(cc8_gt_filename, 'r') as file:
    #     gt_content = file.read()

    # with open(cc8_gt_ego, 'r') as file:
    #     gt_ego_content = file.read()

    # # Dataset parse
    # cc8_dataset.make_detection(file_content)
    # cc8_dataset.make_gt(gt_content, gt_ego_content)
    # cc8_dataset.make_localization(gt_ego_content)
    # cc8_dataset.generate_object_map()

    # # Matching
    # cc8_scene_pair = SceneFrames(1)
    # cc8_scene_pair.parseScene(cc8_dataset)
    # cc8_scene_pair.match()
    DTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/detection/*_detection*.txt")
    GTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/*_gt*.txt") 
    Egofile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/ego/*_gt*.txt")
    Scene_list = []
    paired_scene_list = []
    DTfile_list.sort(); GTfile_list.sort(); Egofile_list.sort()
    for DT, GT, Ego in zip(DTfile_list,GTfile_list,Egofile_list):
        Scene_list.append(dataset_list(DT,GT,Ego))
        paired_scene_list.append(SceneFrames(Scene_list[-1].scene_token))
        print("Parse Scene....")
        paired_scene_list[-1].parseScene(Scene_list[-1])
        print("Match Scene....")
        paired_scene_list[-1].match()
# Planning:
# for loop, 
# Read all the token name in the directory, 
# in the dataset_list, in the __init___: read in the ego, dt, gt file
#                                        and make_detection, gt, localzation, generate_object_map_object_map
# 