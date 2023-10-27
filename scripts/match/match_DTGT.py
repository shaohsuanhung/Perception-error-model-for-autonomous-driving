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
import dataclasses
import glob
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rec
import matplotlib.animation as animation
import math
from celluloid import Camera

@dataclass
class interpolated_gt_data:
    id        : int   
    # Add a detDatastamp????
    frameID: int
    position_x: float
    position_y: float
    position_z: float
    theta     : float
    length    : float
    width     : float
    height    : float
    type      : str     # Can be a filter for selecting specific type of object  
    time_stamp: float
    # 
    currentDet: bool
    previousDet: bool
    # Ego pose at same time_stamp
    ego_position_x: float
    ego_position_y: float
    ego_position_z: float
    ego_theta     : float

class AdvancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__jsonencode__'):
            return obj.__jsonencode__()

        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
        # json.dumps(obj_name,cls=AdvancedJSONEncoder)
    
class Trajectory():
    # Record the trajectory of each object (specific by unique ID)
    # For ID switch case, it will be recordedin the obj_list dict
    #
    def __init__(self, ID):
        self.ID = ID
        self.obj_list = dict() # List of interpolated_gt_data
        self.timeline = dict()
        self.startFrame = math.inf
        self.endFrame = 0

    def __jsonencode__(self):
        # Use the local dictionary to serizalize th DT and GT
        local_obj_list = dict()
        local_timeline = dict()
        for (idx,obj) in zip(self.obj_list.keys(), self.obj_list.values()):
            local_obj_list[idx] = dataclasses.asdict(obj)
        for (idx,obj) in zip(self.timeline.keys(),self.timeline.values()):
            local_timeline[idx] = dataclasses.asdict(obj)

        return  {'ID':self.ID,'obj_list':local_obj_list,'timeline':local_timeline,\
                 'startFrame':self.startFrame,'endFrame':self.endFrame}

    def addObject(self, obj,frameID):
        # Take the obj.id as the frame indicestimelinedFrame,obj.id)
        # self.startFrame = min(self.startFrame,obj.frameID)
        # self.endFrame   = max(self.endFrame,obj.frameID)
        # self.obj_list[obj.time_stamp] = obj
        self.obj_list[frameID] = obj
        self.startFrame = min(self.startFrame,frameID)
        self.endFrame   = max(self.endFrame,frameID)

    def addPairing(self,detObj,frameID):
        # Only the gtTrajectory will call this function
        self.timeline[frameID] = detObj
    
    def makeAttributes(self):
        '''
        '''
        for frameID in range(self.startFrame,self.endFrame+1):
            if frameID in self.obj_list.keys():
            #    self.objList[frameID].firstAppear = False #bool if this obj is first time in GT
                if frameID in self.timeline.keys():
                    # print('frameID:',frameID)
                    self.obj_list[frameID].currentDet = True
                else:
                    self.obj_list[frameID].currentDet = False

                if frameID-1 in self.timeline.keys() and frameID-1 in self.obj_list.keys() :
                    # print('frameID-1:',frameID-1)
                    self.obj_list[frameID].previousDet = True
                    # self.obj_list[frameID].previousPair = self.obj_list[frameID-1].Pair
                else:
                    self.obj_list[frameID].previousDet = False #bool if this obj was detected previous frame

    def plot_trajectory(self,ego_list):
        '''
        Plot trajectory of the objects 
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot the interpolated ground truth
        for obj in self.obj_list.values():
            plt.plot(obj.position_x,obj.position_y,'bo',markersize=1)
            ax.add_patch(Rec(xy=(obj.position_x-(obj.length/2),obj.position_y-(obj.width/2)),width=obj.length,height=obj.width,angle=obj.theta*180/math.pi,color='blue',rotation_point='center',fill=False))
            plt.arrow(obj.position_x,obj.position_y,math.cos(obj.theta),math.sin(obj.theta),width=0.1,color='blue')
            # ego_pose = [ego for ego in ego_list if ego.time_stamp == obj.time_stamp]
            # if ego_pose != []:
            #     ax.scatter(ego_pose[0].ego_position_x,ego_pose[0].ego_position_y,s=2,color='red')
            #     Angle = Quaternion(ego_pose[0].orientation_qw,ego_pose[0].orientation_qx,ego_pose[0].orientation_qy,ego_pose[0].orientation_qz).to_euler().yaw
            #     plt.arrow(ego_pose[0].ego_position_x,ego_pose[0].ego_position_y,math.cos(Angle),math.sin(Angle),width=0.1,color='black')

            # plt.title("Timestamp:{},(Frame ID:{})".format(obj.time_stamp,obj.frameID))
            # plt.legend('Interpolated GT','Ego Pose','GT in prev. frame','GT in next frame')
            # plt.legend(prop = {'size': 6})
            # plt.xlim(330,455)
            # plt.ylim(1075,1230)
            # plt.savefig('./render_traj/'+str(obj.frameID))
        

        # Plot the matched result
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(111)
        ax2 = ax
        for obj in self.timeline.values():
            ax2.scatter(obj.position_x,obj.position_y,color='red',s=2)
            ax2.add_patch(Rec(xy=(obj.position_x-(obj.length/2),obj.position_y-(obj.width/2)),width=obj.length,height=obj.width,angle=obj.theta*180/math.pi,color='red',rotation_point='center',fill=False))
            plt.arrow(obj.position_x,obj.position_y,math.cos(obj.theta),math.sin(obj.theta),width=0.1,color='red')
            
        plt.title("Timestamp:{},(GT Obj ID:{})".format(obj.time_stamp,self.ID))
            # plt.legend('Interpolated GT','Ego Pose','GT in prev. frame','GT in next frame')
            # plt.legend(prop = {'size': 6})
        # plt.xlim(600,700)
        # plt.ylim(1580,1680)
        # plt.xlim(330,440)
        # plt.ylim(1080,1220)
        plt.savefig('./render_traj/DT/'+str(self.ID))
        plt.close()
    
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
        self.gt= self.loadGT(gtDataPrev, gtDataNext)

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
                # print("HIT!")
                # Matching here
                interpolated_position_x = obj.position_x * \
                    (1-ratio)+ratio*next_frame_obj[0].position_x
                interpolated_position_y = obj.position_y * \
                    (1-ratio)+ratio*next_frame_obj[0].position_y
                interpolated_position_z = obj.position_z * \
                    (1-ratio)+ratio*next_frame_obj[0].position_z
                interpolated_length = obj.length * \
                    (1-ratio)+ratio*next_frame_obj[0].length
                interpolated_width = obj.width*(1-ratio)+ratio*next_frame_obj[0].width
                interpolated_height = obj.height * \
                    (1-ratio)+ratio*next_frame_obj[0].height
                new_obj_type = obj.type
                interpolated_timestamp = obj.time_stamp * \
                    (1-ratio)+ratio*next_frame_obj[0].time_stamp
                ## Prevent the theta jump from 2pi to 0 or vice versa
                if (obj.theta  < 2*math.pi and obj.theta>3*math.pi/2 and next_frame_obj[0].theta > 0 and next_frame_obj[0].theta < math.pi/2):
                    # next_frame_obj[0].theta += 2*math.pi
                    interpolated_theta = obj.theta*(1-ratio)+ratio*(next_frame_obj[0].theta+math.pi*2)
                elif (next_frame_obj[0].theta  < 2*math.pi and next_frame_obj[0].theta>3*math.pi/2 and obj.theta > 0 and obj.theta < math.pi/2):
                    # obj.theta += 2*math.pi
                    interpolated_theta = (obj.theta*(1-ratio)+math.pi*2)+ratio*next_frame_obj[0].theta
                else:
                    # pass
                    interpolated_theta = obj.theta*(1-ratio)+ratio*next_frame_obj[0].theta
                # interpolated_theta = obj.theta*(1-ratio)+ratio*next_frame_obj[0].theta
                # Append the new interpolated gt data object in the list
                Angle = Quaternion(self.egoPose.orientation_qw,self.egoPose.orientation_qx,self.egoPose.orientation_qy,self.egoPose.orientation_qz).to_euler().yaw
                interpolated_gt.append(interpolated_gt_data(int(obj.id),self.frameID,interpolated_position_x, interpolated_position_y, interpolated_position_z,
                                    interpolated_theta, interpolated_length, interpolated_width, interpolated_height, new_obj_type, self.detDataTimestamp,\
                                    False,False,self.egoPose.ego_position_x,self.egoPose.ego_position_y,self.egoPose.ego_position_z,Angle))
            # If next_frame_obj is null, then interpolate_gt is null also, 
            # the trakcing object is missing

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
        return ego_pose_data(self.detDataTimestamp, float(position_x), float(position_y), float(position_z),
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
                    angular_dist = PercObj.matchingAngularDist(
                        self.gt[i], self.det[j],maxV)
                    dist = dist + angular_dist
                else:
                    dist = np.nan

                self.matchMatrix[i][j] = dist

    def render_interpolated_GT(self,gt_prev_frame,gt_next_frame):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot the interpolated ground truth
        for (idx,obj) in enumerate(self.gt):
                # if obj.id == 16: # Uncomment to trace specific ID
                    plt.plot(obj.position_x, obj.position_y, 'bo',label='Interpolated GT' if idx ==0 else "")
                    ax.add_patch(Rec(xy=(obj.position_x-(obj.length/2),obj.position_y-(obj.width/2)),width=obj.length,height=obj.width,angle=obj.theta*180/math.pi,color='blue',rotation_point='center',fill=False))
        # Plot ego vehicle pose
        plt.plot(self.egoPose.ego_position_x, self.egoPose.ego_position_y, 'ro',label='Ego Pose')
        Angle = Quaternion(self.egoPose.orientation_qw,self.egoPose.orientation_qx,self.egoPose.orientation_qy,self.egoPose.orientation_qz).to_euler().yaw
        plt.arrow(self.egoPose.ego_position_x,self.egoPose.ego_position_y,math.cos(Angle),math.sin(Angle),width=0.1,color='blue')
        # Plot the prev and next frame ground truth
        for (idx,obj) in enumerate(gt_prev_frame):
                # if obj.id == 16: # Uncomment to trace specific ID
                    plt.plot(obj.position_x, obj.position_y, 'g>',label='GT in prev. frame' if idx == 0 else "")
                    ax.add_patch(Rec(xy=(obj.position_x-(obj.length/2),obj.position_y-(obj.width/2)),width=obj.length,height=obj.width,angle=obj.theta*180/math.pi,color='green',rotation_point='center',fill=False))
                    obj_next_frame = [next_obj for next_obj in gt_next_frame if next_obj.id == obj.id]
                    if obj_next_frame != []:
                        obj_next_frame = obj_next_frame[0]
                        plt.plot(obj_next_frame.position_x, obj_next_frame.position_y, 'c<',label='GT in next frame' if idx == 0 else "")
                        ax.add_patch(Rec(xy=(obj_next_frame.position_x-(obj_next_frame.width/2),obj_next_frame.position_y-(obj_next_frame.length/2)),width=obj_next_frame.width,height=obj_next_frame.length,angle=obj_next_frame.theta*180/math.pi,color='yellow',rotation_point='center',fill=False))      
                        plt.arrow(obj.position_x,obj.position_y,obj_next_frame.position_x-obj.position_x,obj_next_frame.position_y-obj.position_y,width=0.1,color = 'black',ls='--')

        # plt.show()
        plt.title("Timestamp:{},(Frame ID:{})".format(self.detDataTimestamp,self.frameID))
        # plt.legend('Interpolated GT','Ego Pose','GT in prev. frame','GT in next frame')
        plt.legend(prop = {'size': 6})
        # plt.xlim(330,455)
        # plt.ylim(1075,1230)
        plt.savefig('./render_frame/'+str(self.frameID))
        plt.close()

    def render_GT_DT(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot the DT
        for (idx,obj) in enumerate(self.det):
                    plt.plot(obj.position_x, obj.position_y, 'ro',label='DT' if idx == 0 else "",markersize=1)
                    ax.add_patch(Rec(xy=(obj.position_x-(obj.length/2),obj.position_y-(obj.width/2)),width=obj.length,height=obj.width,angle=obj.theta*180/math.pi,color='red',rotation_point='center',fill=False))
                    plt.arrow(obj.position_x,obj.position_y,5*math.cos(obj.theta),5*math.sin(obj.theta),width=0.1,color='red')
            
        # Plot the interpolated ground truth
        for (idx,obj) in enumerate(self.gt):
                # if obj.id == 16: # Uncomment to trace specific ID
                    plt.plot(obj.position_x, obj.position_y, 'bo',label='Interpolated GT' if idx ==0 else "",markersize=1)
                    ax.add_patch(Rec(xy=(obj.position_x-(obj.length/2),obj.position_y-(obj.width/2)),width=obj.length,height=obj.width,angle=obj.theta*180/math.pi,color='blue',rotation_point='center',fill=False))
                    plt.arrow(obj.position_x,obj.position_y,5*math.cos(obj.theta),5*math.sin(obj.theta),width=0.1,color='blue')
            
        # Plot ego vehicle pose
        plt.plot(self.egoPose.ego_position_x, self.egoPose.ego_position_y, 'k*',label='Ego Pose',markersize = 10)
        Angle = Quaternion(self.egoPose.orientation_qw,self.egoPose.orientation_qx,self.egoPose.orientation_qy,self.egoPose.orientation_qz).to_euler().yaw
        # Angle = math.radians(quaternion_to_euler(self.egoPose.orientation_qx,self.egoPose.orientation_qy,self.egoPose.orientation_qz,self.egoPose.orientation_qw))
        plt.arrow(self.egoPose.ego_position_x,self.egoPose.ego_position_y,5*math.cos(Angle),5*math.sin(Angle),width=0.2,color='black')
                
        # plt.show()
        plt.title("Timestamp:{},(Frame ID:{})".format(self.detDataTimestamp,self.frameID))
        # plt.legend('Interpolated GT','Ego Pose','GT in prev. frame','GT in next frame')
        plt.legend(prop = {'size': 6})
        plt.xlim(330,440)
        plt.ylim(1080,1220)
        plt.savefig('./render_frame/'+str(self.frameID))
        plt.close()

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

    @staticmethod
    def matchingAngularDist(a,b,maxV):
        MAX_ANG_ERROR = math.pi/4 # 90 degree
        angle_a = a.theta
        angle_b = b.theta
        angular_dist = abs(angle_a-angle_b)
        if (maxV and (angular_dist > MAX_ANG_ERROR)):
            angular_dist = np.nan
        else:
            angular_dist = 0.0   
        return angular_dist
class SceneFrames(object):
    def __init__(self, id):
        # parseScene should be called before the matching function
        self.id         = id
        self.frameList  = list() # list of AlignedFrame
        self.gtTrajDict = dict() # dict of Trajectory
        self.dtTrajDict = dict() # dict of Trajectory
        ################ For debug variable #########
        self.orientation_record = list()

    def __jsonencode__(self):
        # Use the local dictionary to serizalize th DT and GT
        return  {'Scene Token':self.id,'gtTrajDict':self.gtTrajDict}
    
    def render_GT(self,GTFrameList):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for (idx,oneframe) in enumerate(GTFrameList):
            fig.clear()
            for obj in oneframe:
                plt.plot(obj.position_x, obj.position_y, 'bo')
                ax.add_patch(Rec(xy=(obj.position_x-(obj.width/2),obj.position_y-(obj.length/2)),width=obj.width,height=obj.length,angle=obj.theta*180/math.pi,color='blue',rotation_point='center',fill=False))
                plt.xlim(330,470)
                plt.ylim(1040,1250)
            plt.savefig('./render_trueGT_frame/'+str(idx))

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
        # Render original GT
        # self.render_GT(GTFrameList)
        #

        for detectionFrame in detectionFrameList: 
            newFrame = AlignedFrame(detCount, detectionFrame)
            detDateTimestamp = newFrame.detDataTimestamp
            GTCount = 0  # index of closest GT frame compare to the DT
            egoCount = 0  # index

            # Find 2 closest GT frames, one before perception timestamp and \
            # one for after it
            # GTFrameList[idx][....] share same time_stamp
            while GTFrameList[GTCount][-1].time_stamp < detDateTimestamp:
                gtDataPrev = GTCount
                GTCount += 1
            gtDataNext = GTCount

            while egoFrameList[egoCount].time_stamp < detDateTimestamp:
                egoDataPrev = egoCount
                egoCount = egoCount+1
            egoDataNext = egoCount

            ratio = newFrame.interpolateData(
                GTFrameList[gtDataPrev], GTFrameList[gtDataNext], egoFrameList[egoDataPrev], egoFrameList[egoDataNext])
            self.frameList.append(newFrame)
            detCount += 1
            ####### For Checking the interpolated GT result ##########
            # newFrame.render_interpolated_GT(GTFrameList[gtDataPrev],GTFrameList[gtDataNext])
            # newFrame.render_GT_DT()
        print('foo')

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
                self.gtTrajDict[gtObj.id].addObject(gtObj,frame.frameID)
            
            for dtObj in frame.det:
                if dtObj.id not in self.dtTrajDict:
                    self.dtTrajDict[dtObj.id] = Trajectory(dtObj.id)
                # else:
                #     print('Repeated ID{} in DT!'.format(dtObj.id))
                self.dtTrajDict[dtObj.id].addObject(dtObj,frame.frameID)
            
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
                        self.gtTrajDict[int(row["OId"])].addPairing(frame.det[dt],frame.frameID)
                    if row["Type"] in ["MISS"]:
                        gt = frame.reverseGTID[int(row["OId"])]
                        frame.matchingResults.append((gt,-1))
                    if row["Type"] in ["FP"]:
                        dt = frame.reverseDTID[int(row["HId"])]
                        frame.matchingResults.append((-1,dt))

            frame.getPairings()

        for trj in self.gtTrajDict.keys():
            self.gtTrajDict[trj].makeAttributes()

def plot_ego(scene):
        # Verify the ego pose is correct
        ego_list = scene.ego_vehicle_list
        fig = plt.figure()
        ax = fig.add_subplot(111)
        camera = Camera(fig)
        for (idx,obj) in enumerate(ego_list):
            if idx > 3000:
                break
            else:
                print(idx)
                ax.scatter(obj.ego_position_x,obj.ego_position_y,color='blue')
                Angle = Quaternion(obj.orientation_qx,obj.orientation_qy,obj.orientation_qz,obj.orientation_qw).to_euler().yaw*(180/math.pi)
                # print(Angle)
                plt.arrow(obj.ego_position_x,obj.ego_position_y,math.cos(Angle),math.sin(Angle),width=0.1,color='black')
                camera.snap()

        anim = camera.animate()
        anim.save('./trajectory_ego_ID_{}.mp4'.format(scene.scene_token))



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

def quaternion_to_euler(x,y,z,w):
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

        return X

def make_pairing_list(Scene_list,paired_scene_list,DT,GT,Ego):
    '''
    Pipeline of parse scene and match scene for each nuscene scene record
    '''
    #########################################################################
    # Scene_list.append(dataset_list(DT,GT,Ego))
    # # plot_ego(Scene_list[-1])
    # paired_scene_list.append(SceneFrames(Scene_list[-1].scene_token))
    # print("Parse Scene....")
    # paired_scene_list[-1].parseScene(Scene_list[-1])
    # print("Match Scene....")
    # paired_scene_list[-1].match()
    # # Plat all trajectory
    # # for gtTraj in paired_scene_list[-1].gtTrajDict.values():
    # #     gtTraj.plot_trajectory(Scene_list[-1].ego_vehicle_list)
    #########################################################################
    Scene_list.append(dataset_list(DT,GT,Ego))
    # plot_ego(Scene_list[-1])
    paired_scene_list[Scene_list[-1].scene_token]=SceneFrames(Scene_list[-1].scene_token)
    print("Parse Scene....")
    paired_scene_list[Scene_list[-1].scene_token].parseScene(Scene_list[-1])
    print("Match Scene....")
    paired_scene_list[Scene_list[-1].scene_token].match()
    # Smaller
    paired_scene_list[Scene_list[-1].scene_token].gtTrajDict = {key:value for (key,value) in paired_scene_list[Scene_list[-1].scene_token].gtTrajDict.items() if value.timeline !={}}
    print('Foo')


if __name__ == '__main__':
    ################################# Demo on single scene #########################################
    # # Load the dataset
    # cc8_detection_filename = '/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/detection/cc8c0bf57f984915a77078b10eb33198_detection_sun.txt'
    # cc8_gt_filename = '/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/cc8c0bf57f984915a77078b10eb33198_gt_sun.txt'
    # cc8_gt_ego = '/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/ego/cc8c0bf57f984915a77078b10eb33198_gt_sun_ego-pose.txt'
    # cc8_scene_token = cc8_gt_filename[-43:-11]
    # cc8_scene_list = []
    # cc8_paired_scene_list = []
    
    # # Matching pipeline
    # make_pairing_list(cc8_scene_list,cc8_paired_scene_list,cc8_detection_filename,cc8_gt_filename,cc8_gt_ego)
    # # Serializing to json file
    # with open('dataset.json','w') as f:
    #     f.write(json.dumps(cc8_paired_scene_list[0].gtTrajDict,cls=AdvancedJSONEncoder,indent=2))
    # f.close()
    # print('Encode json file successfully!')
    # # Decode the json file to new TrajDict instance 
    # with open('dataset.json','r') as f:
    #     data_loaded =   json.load(f)
    # NEW_TrajDict = dict()
    # for key,value in data_loaded.items():
    #     NEW_TrajDict[key] = Trajectory(key)
    #     NEW_TrajDict[key].__dict__ = value
    # print('Decode json file successfully!')
    ############################## END of Demo on single scene #########################################
    ####################################################################################################
    ################################# Demo on multiple scene ###########################################
    ### Read all the files in the folder
    DTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/detection/*_detection*.txt")
    GTfile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/*_gt*.txt") 
    Egofile_list = glob.glob("/home/francis/Desktop/internship/apollo/HMM_dataset/text_dataset/ground_truth/ego/*_gt*.txt")
    ####
    Scene_list_sun = []
    Scene_list_night = []
    Scene_list_rain = []
    Scene_list_RainNight = []
    ####
    # paired_scene_list_sun = []
    # paired_scene_list_night = []
    # paired_scene_list_rain = []
    # paired_scene_list_RainNight = []
    paired_scene_list_sun = dict()
    paired_scene_list_night = dict()
    paired_scene_list_rain = dict()
    paired_scene_list_RainNight = dict()
    ####
    ## Matching all the dataset
    DTfile_list.sort(); GTfile_list.sort(); Egofile_list.sort()
    for DT, GT, Ego in zip(DTfile_list,GTfile_list,Egofile_list):
        if 'sun' in DT:
            make_pairing_list(Scene_list_sun,paired_scene_list_sun,DT,GT,Ego)
        elif 'night' in DT:
            make_pairing_list(Scene_list_night,paired_scene_list_night,DT,GT,Ego)
        elif 'rain' in DT:
            make_pairing_list(Scene_list_rain,paired_scene_list_rain,DT,GT,Ego)
        elif 'RainyNight' in DT:
            make_pairing_list(Scene_list_RainNight,paired_scene_list_RainNight,DT,GT,Ego)
        else:
            print('No weather type found!')
    # Serializing to json file, divided the weather type by file name
    # for sceene in paired_scene_list_sun:
    #     with open('AIOHMM_dataset_'+sceene.id+'_sun.json','w') as f:
    #         f.write(json.dumps(sceene.gtTrajDict,cls=AdvancedJSONEncoder,indent=2))
    #     f.close()
    if paired_scene_list_sun != []:
        with open('AIOHMM_dataset_sun.json','w') as f:
            f.write(json.dumps(paired_scene_list_sun,cls=AdvancedJSONEncoder,indent=2))
        f.close()
    
    if  paired_scene_list_night != []:
        with open('AIOHMM_dataset_night.json','w') as f:
            f.write(json.dumps(paired_scene_list_night,cls=AdvancedJSONEncoder,indent=2))
        f.close()
    
    if paired_scene_list_rain != []:
        with open('AIOHMM_dataset_rain.json','w') as f:
            f.write(json.dumps(paired_scene_list_rain,cls=AdvancedJSONEncoder,indent=2))
        f.close()
    
    if paired_scene_list_RainNight!=[]:
        with open('AIOHMM_dataset_RainyNight.json','w') as f:
            f.write(json.dumps(paired_scene_list_sun,cls=AdvancedJSONEncoder,indent=2))
        f.close()
    
    print('foo')
    # Example decode the json file to new TrajDict instance
    # with open('AIOHMM_dataset_sun.json','r') as f:
        # data_loaded =   json.load(f)
    ################################ END of Demo on multiple scene ########################################
