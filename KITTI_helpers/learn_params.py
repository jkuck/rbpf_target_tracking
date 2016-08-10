#Copied from KITTI devkit_tracking/python/evaluate_tracking.py and then edited


#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import sys,os,copy,math
from munkres import Munkres
from collections import defaultdict
try:
    from ordereddict import OrderedDict # can be installed using pip
except:
    from collections import OrderedDict # only included from python 2.7 on

import mailpy
from learn_Q import run_EM_on_Q_multiple_targets
from learn_Q import Target
from learn_Q import default_time_step
import pickle

MIN_SCORE = 2.0 #only consider detections with a score above this value

LEARN_Q_FROM_ALL_GT = False
SKIP_LEARNING_Q = True

CAMERA_PIXEL_WIDTH = 1242
CAMERA_PIXEL_HEIGHT = 375
#########################################################################
# function that does the evaluation
# input:
#   - det_method (method used for frame by frame detection and the name of the folder
#       where the detections are stored)
#   - mail (messenger object for output messages sent via email and to cout)
# output:
#   - True if at least one of the sub-benchmarks could be processed successfully
#   - False otherwise
# data:
#   - the results shall be saved as follows
#     -> summary statistics of the method: results/<det_method>/stats_task.txt
#        here task refers to the sub-benchmark (e.g., um_lane, uu_road etc.)
#        file contents: numbers for main table, format: %.6f (single space separated)
#        note: only files with successful sub-benchmark evaluation must be created
#     -> detailed results/graphics/plots: results/<det_method>/subdir
#        with appropriate subdir and file names (all subdir's need to be created)

class tData:
    def __init__(self,frame=-1,obj_type="unset",truncation=-1,occlusion=-1,\
                 obs_angle=-10,x1=-1,y1=-1,x2=-1,y2=-1,w=-1,h=-1,l=-1,\
                 X=-1000,Y=-1000,Z=-1000,yaw=-10,score=-1000,track_id=-1):

        # init object data
        self.frame      = frame
        self.track_id   = track_id
        self.obj_type   = obj_type
        self.truncation = truncation
        self.occlusion  = occlusion
        self.obs_angle  = obs_angle
        self.x1         = x1
        self.y1         = y1
        self.x2         = x2
        self.y2         = y2
        self.w          = w
        self.h          = h
        self.l          = l
        self.X          = X
        self.Y          = Y
        self.Z          = Z
        self.yaw        = yaw
        self.score      = score
        self.ignored    = False
        self.valid      = False
        self.tracker    = -1

    def __str__(self):
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())

class TargetSet:
    """
    Contains ground truth states for all targets.  Also contains all generated measurements.
    """

    def __init__(self):
        self.living_targets = []
        self.all_targets = [] #alive and dead targets

        self.living_count = 0 #number of living targets
        self.total_count = 0 #number of living targets plus number of dead targets
        self.measurements = [] #generated measurements for a generative TargetSet 

class Measurement:
    def __init__(self, time = -1):
        #self.val is a list of numpy arrays of measurement x, y locations
        self.val = []
        #list of widths of each bounding box
        self.widths = []
        #list of widths of each bounding box        
        self.heights = []
        self.time = time

class trackingEvaluation(object):
    """ tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
             MOTA	- Multi-object tracking accuracy in [0,100]
             MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
             MOTAL	- Multi-object tracking accuracy in [0,100] with log10(id-switches)

             id-switches - number of id switches
             fragments   - number of fragmentations

             MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories

             recall	        - recall = percentage of detected targets
             precision	    - precision = percentage of correctly detected targets
             FAR		    - number of false alarms per frame
             falsepositives - number of false positives (FP)
             missed         - number of missed targets (FN)
    """

    def __init__(self, det_method, gt_path="./data/training_ground_truth", min_overlap=0.5, max_truncation = 0.15, mail=None, cls="car"):
        #jdk parameters to learn
        self.clutter_count_list = []
        self.p_target_emission = -1
        self.birth_count_list = []
        self.discontinuous_target_count = 0
        self.discontinuous_target_ids = [] #these are ignored when computing death probabilities
        self.targ_cnt_still_living_but_unassoc_after_n = [-99]

        #freq_unassociated_frames_before_target_death[i] = j represents that j targets are unassociated
        #for i frames before dying
        self.freq_unassociated_frames_before_target_death = {}
        #targ_cnt_dead_at_n[i] is the number of targets that were:
        # -alive and associated with a measurement at time t
        # -alive and unassociated with a measurement at [t+1, t+i-1] (for i > 1)
        # -dead at time t+i
        self.targ_cnt_dead_at_n = [-99]

        #for i > 0, target_death_probabilities[i] = 
        #targ_cnt_dead_at_n[i] / (targ_cnt_dead_at_n[i] + targ_cnt_still_living_but_unassoc_after_n[i])
        #This is the probability that a target dies i frames after its last association
        self.target_death_probabilities = [-99]

        self.total_death_count = 0

        #groundTruthLocations[i] is a TargetSet of ground truth targets in sequence i
        self.groundTruthTargetSets = []
        self.meas_errors = []
        self.gt_widths = []
        self.gt_heights = []

        self.all_gt_targets_dict = {}
        self.Q_estimate = -99
        self.measurementTargetSetsBySequence = []

        # get number of sequences and
        # get number of frames per sequence from test mapping
        # (created while extracting the benchmark)
        filename_test_mapping = "./data/evaluate_tracking.seqmap"
        self.n_frames         = []
        self.sequence_name    = []
        with open(filename_test_mapping, "r") as fh:
            for i,l in enumerate(fh):
                fields = l.split(" ")
                self.sequence_name.append("%04d" % int(fields[0]))
                self.n_frames.append(int(fields[3]) - int(fields[2])+1) #jdk, why is there a +1 ???
        fh.close()                                
        self.n_sequences = i+1

        # mail object
        self.mail = mail

        # class to evaluate
        self.cls = cls

        # data and parameter
        self.gt_path           = os.path.join(gt_path, "label_02")

        self.det_method = det_method
        self.t_path            = os.path.join("./data/object_detections", self.det_method, "training/det_02")
        self.n_gt              = 0
        self.n_gt_trajectories = 0
        self.n_gt_seq          = []
        self.n_tr              = 0
        self.n_tr_trajectories = 0
        self.n_tr_seq          = []
        self.min_overlap       = min_overlap # minimum bounding box overlap for 3rd party metrics
        self.max_truncation    = max_truncation # maximum truncation of an object for evaluation
        self.n_sample_points   = 500
        # figures for evaluation
        self.MOTA              = 0
        self.MOTP              = 0
        self.MOTAL             = 0
        self.MODA              = 0
        self.MODP              = 0
        self.MODP_t            = []
        self.recall            = 0
        self.precision         = 0
        self.F1                = 0
        self.FAR               = 0
        self.total_cost        = 0
        self.tp                = 0
        self.fn                = 0
        self.fp                = 0
        self.mme               = 0
        self.fragments         = 0
        self.id_switches       = 0
        self.MT                = 0
        self.PT                = 0
        self.ML                = 0
        self.distance          = []
        self.seq_res           = []
        self.seq_output        = []
        # this should be enough to hold all groundtruth trajectories
        # is expanded if necessary and reduced in any case
        self.gt_trajectories   = [[] for x in xrange(self.n_sequences)] 
        self.ign_trajectories  = [[] for x in xrange(self.n_sequences)]

    def createEvalDir(self):
        """Creates directory to store evaluation results and data for visualization"""
        self.eval_dir = os.path.join("./data/object_detections", self.det_method, "eval", self.cls)
        if not os.path.exists(self.eval_dir):
            print "create directory:", self.eval_dir,
            os.makedirs(self.eval_dir)
            print "done"

    def loadGroundtruth(self):
        """Helper function to load ground truth"""
        try:
            self._loadData(self.gt_path, cls=self.cls, loading_groundtruth=True)
        except IOError:
            return False
        return True

    def loadDetections(self):
        """Helper function to load tracker data"""
        try:
            if not self._loadData(self.t_path, cls=self.cls, loading_groundtruth=False):
                return False
        except IOError:
            return False
        return True

    def _loadData(self, root_dir, cls, min_score=-1000, loading_groundtruth=False):
        """
            Generic loader for ground truth and tracking data.
            Use loadGroundtruth() or loadDetections() to load this data.
            Loads detections in KITTI format from textfiles.
        """
        # construct objectDetections object to hold detection data
        t_data  = tData()
        data    = []
        eval_2d = True
        eval_3d = True

        seq_data           = []
        n_trajectories     = 0
        n_trajectories_seq = []

        if not loading_groundtruth:
            fake_track_id = 0 #we'll assign a unique id to every detected object

        for seq, s_name in enumerate(self.sequence_name):
            i              = 0
            filename       = os.path.join(root_dir, "%s.txt" % s_name)
            f              = open(filename, "r") 

            f_data         = [[] for x in xrange(self.n_frames[seq])] # current set has only 1059 entries, sufficient length is checked anyway
            ids            = []
            n_in_seq       = 0
            id_frame_cache = []
            for line in f:
                if not loading_groundtruth:
                    fake_track_id += 1
                # KITTI tracking benchmark data format:
                # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
                line = line.strip()
                fields            = line.split(" ")
                # classes that should be loaded (ignored neighboring classes)
                if "car" in cls.lower():
                    classes = ["car","van"]
                elif "pedestrian" in cls.lower():
                    classes = ["pedestrian","person_sitting"]
                else:
                    classes = [cls.lower()]
                classes += ["dontcare"]
                if not any([s for s in classes if s in fields[2].lower()]):
                    continue
                # get fields from table
                t_data.frame        = int(float(fields[0]))     # frame
                if loading_groundtruth:
                    t_data.track_id     = int(float(fields[1]))     # id
                else: 
                    t_data.track_id = fake_track_id
                t_data.obj_type     = fields[2].lower()         # object type [car, pedestrian, cyclist, ...]
                t_data.truncation   = float(fields[3])          # truncation [0..1]
                t_data.occlusion    = int(float(fields[4]))     # occlusion  [0,1,2]
                t_data.obs_angle    = float(fields[5])          # observation angle [rad]
                t_data.x1           = float(fields[6])          # left   [px]
                t_data.y1           = float(fields[7])          # top    [px]
                t_data.x2           = float(fields[8])          # right  [px]
                t_data.y2           = float(fields[9])          # bottom [px]
                t_data.h            = float(fields[10])         # height [m]
                t_data.w            = float(fields[11])         # width  [m]
                t_data.l            = float(fields[12])         # length [m]
                t_data.X            = float(fields[13])         # X [m]
                t_data.Y            = float(fields[14])         # Y [m]
                t_data.Z            = float(fields[15])         # Z [m]
                t_data.yaw          = float(fields[16])         # yaw angle [rad]
                if not loading_groundtruth:
                    if len(fields) == 17:
                        t_data.score = -1
                    elif len(fields) == 18:
                        t_data.score  = float(fields[17])     # detection score
                        if t_data.score < MIN_SCORE:
                            continue
                    else:
                        self.mail.msg("file is not in KITTI format")
                        return

                # do not consider objects marked as invalid
                if t_data.track_id is -1 and t_data.obj_type != "dontcare":
                    continue

                idx = t_data.frame
                # check if length for frame data is sufficient
                if idx >= len(f_data):
                    print "extend f_data", idx, len(f_data)
                    f_data += [[] for x in xrange(max(500, idx-len(f_data)))]
                try:
                    id_frame = (t_data.frame,t_data.track_id)
                    if id_frame in id_frame_cache and not loading_groundtruth:
                        self.mail.msg("track ids are not unique for sequence %d: frame %d" % (seq,t_data.frame))
                        self.mail.msg("track id %d occured at least twice for this frame" % t_data.track_id)
                        self.mail.msg("Exiting...")
                        #continue # this allows to evaluate non-unique result files
                        return False
                    id_frame_cache.append(id_frame)
                    f_data[t_data.frame].append(copy.copy(t_data))
                except:
                    print len(f_data), idx
                    raise

                if t_data.track_id not in ids and t_data.obj_type!="dontcare":
                    ids.append(t_data.track_id)
                    n_trajectories +=1
                    n_in_seq +=1

                # check if uploaded data provides information for 2D and 3D evaluation
                if not loading_groundtruth and eval_2d is True and(t_data.x1==-1 or t_data.x2==-1 or t_data.y1==-1 or t_data.y2==-1):
                    eval_2d = False
                if not loading_groundtruth and eval_3d is True and(t_data.X==-1000 or t_data.Y==-1000 or t_data.Z==-1000):
                    eval_3d = False

            # only add existing frames
            n_trajectories_seq.append(n_in_seq)
            seq_data.append(f_data)
            f.close()

        if not loading_groundtruth:
            self.tracker=seq_data
            self.n_tr_trajectories=n_trajectories
            self.eval_2d = eval_2d
            self.eval_3d = eval_3d
            self.n_tr_seq = n_trajectories_seq
            if self.n_tr_trajectories==0:
                return False
        else: 
            # split ground truth and DontCare areas
            self.dcareas     = []
            self.groundtruth = []
            for seq_idx in range(len(seq_data)):
                seq_gt = seq_data[seq_idx]
                s_g, s_dc = [],[]
                for f in range(len(seq_gt)):
                    all_gt = seq_gt[f]
                    g,dc = [],[]
                    for gg in all_gt:
                        if gg.obj_type=="dontcare":
                            dc.append(gg)
                        else:
                            g.append(gg)
                    s_g.append(g)
                    s_dc.append(dc)
                self.dcareas.append(s_dc)
                self.groundtruth.append(s_g)
            self.n_gt_seq=n_trajectories_seq
            self.n_gt_trajectories=n_trajectories
        return True
            
            
    def boxoverlap(self,a,b,criterion="union"):
        """
            boxoverlap computes intersection over union for bbox a and b in KITTI format.
            If the criterion is 'union', overlap = (a inter b) / a union b).
            If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
        """
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        
        w = x2-x1
        h = y2-y1

        if w<=0. or h<=0.:
            return 0.
        inter = w*h
        aarea = (a.x2-a.x1) * (a.y2-a.y1)
        barea = (b.x2-b.x1) * (b.y2-b.y1)
        # intersection over union overlap
        if criterion.lower()=="union":
            o = inter / float(aarea+barea-inter)
        elif criterion.lower()=="a":
            o = float(inter) / float(aarea)
        else:
            raise TypeError("Unkown type for criterion")
        return o

    def compute3rdPartyMetrics(self):
        """
            Computes the metrics defined in 
                - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
                  MOTA, MOTAL, MOTP
                - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows
                  MT/PT/ML
        """

        # construct Munkres object for Hungarian Method association
        hm = Munkres()
        max_cost = 1e9


        #find the frequency of clutter (false positive) counts
        #clutter_count_dict[5] = 40 will mean that 40 frames contain 5 clutter measurements
        clutter_count_dict = {}

        #find the frequency of birth counts
        #birth_count_freq_dict[2] = 25 will mean that 25 ground truth frames contain 2 births 
        #(ground truth objects that have never been seen before)
        birth_count_freq_dict = {}

        #all_associated_gt_ids is a list of all associated ground truth objects in each video sequence
        #all_associated_gt_ids[i] is a list of all associated ground truth objects in the ith video sequence
        #all_associated_gt_ids[i][j] is a list of all associated ground truth objects in the jth frame of the ith video sequence
        #all_associated_gt_ids[i][j][k] is the track_id of the kth associated ground truth object in the jth frame of the ith video sequence
        #an "associated ground truth object" refers to a ground truth object that is associated
        #with a measurement in the current frame
        all_associated_gt_ids = [] #jdk
        all_associated_gt_locations = [] #jdk

        #living_ground_truth_objects_by_frame is a list of all living ground truth objects in each video sequence
        #living_ground_truth_objects_by_frame[i] is a list of all living ground truth objects in the ith video sequence
        #living_ground_truth_objects_by_frame[i][j] is a list of all living ground truth objects in the jth frame of the ith video sequence
        #living_ground_truth_objects_by_frame[i][j][k] is the track_id of the kth living ground truth object in the jth frame of the ith video sequence

        living_ground_truth_objects_by_frame = []#jdk

        # go through all frames and associate ground truth and tracker results
        # groundtruth and tracker contain lists for every single frame containing lists of KITTI format detections
        fr, ids = 0,0 
        num_frames = 0
        total_target_count = 0
        visible_target_count = 0
        for seq_idx in range(len(self.groundtruth)):
            x_min = 9999999
            x_max = -9999999
            y_min = 9999999
            y_max = -9999999

            all_associated_gt_ids.append([]) #jdk
            all_associated_gt_locations.append([]) #jdk
            living_ground_truth_objects_by_frame.append([]) #jdk

            seq_gt           = self.groundtruth[seq_idx]
            seq_dc           = self.dcareas[seq_idx]
            seq_tracker      = self.tracker[seq_idx]
            seq_trajectories = defaultdict(list)
            seq_ignored      = defaultdict(list)
            seqtp            = 0
            seqfn            = 0
            seqfp            = 0
            seqcost          = 0

            last_ids = [[],[]]
            tmp_frags = 0

            existing_targets = []
            #frames_gt_targets_appear_in[i] is a list of the frames the target with id i appears in
            frames_gt_targets_appear_in = {}

            print "!!!!!!!!!!", len(seq_gt)
            cur_discontinuous_target_count = 0

            cur_seq_meas_target_set = TargetSet()

            for f in range(len(seq_gt)):
                all_associated_gt_ids[-1].append([]) #jdk
                all_associated_gt_locations[-1].append([]) #jdk


                living_ground_truth_objects_by_frame[-1].append([]) #jdk

                g = seq_gt[f]
                dc = seq_dc[f]
                        
                t = seq_tracker[f]
                # counting total number of ground truth and tracker objects
                self.n_gt += len(g)
                self.n_tr += len(t)

                # use hungarian method to associate, using boxoverlap 0..1 as cost
                # build cost matrix
                cost_matrix = []
                this_ids = [[],[]]
                birth_count = 0 # jdk

                #begin jdk, extract measurements
                cur_frame_measurements = Measurement()
                cur_frame_measurements.time = f*.1 #frames are .1 seconds apart
                for tt in t:
                    meas_pos = np.array([(tt.x2 + tt.x1)/2.0, (tt.y2 + tt.y1)/2.0])
                    meas_width = tt.x2 - tt.x1
                    meas_height = tt.y2 - tt.y1
                    cur_frame_measurements.val.append(meas_pos)
                    cur_frame_measurements.widths.append(meas_width)
                    cur_frame_measurements.heights.append(meas_height)
                cur_seq_meas_target_set.measurements.append(cur_frame_measurements)
                #end jdk, extract measurements


                for gg in g:
                    #jdk
                    if not gg.track_id in existing_targets:
                        birth_count += 1
                        existing_targets.append(gg.track_id)
                    if gg.track_id in frames_gt_targets_appear_in:
                        frames_gt_targets_appear_in[gg.track_id].append(f)
                    else:
                        frames_gt_targets_appear_in[gg.track_id] = [f]                

                    assert(seq_idx + 1 == len(living_ground_truth_objects_by_frame)), (seq_idx, len(living_ground_truth_objects_by_frame))
                    assert(f + 1 == len(living_ground_truth_objects_by_frame[-1]))
                    living_ground_truth_objects_by_frame[-1][-1].append(gg.track_id)

                    if gg.x1 < x_min:
                        x_min = gg.x1
                    if gg.y1 < y_min:
                        y_min = gg.y1
                    if gg.x2 > x_max:
                        x_max = gg.x2
                    if gg.y2 > y_max:
                        y_max = gg.y2

                    if LEARN_Q_FROM_ALL_GT:
                        #learn Q from all gt locations
                        gt_pos_format2 = np.array([[(gg.x2 + gg.x1)/2.0], 
                                                   [(gg.y2 + gg.y1)/2.0]])
                        if (seq_idx, gg.track_id) in self.all_gt_targets_dict:
                            self.all_gt_targets_dict[(seq_idx, gg.track_id)].measurements.append(gt_pos_format2)
                            self.all_gt_targets_dict[(seq_idx, gg.track_id)].measurement_time_stamps.append(f*default_time_step)
    
                        else:
                            self.all_gt_targets_dict[(seq_idx, gg.track_id)] = Target(gt_pos_format2)
                            self.all_gt_targets_dict[(seq_idx, gg.track_id)].measurements.append(gt_pos_format2)
                            self.all_gt_targets_dict[(seq_idx, gg.track_id)].measurement_time_stamps.append(f*default_time_step)


                    #jdk


                    # save current ids
                    this_ids[0].append(gg.track_id)
                    this_ids[1].append(-1)
                    gg.tracker       = -1
                    gg.id_switch     = 0
                    gg.fragmentation = 0
                    cost_row         = []
                    for tt in t:
                        # overlap == 1 is cost ==0
                        c = 1-self.boxoverlap(gg,tt)
                        # gating for boxoverlap
                        if c<=self.min_overlap:
                            cost_row.append(c)
                        else:
                            cost_row.append(max_cost)
                    cost_matrix.append(cost_row)
                    # all ground truth trajectories are initially not associated
                    # extend groundtruth trajectories lists (merge lists)
                    seq_trajectories[gg.track_id].append(-1)
                    seq_ignored[gg.track_id].append(False)

                if birth_count in birth_count_freq_dict:
                    birth_count_freq_dict[birth_count] += 1
                else:
                    birth_count_freq_dict[birth_count] = 1

                if len(g) is 0:
                    cost_matrix=[[]]
                # associate
                association_matrix = hm.compute(cost_matrix)

                # mapping for tracker ids and ground truth ids
                tmptp = 0
                tmpfp = 0
                tmpfn = 0
                tmpc  = 0
                this_cost = [-1]*len(g)
                for row,col in association_matrix:
                    # apply gating on boxoverlap
                    c = cost_matrix[row][col]
                    if c < max_cost:
                        #jdk begin
                        assert(seq_idx + 1 == len(all_associated_gt_ids))
                        assert(f + 1 == len(all_associated_gt_ids[-1]))
                        assert(seq_idx + 1 == len(all_associated_gt_locations))
                        assert(f + 1 == len(all_associated_gt_locations[-1]))

                        all_associated_gt_ids[-1][-1].append(g[row].track_id)
                        all_associated_gt_locations[-1][-1].append((g[row].x1, g[row].x2, g[row].y1, g[row].y2))

                        gt_pos = np.array([(g[row].x2 + g[row].x1)/2.0, (g[row].y2 + g[row].y1)/2.0])
                        meas_pos = np.array([(t[col].x2 + t[col].x1)/2.0, (t[col].y2 + t[col].y1)/2.0])
                        self.meas_errors.append(meas_pos - gt_pos)
                        self.gt_widths.append(g[row].x2 - g[row].x1)
                        self.gt_heights.append(g[row].y2 - g[row].y1)

                        if(not LEARN_Q_FROM_ALL_GT):
                            #only learn Q based on gt locations that are associated with detections of the current algorithm   
                            gt_pos_format2 = np.array([[(g[row].x2 + g[row].x1)/2.0], 
                                                       [(g[row].y2 + g[row].y1)/2.0]])                     
                            if (seq_idx, g[row].track_id) in self.all_gt_targets_dict:
                                self.all_gt_targets_dict[(seq_idx, g[row].track_id)].measurements.append(gt_pos_format2)
                                self.all_gt_targets_dict[(seq_idx, g[row].track_id)].measurement_time_stamps.append(f*default_time_step)    

                            else:
                                self.all_gt_targets_dict[(seq_idx, g[row].track_id)] = Target(gt_pos_format2)
                                self.all_gt_targets_dict[(seq_idx, g[row].track_id)].measurements.append(gt_pos_format2)
                                self.all_gt_targets_dict[(seq_idx, g[row].track_id)].measurement_time_stamps.append(f*default_time_step)

                        #jdk end

                        g[row].tracker   = t[col].track_id
                        this_ids[1][row] = t[col].track_id
                        t[col].valid     = True
                        g[row].distance  = c
                        self.total_cost += 1-c
                        seqcost         += 1-c
                        tmpc            += 1-c
                        seq_trajectories[g[row].track_id][-1] = t[col].track_id

                        # true positives are only valid associations
                        self.tp += 1
                        tmptp   += 1
                        this_cost.append(c)
                    else:
                        g[row].tracker = -1
                        self.fn       += 1
                        tmpfn         += 1

                # associate tracker and DontCare areas
                # ignore tracker in neighboring classes
                nignoredtracker = 0
                for tt in t:
                    if (self.cls=="car" and tt.obj_type=="van") or (self.cls=="pedestrian" and tt.obj_type=="person_sitting"):
                        nignoredtracker+= 1
                        tt.ignored      = True
                        continue
                    for d in dc:
                        overlap = self.boxoverlap(tt,d,"a")
                        if overlap>0.5 and not tt.valid:
                            tt.ignored      = True
                            nignoredtracker+= 1
                            break

                # check for ignored FN/TP (truncation or neighboring object class)
                ignoredfn  = 0
                nignoredtp = 0
                for gg in g:
                    if gg.tracker < 0:
                        # ignored FN due to truncation
                        if gg.truncation>self.max_truncation:
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            ignoredfn += 1
                        # ignored FN due to neighboring object class
                        elif (self.cls=="car" and gg.obj_type=="van") or (self.cls=="pedestrian" and gg.obj_type=="person_sitting"):
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            ignoredfn += 1
                    elif gg.tracker>=0:
                        # ignored TP due to truncation
                        if gg.truncation>self.max_truncation:
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            nignoredtp += 1
                        # ignored TP due nieghboring object class
                        elif (self.cls=="car" and gg.obj_type=="van") or (self.cls=="pedestrian" and gg.obj_type=="person_sitting"):
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            nignoredtp += 1

                # correct TP by number of ignored TP due to truncation
                # ignored TP are shown as tracked in visualization
                tmptp -= nignoredtp
                self.n_gt -= (ignoredfn + nignoredtp)

                # false negatives = associated gt bboxes exceding association threshold + non-associated gt bboxes
                tmpfn   += len(g)-len(association_matrix)-ignoredfn
                self.fn += len(g)-len(association_matrix)-ignoredfn
                # false positives = tracker bboxes - associated tracker bboxes
                # mismatches (mme_t) 
                tmpfp   += len(t) - tmptp - nignoredtracker - nignoredtp
                self.fp += len(t) - tmptp - nignoredtracker - nignoredtp



                num_frames += 1
                if ((len(t) - tmptp - nignoredtracker - nignoredtp) in clutter_count_dict):
                    clutter_count_dict[len(t) - tmptp - nignoredtracker - nignoredtp] += 1
                else:
                    clutter_count_dict[len(t) - tmptp - nignoredtracker - nignoredtp] = 1

                total_target_count += tmptp + tmpfn
                visible_target_count += tmptp

                # append single distance values
                self.distance.append(this_cost)

                # update sequence data
                seqtp += tmptp
                seqfp += tmpfp
                seqfn += tmpfn

                # sanity checks
                if tmptp + tmpfn is not len(g)-ignoredfn-nignoredtp:
                    print "seqidx", seq_idx
                    print "frame ", f
                    print "TP    ", tmptp
                    print "FN    ", tmpfn
                    print "FP    ", tmpfp
                    print "nGT   ", len(g)
                    print "nAss  ", len(association_matrix)
                    print "ign GT", ignoredfn
                    print "ign TP", nignoredtp
                    raise NameError("Something went wrong! nGroundtruth is not TP+FN")
                if tmptp+tmpfp+nignoredtracker+nignoredtp is not len(t):
                    print seq_idx, f, len(t), tmptp, tmpfp
                    print len(association_matrix), association_matrix
                    raise NameError("Something went wrong! nTracker is not TP+FP")

                # check for id switches or fragmentations
                for i,tt in enumerate(this_ids[0]):
                    if tt in last_ids[0]:
                        idx = last_ids[0].index(tt)
                        tid = this_ids[1][i]
                        lid = last_ids[1][idx]
                        if tid != lid and lid != -1 and tid != -1:
                            if g[i].truncation<self.max_truncation:
                                g[i].id_switch = 1
                                ids +=1
                        if tid != lid and lid != -1:
                            if g[i].truncation < self.max_truncation:
                                g[i].fragmentation = 1
                                tmp_frags +=1
                                fr +=1    

                # save current index
                last_ids = this_ids
                # compute MOTP_t
                MODP_t = 0
                if tmptp!=0:
                    MODP_t = tmpc/float(tmptp)
                self.MODP_t.append(MODP_t)


            self.measurementTargetSetsBySequence.append(cur_seq_meas_target_set)                


            # remove empty lists for current gt trajectories
            self.gt_trajectories[seq_idx]  = seq_trajectories
            self.ign_trajectories[seq_idx] = seq_ignored

            #jdk begin

            print x_min, x_max, y_min, y_max

            #frames_gt_targets_appear_in[i] is a list of the frames the target with id i appears in
            for target_id, frames in frames_gt_targets_appear_in.iteritems():
                for idx in range(len(frames)):
                    if(idx != 0 and frames[idx] != frames[idx-1] + 1):
                        self.discontinuous_target_count += 1
                        self.discontinuous_target_ids.append((seq_idx, target_id))
                        cur_discontinuous_target_count += 1
                        #assert(frames[idx] == frames[idx-1] + 1), frames
            print "cur_discontinuous_target_count = ", cur_discontinuous_target_count

            #jdk end


        #jdk begin
        f = open("KITTI_measurements_%s_%s_min_score_%s.pickle" % (self.cls, self.det_method, MIN_SCORE), 'w')
        pickle.dump(self.measurementTargetSetsBySequence, f)
        f.close()

        max_clutter_count = 0
        freq_sum = 0
        for clutter_count, frequency in clutter_count_dict.iteritems():
            freq_sum += frequency
            if clutter_count > max_clutter_count:
                max_clutter_count = clutter_count
        for clutter_count in range(max_clutter_count + 1):
            if clutter_count in clutter_count_dict:
                self.clutter_count_list.append(clutter_count_dict[clutter_count]/float(num_frames))
            else:
                self.clutter_count_list.append(0)
        print freq_sum, num_frames
        assert(abs(sum(self.clutter_count_list) - 1.0) < .000001), sum(self.clutter_count_list)

        self.p_target_emission = visible_target_count/float(total_target_count)

        max_birth_count = 0
        for birth_count, frequency in birth_count_freq_dict.iteritems():
            if birth_count > max_birth_count:
                max_birth_count = birth_count
        self.birth_count_list = []
        for birth_count in range(max_birth_count + 1):
            if birth_count in birth_count_freq_dict:
                self.birth_count_list.append(birth_count_freq_dict[birth_count]/float(num_frames))
            else:
                self.birth_count_list.append(0)



        #targ_cnt_dead_at_n[i] is the number of targets that were:
        # -alive and associated with a measurement at time t
        # -alive and unassociated with a measurement at [t+1, t+i-1] (for i > 1)
        # -dead at time t+i

        never_associated_gt_target_count = 0

        for seq_idx in range(len(all_associated_gt_ids)):
            for frame_idx in range(len(all_associated_gt_ids[seq_idx]) - 1):
                for cur_gt_obj_id in living_ground_truth_objects_by_frame[seq_idx][frame_idx]:
                    if(not cur_gt_obj_id in living_ground_truth_objects_by_frame[seq_idx][frame_idx + 1]):
                        self.total_death_count += 1
                        if(not (seq_idx, cur_gt_obj_id) in self.discontinuous_target_ids):
                            unassoc_frames_before_death = 0
                            while(frame_idx - unassoc_frames_before_death >= 0 and not cur_gt_obj_id in all_associated_gt_ids[seq_idx][frame_idx - unassoc_frames_before_death]):
                                unassoc_frames_before_death += 1
                            if frame_idx - unassoc_frames_before_death == 0:
                                never_associated_gt_target_count += 1
                            if unassoc_frames_before_death in self.freq_unassociated_frames_before_target_death:
                                self.freq_unassociated_frames_before_target_death[unassoc_frames_before_death] += 1
                            else:
                                self.freq_unassociated_frames_before_target_death[unassoc_frames_before_death] = 1

        print "!!!!!!!!!!!!!!!!!!!!!never_associated_gt_target_count = ", never_associated_gt_target_count, "!!!!!!"
        delay_n = 0
        while(self.targ_cnt_still_living_but_unassoc_after_n[-1] != 0):
            self.targ_cnt_still_living_but_unassoc_after_n.append(0)
            self.targ_cnt_dead_at_n.append(0)

            delay_n += 1
            for seq_idx in range(len(all_associated_gt_ids)):
                for frame_idx in range(len(all_associated_gt_ids[seq_idx]) - delay_n):
                    #go through all associated gt objects frame by frame
                    for cur_obj_index in range(len(all_associated_gt_ids[seq_idx][frame_idx])):
                        cur_gt_obj_id = all_associated_gt_ids[seq_idx][frame_idx][cur_obj_index]
                        #ignore ground truth objects that disappear without dying (should only be 2 in training data)
                        if(not (seq_idx, cur_gt_obj_id) in self.discontinuous_target_ids and \
                             (all_associated_gt_locations[seq_idx][frame_idx][cur_obj_index][0] < 10 or \
                             all_associated_gt_locations[seq_idx][frame_idx][cur_obj_index][1] > CAMERA_PIXEL_WIDTH - 15 or \
                             all_associated_gt_locations[seq_idx][frame_idx][cur_obj_index][2] < 10 or \
                             all_associated_gt_locations[seq_idx][frame_idx][cur_obj_index][3] > CAMERA_PIXEL_HEIGHT - 15 )):
                            #is cur_gt_obj_id still alive at frame_idx+delay_n?
                            if (cur_gt_obj_id in living_ground_truth_objects_by_frame[seq_idx][frame_idx + delay_n]):
                                unassoc = True # is cur_gt_obj_id unassociated in frames frame_idx+1 through frame_idx+delay_n?
#                                for t_idx in range(1, delay_n+1): #THIS IS CORRECT
                                for t_idx in range(1, delay_n): #THIS IS HACKY TESING original model with deaths sampled, and then associations
                                    if cur_gt_obj_id in all_associated_gt_ids[seq_idx][frame_idx + t_idx]:
                                        unassoc = False
                                if(unassoc):
                                    self.targ_cnt_still_living_but_unassoc_after_n[-1] += 1
                            #did the object just die
                            elif(cur_gt_obj_id in living_ground_truth_objects_by_frame[seq_idx][frame_idx + delay_n - 1]):
                                unassoc = True # is cur_gt_obj_id unassociated in frames frame_idx+1 through frame_idx+delay_n?
                                for t_idx in range(1, delay_n):
                                    if cur_gt_obj_id in all_associated_gt_ids[seq_idx][frame_idx + t_idx]:
                                        unassoc = False
                                if(unassoc):
                                    self.targ_cnt_dead_at_n[-1] += 1

        for death_delay in range(1, len(self.targ_cnt_dead_at_n)-1):
            self.target_death_probabilities.append(float(self.targ_cnt_dead_at_n[death_delay]) / \
                (self.targ_cnt_dead_at_n[death_delay] + self.targ_cnt_still_living_but_unassoc_after_n[death_delay]))
        

        all_targets_list = []
        for key, target in self.all_gt_targets_dict.iteritems():
            all_targets_list.append(target)

        if not SKIP_LEARNING_Q:
            (log_likelihoods, self.Q_estimate) = run_EM_on_Q_multiple_targets(all_targets_list)
            print '@'*80
            print "log_likelihoods:"
            print log_likelihoods
            print '@'*80
        #jdk end

        # compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
        n_ignored_tr_total = 0
        for seq_idx, (seq_trajectories,seq_ignored) in enumerate(zip(self.gt_trajectories, self.ign_trajectories)):
            if len(seq_trajectories)==0:
                continue
            tmpMT, tmpML, tmpPT, tmpId_switches, tmpFragments = [0]*5
            n_ignored_tr = 0
            for g, ign_g in zip(seq_trajectories.values(), seq_ignored.values()):
                # all frames of this gt trajectory are ignored
                if all(ign_g):
                    n_ignored_tr+=1
                    n_ignored_tr_total+=1
                    continue
                if all([this==-1 for this in g]):
                    tmpML+=1
                    self.ML+=1
                    continue
                # compute tracked frames in trajectory
                last_id = g[0]
                # first detection (necessary to be in gt_trajectories) is always tracked
                tracked = 1 if g[0]>=0 else 0
                lgt = 0 if ign_g[0] else 1
                for f in range(1,len(g)):
                    if ign_g[f]:
                        last_id = -1
                        continue
                    lgt+=1
                    if last_id != g[f] and last_id != -1 and g[f] != -1 and g[f-1] != -1:
                        tmpId_switches   += 1
                        self.id_switches += 1
                    if f < len(g)-1 and g[f-1] != g[f] and last_id != -1  and g[f] != -1 and g[f+1] != -1:
                        tmpFragments   += 1
                        self.fragments += 1
                    if g[f] != -1:
                        tracked += 1
                        last_id = g[f]
                # handle last frame; tracked state is handeled in for loop (g[f]!=-1)
                if len(g)>1 and g[f-1] != g[f] and last_id != -1  and g[f] != -1 and not ign_g[f]:
                    tmpFragments   += 1
                    self.fragments += 1

                # compute MT/PT/ML
                tracking_ratio = tracked/float(len(g))
                if tracking_ratio > 0.8:
                    tmpMT   += 1
                    self.MT += 1
                elif tracking_ratio < 0.2:
                    tmpML   += 1
                    self.ML += 1
                else: # 0.2 <= tracking_ratio <= 0.8
                    tmpPT   += 1
                    self.PT += 1

        if (self.n_gt_trajectories-n_ignored_tr_total)==0:
            self.MT = 0.
            self.PT = 0.
            self.ML = 0.
        else:
            self.MT /= float(self.n_gt_trajectories-n_ignored_tr_total)
            self.PT /= float(self.n_gt_trajectories-n_ignored_tr_total)
            self.ML /= float(self.n_gt_trajectories-n_ignored_tr_total)

        # precision/recall etc.
        if (self.fp+self.tp)==0 or (self.tp+self.fn)==0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = self.tp/float(self.tp+self.fn)
            self.precision = self.tp/float(self.fp+self.tp)
        if (self.recall+self.precision)==0:
            self.F1 = 0.
        else:
            self.F1 = 2.*(self.precision*self.recall)/(self.precision+self.recall)
        if sum(self.n_frames)==0:
            self.FAR = "n/a"
        else:
            self.FAR = self.fp/float(sum(self.n_frames))

        # compute CLEARMOT
        if self.n_gt==0:
            self.MOTA = -float("inf")
            self.MODA = -float("inf")
        else:
            self.MOTA  = 1 - (self.fn + self.fp + self.id_switches)/float(self.n_gt)
            self.MODA  = 1 - (self.fn + self.fp) / float(self.n_gt)
        if self.tp==0:
            self.MOTP  = float("inf")
        else:
            self.MOTP  = self.total_cost / float(self.tp)
        if self.n_gt!=0:
            if self.id_switches==0:
                self.MOTAL = 1 - (self.fn + self.fp + self.id_switches)/float(self.n_gt)
            else:
                self.MOTAL = 1 - (self.fn + self.fp + math.log10(self.id_switches))/float(self.n_gt)
        else:
            self.MOTAL = -float("inf")
        if sum(self.n_frames)==0:
            self.MODP = "n/a"
        else:
            self.MODP = sum(self.MODP_t)/float(sum(self.n_frames))
        return True

    def summary(self):
        mail.msg('-'*80)
        mail.msg("jdk's learned parameters".center(20,"#"))
        mail.msg(self.printEntry("clutter count probabilities: ", self.clutter_count_list))
        mail.msg(self.printEntry("p_target_emission: ", self.p_target_emission))
        mail.msg(self.printEntry("birth count probabilities: ", self.birth_count_list))
        mail.msg(self.printEntry("discontinuous_target_count (should be small!!, ignored when computing death probabilities): ", self.discontinuous_target_count))
        mail.msg(self.printEntry("targ_cnt_still_living_but_unassoc_after_n: ", self.targ_cnt_still_living_but_unassoc_after_n))
        mail.msg(self.printEntry("targ_cnt_dead_at_n: ", self.targ_cnt_dead_at_n))
        mail.msg(self.printEntry("total_death_count: ", self.total_death_count))
        mail.msg(self.printEntry("target_death_probabilities: ", self.target_death_probabilities))
        mail.msg(self.printEntry("freq_unassociated_frames_before_target_death: ", self.freq_unassociated_frames_before_target_death))
        mail.msg(self.printEntry("measurement error covariance matrix: ", np.cov(np.asarray(self.meas_errors).T)))
        mail.msg(self.printEntry("measurement error means: ", np.mean(np.asarray(self.meas_errors), 0)))
        mail.msg(self.printEntry("ground truth mean width: ", sum(self.gt_widths)/float(len(self.gt_widths))))
        mail.msg(self.printEntry("ground truth mean height: ", sum(self.gt_heights)/float(len(self.gt_heights))))
        mail.msg(self.printEntry("estimated Q: ", self.Q_estimate))


        mail.msg("tracking evaluation summary".center(80,"="))
        mail.msg(self.printEntry("Multiple Object Tracking Accuracy (MOTA)", self.MOTA))
        mail.msg(self.printEntry("Multiple Object Tracking Precision (MOTP)", self.MOTP))
        mail.msg(self.printEntry("Multiple Object Tracking Accuracy (MOTAL)", self.MOTAL))
        mail.msg(self.printEntry("Multiple Object Detection Accuracy (MODA)", self.MODA))
        mail.msg(self.printEntry("Multiple Object Detection Precision (MODP)", self.MODP))
        mail.msg("")
        mail.msg(self.printEntry("Recall", self.recall))
        mail.msg(self.printEntry("Precision", self.precision))
        mail.msg(self.printEntry("F1", self.F1))
        mail.msg(self.printEntry("False Alarm Rate", self.FAR))
        mail.msg("")
        mail.msg(self.printEntry("Mostly Tracked", self.MT))
        mail.msg(self.printEntry("Partly Tracked", self.PT))
        mail.msg(self.printEntry("Mostly Lost", self.ML))
        mail.msg("")
        mail.msg(self.printEntry("True Positives", self.tp))
        mail.msg(self.printEntry("False Positives", self.fp))
        mail.msg(self.printEntry("Missed Targets", self.fn))
        mail.msg(self.printEntry("ID-switches", self.id_switches))
        mail.msg(self.printEntry("Fragmentations", self.fragments))
        mail.msg("")
        mail.msg(self.printEntry("Ground Truth Objects", self.n_gt))
        mail.msg(self.printEntry("Ground Truth Trajectories", self.n_gt_trajectories))
        mail.msg(self.printEntry("Tracker Objects", self.n_tr))
        mail.msg(self.printEntry("Tracker Trajectories", self.n_tr_trajectories))
        mail.msg("="*80)
        #self.saveSummary()

    def printEntry(self, key, val,width=(43,10)):
        s_out =  key.ljust(width[0])
        if type(val)==int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val)==float:
            s = "%%%df" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s"%val).rjust(width[1])
        return s_out

    def saveSummary(self):
        filename = os.path.join("./data/object_detections", self.det_method, "3rd_party_metrics.txt")
        open(filename, "w").close()
        dump = open(filename, "a")
        print>>dump, "MOTA,", self.MOTA
        print>>dump, "MOTP,", self.MOTP
        print>>dump, "MOTAL,", self.MOTAL
        print>>dump, "MODA,", self.MODA
        print>>dump, "MODP,", self.MODP
        #print>>dump, ""
        print>>dump, "Recall,", self.recall
        print>>dump, "Precision,", self.precision
        print>>dump, "F1,", self.F1
        print>>dump, "FAR,", self.FAR
        #print>>dump, ""
        print>>dump, "MT,", self.MT
        print>>dump, "PT,", self.PT
        print>>dump, "ML,", self.ML
        #print>>dump, ""
        print>>dump, "TP,", self.tp
        print>>dump, "FP,", self.fp
        print>>dump, "Misses,", self.fn
        print>>dump, "ID-switches,", self.id_switches
        print>>dump, "Fragmentations,", self.fragments
        #print>>dump, ""
        print>>dump, "Ground Truth Objects,", self.n_gt
        print>>dump, "Ground Truth Trajectories,", self.n_gt_trajectories
        print>>dump, "Tracker Objects,", self.n_tr 
        print>>dump, "Tracker Trajectories,", self.n_tr_trajectories
        dump.close()

    def saveToStats(self):
        self.summary()
        filename = os.path.join("./data/object_detections", self.det_method, "stats_%s.txt" % self.cls)
        dump = open(filename, "w+")
        print>>dump, "%.6f " * 21 \
                % (self.MOTA, self.MOTP, self.MOTAL, self.MODA, self.MODP, \
                   self.recall, self.precision, self.F1, self.FAR, \
                   self.MT, self.PT, self.ML, self.tp, self.fp, self.fn, self.id_switches, self.fragments, \
                   self.n_gt, self.n_gt_trajectories, self.n_tr, self.n_tr_trajectories)
        dump.close()
        filename = os.path.join("./data/object_detections", self.det_method, "description.txt")
        dump = open(filename, "w+")
        print>>dump, "MOTA", "MOTP", "MOTAL", "MODA", "MODP", "recall", "precision", "F1", "FAR",
        print>>dump, "MT", "PT", "ML", "tp", "fp", "fn", "id_switches", "fragments",
        print>>dump, "n_gt", "n_gt_trajectories", "n_tr", "n_tr_trajectories"

    def sequenceSummary(self):
        filename = os.path.join("./data/object_detections", self.det_method, self.dataset, "sequences.txt")
        open(filename, "w").close()
        dump = open(filename, "a")

        self.printSep("Sequence Evaluation")
        self.printSep()
        print "seq\t", "\t".join(self.seq_res[0].keys())
        print>>dump, "seq\t", "\t".join(self.seq_res[0].keys())
        for i,s in enumerate(self.seq_res):
            print i,"\t",
            print>>dump, i,"\t",
            for e in s.values():
                if type(e) is int:
                    print "%d" % e, "\t",
                    print>>dump,"%d\t" % e,                                                 
                elif type(e) is float:
                    print "%.3f" % e, "\t",
                    print>>dump, "%.3f\t" % e,
                else:
                    print "%s" % e, "\t",
                    print>>dump, "%s\t" % e,
            print ""
            print>>dump, ""

        self.printSep()
        dump.close()

def evaluate(det_method,mail):
    # start evaluation and instanciated eval object
    mail.msg("Processing Result for KITTI Tracking Benchmark")
    classes = []
    for c in ("car", "pedestrian"):
        e = trackingEvaluation(det_method=det_method, mail=mail,cls=c)
        # load tracker data and check provided classes
        try:
            if not e.loadDetections():
                continue
            mail.msg("Loading Results - Success")
            mail.msg("Evaluate Object Class: %s" % c.upper())
            classes.append(c)
        except:
            mail.msg("Feel free to contact us (lenz@kit.edu), if you receive this error message:")
            mail.msg("   Caught exception while loading result data.")
            break
        # load groundtruth data for this class
        if not e.loadGroundtruth():
            raise ValueError("Ground truth not found.")
        mail.msg("Loading Groundtruth - Success")
        # sanity checks
        if len(e.groundtruth) is not len(e.tracker):
            mail.msg("The uploaded data does not provide results for every sequence.")
            return False
        mail.msg("Loaded %d Sequences." % len(e.groundtruth))
        mail.msg("Start Evaluation...")
        # create needed directories, evaluate and save stats
        try:
            e.createEvalDir()
        except:
            mail.msg("Feel free to contact us (lenz@kit.edu), if you receive this error message:")
            mail.msg("   Caught exception while creating results.")
        if e.compute3rdPartyMetrics():
            e.saveToStats()
        else:
            mail.msg("There seem to be no true positives or false positives at all in the submitted data.")

    # finish
    if len(classes)==0:
        mail.msg("The uploaded results could not be evaluated. Check for format errors.")
        return False
    mail.msg("Thank you for participating in our benchmark!")
    return True

#########################################################################
# entry point of evaluation script
# input:
#   - det_method (method used for frame by frame object detection)
#   - user_sha (key of user who submitted the results, optional)
#   - user_sha (email of user who submitted the results, optional)
if __name__ == "__main__":

    # check for correct number of arguments. if user_sha and email are not supplied,
    # no notification email is sent (this option is used for auto-updates)
    if len(sys.argv)!=2 or (sys.argv[1] != 'lsvm' and sys.argv[1] != 'regionlets'):
      print "Usage: python eval_tracking.py lsvm"
      print "--OR--"
      print "Usage: python eval_tracking.py regionlets"
      sys.exit(1);

    det_method = sys.argv[1]

    mail = mailpy.Mail("")
    # evaluate results and send notification email to user
    success = evaluate(det_method,mail)
    mail.finalize(success,"tracking",det_method,"")


