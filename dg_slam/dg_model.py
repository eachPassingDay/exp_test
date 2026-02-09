import torch
import numpy as np
from lietorch import SE3
from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from gs_tracking_mapping import gs_tracking_mapping
from dg_slam.pose_transform import quaternion_to_transform_noBatch

def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    from scipy.spatial.transform import Rotation

    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose

class dg_model:
    def __init__(self, cfg, args):
        super(dg_model, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis
        self.cfg = cfg

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)
        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)
        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        self.tracking_mapping = gs_tracking_mapping(self.cfg, self.args, self.video)
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        self.mapping_counter = 0
        self.inv_pose = None
        
        # [State Variables]
        self.is_high_dynamic_mode = False 
        self.frames_since_last_kf = 0 # 距离上一次关键帧过去了多少帧

    def load_weights(self, weights):
        """ load trained model weights """
        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()
        
        
    def track(self, tstamp, image, depth, pose, intrinsics, seg_mask):
        """ main thread - update map """

        # 1. 计算动态比例
        total_pixels = seg_mask.numel()
        dynamic_pixels = seg_mask.sum().item()
        current_dynamic_ratio = dynamic_pixels / total_pixels
        
        # [Config] 阈值设置
        HIGH_DYN_THRESH = 0.10  # 超过 3% 认为是高动态 (人来了)
        LOW_DYN_THRESH = 0.10   # 低于 2% 认为是低动态 (安全区间)
        FORCE_FREQ = 2          # 低动态区间内，强制每 5 帧一个关键帧

        force_keyframe = False
        
        # 2. 状态机逻辑 (State Machine)
        if current_dynamic_ratio > HIGH_DYN_THRESH:
            # === [Case A: 危险区 (高动态)] ===
            self.is_high_dynamic_mode = True
            # 在高动态下，我们完全信任 DROID 的判断，不强制干预
            
        elif current_dynamic_ratio < LOW_DYN_THRESH:
            # === [Case B: 安全区 (低动态)] ===
            
            # 策略 1: 抓拍“人刚走”的瞬间 (Disocclusion)
            if self.is_high_dynamic_mode:
                print(f">>> [Force KF] Object Left! Ratio dropped to {current_dynamic_ratio:.4f}")
                force_keyframe = True
                self.is_high_dynamic_mode = False # 退出警戒状态
            
            # 策略 2: 持续低动态下的高频采样 (High Freq Sampling)
            # 如果距离上次关键帧已经超过 Force_freq 帧，且画面依然干净，强制建图！
            elif self.frames_since_last_kf >= FORCE_FREQ:
                # print(f">>> [Force KF] Low Dynamic Sustain. Force update at interval {self.frames_since_last_kf}")
                force_keyframe = True
        
        else:
            # === [Case C: 过渡区] ===
            pass

        with torch.no_grad():
            self.filterx.track(tstamp, image, depth, pose, intrinsics, seg_mask)
            
            # ================= [核心修改] =================
            original_thresh = self.frontend.keyframe_thresh
            
            if force_keyframe:
                self.frontend.keyframe_thresh = -100.0 # 强制保留
            
            # DROID 前端决定是否生成关键帧
            update_status = self.frontend()
            
            if force_keyframe:
                self.frontend.keyframe_thresh = original_thresh
                update_status = True
            # =============================================
            
            # [Counter Update]
            # 如果这一帧最终成为了关键帧 (无论是被强制的，还是 DROID 自己想加的)
            # 我们都重置计数器
            if update_status:
                self.frames_since_last_kf = 0
            else:
                self.frames_since_last_kf += 1

        if update_status or (tstamp == self.cfg["data"]["n_img"] - 1):
            tracking_min = self.frontend.graph.ii.min().item()
            tracking_max = self.frontend.graph.ii.max().item()
            while(True):
                # ... (以下代码保持不变) ...
                if (self.mapping_counter < tracking_min) or ((self.mapping_counter < tracking_max + 1) and (tstamp == self.cfg["data"]["n_img"] - 1)):
                    idx = self.mapping_counter
                    idx = torch.tensor(idx)
                    img_idx = self.video.images[self.mapping_counter] / 255
                    
                    disps_up = self.video.disps_up[self.mapping_counter]
                    depth_idx = torch.where(disps_up > 0, 1.0/disps_up, disps_up)

                    gt_pose_idx = self.video.poses_gt[self.mapping_counter]
                    gt_pose_idx = torch.from_numpy(pose_matrix_from_quaternion(gt_pose_idx.cpu())).cuda() 

                    pose_idx = self.video.poses[self.mapping_counter] 
                    pose_idx = quaternion_to_transform_noBatch(SE3(pose_idx).inv().data) 

                    if self.inv_pose is None:
                        init_pose = gt_pose_idx
                        self.inv_pose = torch.inverse(init_pose)
                        gt_pose_idx = self.inv_pose @ gt_pose_idx
                    else:
                        gt_pose_idx = self.inv_pose @ gt_pose_idx

                    # 注意：这里是进入 Mapping 的 mask，这里做了取反操作 (~)
                    # 说明 Mapping 内部使用的是 1=静态
                    seg_mask_idx = ~ self.video.seg_masks_ori[self.mapping_counter]

                    self.tracking_mapping.run(idx, img_idx, depth_idx, gt_pose_idx, pose_idx, seg_mask_idx)
                    self.mapping_counter += 1
                else:
                    break

    def terminate_woBA(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """
        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy() # c2w