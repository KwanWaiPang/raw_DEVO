import numpy as np
import os
import argparse
import cv2
import tqdm
import glob
import multiprocessing

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tqdm as tqdm
import h5py

import sys
sys.path.append('/home/gwp/raw_DEVO')

from utils.bag_utils import read_H_W_from_bag, read_tss_us_from_rosbag, read_images_from_rosbag, read_evs_from_rosbag, read_calib_from_bag, read_t0us_evs_from_rosbag, read_poses_from_rosbag, read_imu_from_rosbag, read_tss_ns_from_rosbag

# 处理服务器中evo的可视化问题
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'
from evo.tools import plot

from utils.event_utils import write_evs_arr_to_h5
from utils.load_utils import compute_rmap_vector
from utils.viz_utils import render

def write_gt_stamped(poses, tss_us_gt, outfile):
    with open(outfile, 'w') as f:
        for pose, ts in zip(poses, tss_us_gt):
            f.write(f"{ts} ")
            for i, p in enumerate(pose):
                if i < len(pose) - 1:
                    f.write(f"{p} ")
                else:
                    f.write(f"{p}")
            f.write("\n")

def write_imu(imu, outfile):
    with open(outfile, 'w') as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for pose in imu:
            # f.write(f"{pose} ")
            #将 pose 列表中的每个元素转换为字符串并以逗号连接成一个字符串，从而避免输出带有方括号的列表形式。
            f.write(",".join(map(str, pose)))
            f.write("\n")


def process_dirs(indirs, side="left", DELTA_MS=None):
    for indir in indirs: 
        seq = indir.split("/")[-1] #获取序列的名字，以“/”划分，取最后一个
        print(f"\n\n davis240c: Undistorting {seq} evs & rgb & IMU & GT")#处理某个序列的数据

        inbag = os.path.join(indir, f"../{seq}.bag")#获取bag文件的路径
        bag = rosbag.Bag(inbag, "r")#读取bag文件
        topics = list(bag.get_type_and_topic_info()[1].keys())#获取所有的topic
        topics = sorted([t for t in topics if "events" in t or "image" in t])#将所有的topic按照events和image进行排序
        assert topics == sorted(['/dvs/events', '/dvs/image_raw']) 
        if side == "left":
            imgtopic_idx = 1
            evtopic_idx = 0
        else:
            raise NotImplementedError

        imgdirout = os.path.join(indir, f"images_undistorted_{side}")#创建一个文件夹，用于存放处理后的图片
        H, W = read_H_W_from_bag(bag, topics[imgtopic_idx])#获取图片的高和宽
        assert (H == 180 and W == 240) #检查图片的高和宽是否符合要求

        if not os.path.exists(imgdirout):#如果文件夹不存在，则创建文件夹
            os.makedirs(imgdirout)
        else:#如果文件夹存在，则检查是否已经处理过
            img_list_undist = [os.path.join(indir, imgdirout, im) for im in sorted(os.listdir(imgdirout)) if im.endswith(".png")]
            if bag.get_message_count(topics[1]) == len(img_list_undist):
                print(f"\n\nWARNING **** Images already undistorted. Skipping {indir} ***** \n\n")
                assert os.path.isfile(os.path.join(indir, f"rectify_map_{side}.h5")) or seq == "simulation_3planes"
                # continue
        #读取图片
        imgs = read_images_from_rosbag(bag, topics[imgtopic_idx], H=H, W=W)
        #要resize图片
    
        # creating rectify map（进行去除失真）
        if side == "left":
            intrinsics = [199.092366542, 198.82882047, 132.192071378, 110.712660011, 
                        -0.368436311798,  0.150947243557,  -0.000296130534385,  -0.000759431726241]
        else:
            raise NotImplementedError #如果是右边的相机，则抛出异常
        fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics
        Kdist =  np.zeros((3,3))   
        Kdist[0,0] = fx
        Kdist[0,2] = cx
        Kdist[1,1] = fy
        Kdist[1,2] = cy
        Kdist[2, 2] = 1
        dist_coeffs = np.asarray([k1, k2, p1, p2])

        K_new, roi = cv2.getOptimalNewCameraMatrix(Kdist, dist_coeffs, (W, H), alpha=0, newImgSize=(W, H))

        f = open(os.path.join(indir, f"calib_undist_{side}.txt"), 'w')#将去除失真的参数保存到文件中
        f.write(f"{K_new[0,0]} {K_new[1,1]} {K_new[0,2]} {K_new[1,2]}")
        f.close()
        
        # coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).astype("float32") # TODO: +-1 missing??
        # term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
        # points = cv2.undistortPointsIter(coords, Kdist, dist_coeffs, np.eye(3), K_new, criteria=term_criteria)
        # rectify_map = points.reshape((H, W, 2))       

        # h5outfile = os.path.join(indir, f"rectify_map_{side}.h5")#将去除失真的结果保存到h5文件中
        # ef_out = h5py.File(h5outfile, 'w')
        # ef_out.clear()
        # ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
        # ef_out["rectify_map"][:] = rectify_map
        # ef_out.close() 

        # undistorting images
        img_mapx, img_mapy = cv2.initUndistortRectifyMap(Kdist, dist_coeffs, np.eye(3), K_new, (W, H), cv2.CV_32FC1)  

        # undistorting images
        pbar = tqdm.tqdm(total=len(imgs)-1)
        for i, img in enumerate(imgs):
            # cv2.imwrite(os.path.join(imgdirout, f"{i:012d}_DIST.png"), img)
            img = cv2.remap(img, img_mapx, img_mapy, cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(imgdirout, f"{i:012d}.png"), img)#将去除失真后的图片保存到文件夹中
            pbar.update(1)

        # writing pose to file(获取真值pose)
        posetopic = "/optitrack/davis"
        T_marker_cam0 = np.eye(4)
        if side == "left":
            T_cam0_cam1 = np.eye(4)
        else:
            raise NotImplementedError #如果是右边的相机，则抛出异常


        tss_imgs_us = read_tss_us_from_rosbag(bag, topics[imgtopic_idx])#获取图片的时间戳
        assert len(tss_imgs_us) == len(imgs)

        ts_imgs_ns = read_tss_ns_from_rosbag(bag, topics[imgtopic_idx])#获取图片的时间戳(纳秒为单位)
        # saving 原始的时间
        f = open(os.path.join(indir, f"raw_tss_imgs_ns_{side}.txt"), 'w')#注意这里保存的时间单位是ns并且是原始的时间
        for t in ts_imgs_ns:
            f.write(f"{t}\n")
        f.close()

        # 获取GT pose（注意时间以微妙为单位！）
        poses, tss_gt_us = read_poses_from_rosbag(bag, posetopic, T_marker_cam0, T_cam0_cam1=T_cam0_cam1)
        t0_evs = read_t0us_evs_from_rosbag(bag, topics[evtopic_idx])
        assert sorted(tss_imgs_us) == tss_imgs_us
        assert sorted(tss_gt_us) == tss_gt_us

        write_gt_stamped(poses, tss_gt_us, os.path.join(indir, f"raw_gt_stamped_ns_{side}.txt"))#保存真值pose（注意此时还是微妙为单位）

        # 选择最小的时间戳作为起始时间
        t0_us = np.minimum(np.minimum(tss_gt_us[0], tss_imgs_us[0]), t0_evs)
        tss_imgs_us = [t - t0_us for t in tss_imgs_us]#减去起始时间，获得的就是相对时间

        # saving tss
        f = open(os.path.join(indir, f"tss_imgs_us_{side}.txt"), 'w')#注意这里保存的时间单位是us
        for t in tss_imgs_us:
            f.write(f"{t:.012f}\n")
        f.close()

        tss_gt_us = [t - t0_us for t in tss_gt_us]#减去起始时间，获得的就是相对时间
        write_gt_stamped(poses, tss_gt_us, os.path.join(indir, f"gt_stamped_{side}.txt"))#保存真值pose

        #保存IMU数据
        imu_topic = "/dvs/imu"
        all_imu=read_imu_from_rosbag(bag, imu_topic)
        write_imu(all_imu,os.path.join(indir, f"imu_data.csv"))

        # TODO: write events (and also substract t0_evs)
        evs = read_evs_from_rosbag(bag, topics[evtopic_idx], H=H, W=W)#读取events
        # f = open(os.path.join(indir, f"evs_{side}.txt"), 'w')#将events保存到txt文件中
        # for i in range(evs.shape[0]):
        #     f.write(f"{(evs[i, 2] - t0_us):.04f} {int(evs[i, 0])} {int(evs[i, 1])} {int(evs[i, 3])}\n")
        # f.close()

        # 下面要double check，由于涉及时间戳出现一些问题，故此还是保留相对时间吧！
        # raw_timestamp_h5outfile = os.path.join(indir, f"raw_timestamp_evs_{side}.h5")
        # write_evs_arr_to_h5(evs, raw_timestamp_h5outfile)#将events保存到h5文件中

        for ev in evs:
            ev[2] -= t0_us #减去起始时间,获得的就是相对时间
        h5outfile = os.path.join(indir, f"evs_{side}.h5")#注意此文件只保留了相对时间
        write_evs_arr_to_h5(evs, h5outfile)#将events保存到h5文件中

        distcoeffs=dist_coeffs#获取失真参数
        
        rectify_map, K_new_evs = compute_rmap_vector(Kdist, distcoeffs, indir, side, H=H, W=W)
        assert np.all(abs(K_new_evs - K_new)<1e-5) 

        ######## [DEBUG] viz undistorted events
        outvizfolder = os.path.join(indir, f"evs_{side}_undist")#创建一个文件夹，用于存放处理后的图片
        os.makedirs(outvizfolder, exist_ok=True)
        pbar = tqdm.tqdm(total=len(tss_imgs_us)-1)
        for (ts_idx, ts_us) in enumerate(tss_imgs_us):
            if ts_idx == len(tss_imgs_us) - 1:
                break
            
            if DELTA_MS is None:
                evs_idx = np.where((evs[:, 2] >= ts_us) & (evs[:, 2] < tss_imgs_us[ts_idx+1]))[0]
            else:
                evs_idx = np.where((evs[:, 2] >= ts_us) & (evs[:, 2] < ts_us + DELTA_MS*1e3))[0]
                
            if len(evs_idx) == 0:
                print(f"no events in range {ts_us*1e-3} - {tss_imgs_us[ts_idx+1]*1e-3} milisecs")
                continue
            evs_batch = np.array(evs[evs_idx, :]).copy()


            img = render(evs_batch[:, 0], evs_batch[:, 1], evs_batch[:, 3], H, W)
            imfnmae = os.path.join(outvizfolder, f"{ts_idx:06d}_dist.png")
            cv2.imwrite(imfnmae, img)

            rect = rectify_map[evs_batch[:, 1].astype(np.int32), evs_batch[:, 0].astype(np.int32)]
            img = render(rect[:, 0], rect[:, 1], evs_batch[:, 3], H, W)
            
            imfnmae = imfnmae.split(".")[0] + ".png"
            cv2.imwrite(os.path.join(outvizfolder, imfnmae), img)

            pbar.update(1)
        ############ [end DEBUG] viz undistorted events

        print(f"Finshied processing {indir}\n\n")
  
    
if __name__ == "__main__":
    # python scripts/pp_davis240c.py --indir=/media/lfl-data2/davis240c/
    parser = argparse.ArgumentParser(description="PP davis240c data in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    roots = []
    for root, dirs, files in os.walk(args.indir):
        for f in files:
            if f.endswith(".bag"):#如果是rosbag文件
            # if f=="boxes_translation.bag": #debug used
                p = os.path.join(root, f"{f.split('.')[0]}")
                #如果存在，先删除
                if os.path.exists(p):
                    os.system(f"rm -rf {p}")
                os.makedirs(p, exist_ok=True)#创建文件夹（对于每个都创建一个文件夹）
                if p not in roots:
                    roots.append(p)#将文件夹的路径加入到roots中

    
    cors = 4 #3
    assert cors <= 9
    roots_split = np.array_split(roots, cors)

    # 进行多线程处理，每个线程处理几个文件夹
    processes = []
    for i in range(cors):
        p = multiprocessing.Process(target=process_dirs, args=(roots_split[i].tolist(), ))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    print(f"Finished processing all davis240c scenes")
