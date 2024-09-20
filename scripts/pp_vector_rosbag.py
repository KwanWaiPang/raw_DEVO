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
import math

import sys
sys.path.append('/home/gwp/raw_DEVO')

from utils.bag_utils import read_H_W_from_bag, read_tss_us_from_rosbag, read_images_from_rosbag, read_evs_from_rosbag, read_calib_from_bag, read_t0us_evs_from_rosbag, read_poses_from_rosbag, read_imu_from_rosbag, read_tss_ns_from_rosbag, read_rgb_images_from_rosbag, read_and_saved_evs_from_rosbag, read_evs_from_rosbag_witht0

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

OUTLIER = {
    # "vicon_aggressive_hdr": 7,  # from img_idx=7
}

def process_dirs(indirs, DELTA_MS=None):
    for indir in indirs: 
        seq = indir.split("/")[-1] #获取序列的名字，以“/”划分，取最后一个
        print(f"\n\n Vector: Undistorting {seq} evs & rgb & IMU & GT")#处理某个序列的数据

        inbag = os.path.join(indir, f"../{seq}.synced.merged.bag")#获取bag文件的路径
        bag = rosbag.Bag(inbag, "r")#读取bag文件
        # topics = list(bag.get_type_and_topic_info()[1].keys())#获取所有的topic

        imagetopic='/camera/left/image_mono'

        #创建一个文件夹，用于存放去除失真后的图片
        imgdirout = os.path.join(indir, f"images_undistorted_left")
        H, W = read_H_W_from_bag(bag, imagetopic)#获取图片的高和宽

        if not os.path.exists(imgdirout):#如果文件夹不存在，则创建文件夹
            os.makedirs(imgdirout)
        else:#如果文件夹存在，则报错（每次都会重新生成的！）
            raise NotImplementedError
        
        #读取图片
        img_start_idx = 0
        has_outlier = False
        for outlier in OUTLIER.keys():
            if outlier in indir:
                img_start_idx = OUTLIER[outlier]
                has_outlier = True
                break

        imgs = read_images_from_rosbag(bag, imagetopic, H=H, W=W)
        imgs = imgs[img_start_idx:]

        # creating rectify map（进行去除失真）
        intrinsics = [886.191073, 886.591633, 610.578911, 514.59271, 
                    -0.315760, 0.104955, 0.000320,  -0.000156]
        fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics
        Kdist =  np.zeros((3,3))   
        Kdist[0,0] = fx
        Kdist[0,2] = cx
        Kdist[1,1] = fy
        Kdist[1,2] = cy
        Kdist[2, 2] = 1
        dist_coeffs = np.asarray([k1, k2, p1, p2])

        K_new, roi = cv2.getOptimalNewCameraMatrix(Kdist, dist_coeffs, (W, H), alpha=0, newImgSize=(W, H))

        #将去除失真后的内参保存到文件中（注意对于davis346，imgae和event的内参应该是一致的）
        f = open(os.path.join(indir, f"calib_undist_left.txt"), 'w')
        f.write(f"{K_new[0,0]} {K_new[1,1]} {K_new[0,2]} {K_new[1,2]}")
        f.close()

        # undistorting images（将去除失真后的图片保存）
        img_mapx, img_mapy = cv2.initUndistortRectifyMap(Kdist, dist_coeffs, np.eye(3), K_new, (W, H), cv2.CV_32FC1)  

        # undistorting images(将去除失真后的图片保存到文件夹中)
        pbar = tqdm.tqdm(total=len(imgs)-1)
        for i, img in enumerate(imgs):
            # cv2.imwrite(os.path.join(imgdirout, f"{i:012d}_DIST.png"), img)
            img = cv2.remap(img, img_mapx, img_mapy, cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(imgdirout, f"{i:012d}.png"), img)#将去除失真后的图片保存到文件夹中
            pbar.update(1)

        tss_imgs_us = read_tss_us_from_rosbag(bag, imagetopic)#获取图片的时间戳
        tss_imgs_us = tss_imgs_us[img_start_idx:]
        assert len(tss_imgs_us) == len(imgs)#确保图片与时间戳的数量一致
        assert sorted(tss_imgs_us) == tss_imgs_us#确保时间戳是有序的


        ts_imgs_ns = read_tss_ns_from_rosbag(bag, imagetopic)#获取图片的时间戳(纳秒为单位)
        ts_imgs_ns = ts_imgs_ns[img_start_idx:]

        # ts_imgs_ns = [t * 1e3 for t in tss_imgs_us]  # 将微秒转换为纳秒(此处可能会导致存在科学计数法，待解决！)
        assert len(ts_imgs_ns) == len(imgs)
        # 保存原始的图片的时间（纳秒级别）
        f = open(os.path.join(indir, f"raw_tss_imgs_ns_left.txt"), 'w')#注意这里保存的时间单位是ns并且是原始的时间
        for t in ts_imgs_ns:
            f.write(f"{t}\n")
        f.close()

        # # 获取GT pose（注意时间以微妙为单位！）
        # # writing pose to file(获取真值pose)
        # posetopic='/gt/pose'
        # T_marker_cam0 = np.eye(4)
        # T_cam0_cam1 = np.eye(4)
        # poses, tss_gt_us = read_poses_from_rosbag(bag, posetopic, T_marker_cam0, T_cam0_cam1=T_cam0_cam1)
        # assert sorted(tss_gt_us) == tss_gt_us
        # write_gt_stamped(poses, tss_gt_us, os.path.join(indir, f"raw_gt_stamped_us.txt"))#保存真值pose（注意此时还是微妙为单位）

        #读取events数据的起始时间
        loweventtopic='/davis/left/events'
        if has_outlier:
            t0_evs = read_t0us_evs_from_rosbag(bag, loweventtopic, t0us_start=tss_imgs_us[0])#获取events的起始时间
        else:
            t0_evs = read_t0us_evs_from_rosbag(bag, loweventtopic)

        # 选择最小的时间戳作为起始时间
        # t0_us = np.minimum(np.minimum(tss_gt_us[0], tss_imgs_us[0]), t0_evs)
        t0_us = np.minimum(tss_imgs_us[0], t0_evs)
        f = open(os.path.join(indir, f"t0_us.txt"), 'w')
        f.write(f"{t0_us}\n")#将起始时间保存到文件中
        f.close()

        # 保存图片的相对时间（微秒）
        tss_imgs_us = [t - t0_us for t in tss_imgs_us]#减去起始时间，获得的就是相对时间
        f = open(os.path.join(indir, f"tss_imgs_us_left.txt"), 'w')#注意这里保存的时间单位是us
        for t in tss_imgs_us:
            f.write(f"{t:.012f}\n")
        f.close()

        # tss_gt_us = [t - t0_us for t in tss_gt_us]#减去起始时间，获得的就是相对时间
        # write_gt_stamped(poses, tss_gt_us, os.path.join(indir, f"gt_stamped_us.txt"))#保存真值pose(此处跟上面不一样是相对时间)

        # TODO: write events (and also substract t0_evs)
        # 清空imgs，释放内存
        del imgs
        #清空poses，释放内存
        # del poses        
        #保存events数据（davis346）
        h5outfile_davis346 = os.path.join(indir, f"evs_left.h5")#保存events数据的路径
        # evs = read_evs_from_rosbag_witht0(bag, loweventtopic, t0_us=t0_us, H=H, W=W)
        # write_evs_arr_to_h5(evs, h5outfile_davis346)
        # del evs  # 清空evs，释放内存
        read_and_saved_evs_from_rosbag(bag, loweventtopic, H=H, W=W, t0=t0_us,h5outfile=h5outfile_davis346)#读取events数据并保存到h5文件中

        distcoeffs=dist_coeffs#获取失真参数（跟相机一样的~）
        
        rectify_map, K_new_evs = compute_rmap_vector(Kdist, distcoeffs, indir, "left", H=H, W=W)
        assert np.all(abs(K_new_evs - K_new)<1e-5) #由于是同一个相机，所以内参应该是一样的


        #保存IMU数据
        imu1topic='/imu/data'
        all_imu1=read_imu_from_rosbag(bag, imu1topic)
        write_imu(all_imu1,os.path.join(indir, f"imu_data.csv"))

        print(f"Finshied processing {seq}\n\n")#处理完一个序列的数据
  

if __name__ == "__main__":
    # python scripts/pp_davis240c.py --indir=/media/lfl-data2/davis240c/
    parser = argparse.ArgumentParser(description="PP Vector data in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    record_file = os.path.join(args.indir, "record_processed_vector.txt")
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            has_processed_dirs = f.readlines()
            has_processed_dirs = [d.strip() for d in has_processed_dirs if d.strip() != '']
    else:
        has_processed_dirs = []

    roots = []
    for root, dirs, files in os.walk(args.indir):
        for f in files:
            try:
                if f.endswith(".bag"):#如果是rosbag文件
                # if f=="corner_slow1.synced.merged.bag": #debug used
                    p = os.path.join(root, f"{f.split('.')[0]}")
                    if p in has_processed_dirs:
                        continue
                    #如果存在，先删除
                    if os.path.exists(p):
                        os.system(f"rm -rf {p}")
                    os.makedirs(p, exist_ok=True)#创建文件夹（对于每个都创建一个文件夹）
                    # if p not in roots:
                    #     roots.append(p)#将文件夹的路径加入到roots中
                    process_dirs([p])

                    has_processed_dirs.append(p)
                    with open(record_file, "a") as f:
                        f.write(f"{p}\n")
            
            except:
                print(f"\033[31m Error processing {f} \033[0m")
                continue

    
    # cors = 4 #3
    # assert cors <= 9
    # roots_split = np.array_split(roots, cors)

    # # 进行多线程处理，每个线程处理几个文件夹
    # processes = []
    # for i in range(cors):
    #     p = multiprocessing.Process(target=process_dirs, args=(roots_split[i].tolist(), ))
    #     p.start()
    #     processes.append(p)
        
    # for p in processes:
    #     p.join()

    print(f"Finished processing all Vector scenes")
