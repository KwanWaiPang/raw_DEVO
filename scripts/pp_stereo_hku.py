import numpy as np
import os
import argparse
import cv2
import tqdm
import glob
import multiprocessing

import h5py
import sys
sys.path.append('/home/gwp/raw_DEVO')

# 处理服务器中evo的可视化问题
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'
from evo.tools import plot

from utils.load_utils import compute_rmap_vector
from utils.event_utils import write_evs_arr_to_h5
from utils.viz_utils import render
import rosbag
from utils.bag_utils import read_H_W_from_bag, read_tss_us_from_rosbag, read_images_from_rosbag, read_evs_from_rosbag, read_calib_from_bag, read_t0us_evs_from_rosbag, read_poses_from_rosbag,read_tss_ns_from_rosbag,read_imu_from_rosbag

H, W = 260, 346

def write_imu(imu, outfile):
    with open(outfile, 'w') as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for pose in imu:
            # f.write(f"{pose} ")
            #将 pose 列表中的每个元素转换为字符串并以逗号连接成一个字符串，从而避免输出带有方括号的列表形式。
            f.write(",".join(map(str, pose)))
            f.write("\n")

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

def get_calib_hku(side): # hku uses davis
    if side == "left":
        intr = [249.69341447817564, 248.41625664694038, 176.74240257052816, 129.47631010746218]
        distcoeffs = np.array([-0.3794794654640921, 0.15393049046270296, 0.0011400586965363895, -0.0019042695753031854])
        Kdist = np.eye(3)
        Kdist[0,0] = intr[0]
        Kdist[1,1] = intr[1]
        Kdist[0,2] = intr[2]
        Kdist[1,2] = intr[3]

    elif side == "right":
        intr = [258.61441518089174, 258.00363445501824, 178.44356547141308, 135.84792628403616]
        distcoeffs = np.array([-0.3864639588089853, 0.1707517912637013, -0.00046695742172563157, 0.0006610867041757214])

        Kdist = np.eye(3)
        Kdist[0,0] = intr[0]
        Kdist[1,1] = intr[1]
        Kdist[0,2] = intr[2]
        Kdist[1,2] = intr[3]

    return Kdist, distcoeffs

def process_seq_hku(indirs, side="left", DELTA_MS=None):
    for indir in indirs: 
        seq = indir.split("/")[-1]
        print(f"\n\n HKU: Undistorting {seq} evs & rgb & IMU & GT")

        inbag = os.path.join(indir, f"../{seq}.bag")
        bag = rosbag.Bag(inbag, "r")
        topics = list(bag.get_type_and_topic_info()[1].keys())
        if side == "left":
            imgtopic_idx = 2
            evtopic_idx = 1
        elif side == "right":
            imgtopic_idx = 5
            evtopic_idx = 4
        else:
            raise NotImplementedError

        #创建一个文件夹，用于存放去除失真后的图片
        imgdirout = os.path.join(indir, f"images_undistorted_{side}")
        Hbag, Wbag = read_H_W_from_bag(bag, topics[imgtopic_idx])#获取图片的高和宽
        assert (Hbag == H and Wbag == W)

        if not os.path.exists(imgdirout):
            os.makedirs(imgdirout)
        else:
            raise NotImplementedError #如果文件夹存在，则报错（每次都会重新生成的！）
            # img_list_undist = [os.path.join(indir, imgdirout, im) for im in sorted(os.listdir(imgdirout)) if im.endswith(".png")]
            # if bag.get_message_count(topics[imgtopic_idx]) == len(img_list_undist):
            #     print(f"\n\nWARNING **** Images already undistorted. Skipping {indir} ***** \n\n")
            #     assert os.path.isfile(os.path.join(indir, f"rectify_map_{side}.h5"))
            #     continue

        imgs = read_images_from_rosbag(bag, topics[imgtopic_idx], H=H, W=W)
        imgs = [cv2.resize(img, (W, H)) for img in imgs] #每张图片都resize到260*346
        Kdist, distcoeffs = get_calib_hku(side)

        # undistorting images
        K_new, roi = cv2.getOptimalNewCameraMatrix(Kdist, distcoeffs, (W, H), alpha=0, newImgSize=(W, H))
        f = open(os.path.join(indir, f"calib_undist_{side}.txt"), 'w')
        f.write(f"{K_new[0,0]} {K_new[1,1]} {K_new[0,2]} {K_new[1,2]}")
        f.close()

        img_mapx, img_mapy = cv2.initUndistortRectifyMap(Kdist, distcoeffs, np.eye(3), K_new, (W, H), cv2.CV_32FC1) 
        # undistorting images
        pbar = tqdm.tqdm(total=len(imgs)-1)
        for i, img in enumerate(imgs):
            # cv2.imwrite(os.path.join(imgdirout, f"{i:012d}_DIST.png"), img) # DEBUG only
            img = cv2.remap(img, img_mapx, img_mapy, cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(imgdirout, f"{i:012d}.png"), img)
            pbar.update(1)
        # imgs = [] # free memory

        # writing pose to file(写真实位姿到文件中)
        posetopic ='/cpy_uav/viconros/odometry'
        if side == "left":
            T_cam0_cam1 = np.eye(4)
        else:
            T_cam0_cam1 =  np.array([[0.9999189999842378, 0.00927392731970859, -0.00871709484799569, -0.05968052204060377],
                                     [-0.009231577824269699, 0.9999454511978819, 0.004885959428529005, -0.0005334476469976882],
                                     [0.008761931373541011, -0.004805091126247473, 0.9999500685823629, 0.0005990728587972945],
                                     [0.0, 0.0, 0.0, 1.0]])
            
        # TODO: check inv or not?
        T_marker_cam0 = np.linalg.inv(
                        np.array([[0.9999552277012158, -0.00603191153357543, 0.007290996931816412, 0.00011018857347815285],
                                 [0.005994670026470383, 0.9999689294906282, 0.005118982773930891, -0.0007730487905611042], 
                                 [-0.007321647648062164, -0.005075046464534421, 0.9999603179022153, -0.060160984076249716],
                                 [0.0, 0.0, 0.0, 1.0]])) 

        tss_imgs_us = read_tss_us_from_rosbag(bag, topics[imgtopic_idx])
        assert len(tss_imgs_us) == len(imgs)
        ts_imgs_ns = read_tss_ns_from_rosbag(bag, topics[imgtopic_idx])#获取图片的时间戳(纳秒为单位)
        assert len(ts_imgs_ns) == len(imgs)
        imgs = [] # free memory
        # 保存原始的图片的时间（纳秒级别）
        f = open(os.path.join(indir, f"raw_tss_imgs_ns_{side}.txt"), 'w')#注意这里保存的时间单位是ns并且是原始的时间
        for t in ts_imgs_ns:
            f.write(f"{t}\n")
        f.close()

        poses, tss_gt_us = read_poses_from_rosbag(bag, posetopic, T_marker_cam0, T_cam0_cam1=T_cam0_cam1)
        write_gt_stamped(poses, tss_gt_us, os.path.join(indir, f"raw_gt_stamped_us.txt"))#保存真值pose（注意此时还是微妙为单位）

        # 读取第一个事件的时间戳
        t0_evs = read_t0us_evs_from_rosbag(bag, topics[evtopic_idx])
        assert sorted(tss_imgs_us) == tss_imgs_us
        assert sorted(tss_gt_us) == tss_gt_us

        t0_us = np.minimum(np.minimum(tss_gt_us[0], tss_imgs_us[0]), t0_evs)
        f = open(os.path.join(indir, f"t0_us.txt"), 'w')
        f.write(f"{t0_us}\n")#将起始时间保存到文件中
        f.close()

        # 保存图像的相对时间（微妙为单位）
        tss_imgs_us = [t - t0_us for t in tss_imgs_us]
        # saving tss
        f = open(os.path.join(indir, f"tss_imgs_us_{side}.txt"), 'w')
        for t in tss_imgs_us:
            f.write(f"{t:.012f}\n")
        f.close()

        tss_gt_us = [t - t0_us for t in tss_gt_us]
        write_gt_stamped(poses, tss_gt_us, os.path.join(indir, f"gt_stamped_{side}.txt"))

        # write events (and also substract t0_evs)
        evs = read_evs_from_rosbag(bag, topics[evtopic_idx], H=H, W=W)
        for ev in evs:
            ev[2] -= t0_us
        h5outfile = os.path.join(indir, f"evs_{side}.h5")
        write_evs_arr_to_h5(evs, h5outfile)

        rectify_map, K_new_evs = compute_rmap_vector(Kdist, distcoeffs, indir, side, H=H, W=W)
        assert np.all(abs(K_new_evs - K_new)<1e-5) 

        # ######## [DEBUG] viz undistorted events
        # outvizfolder = os.path.join(indir, f"evs_{side}_undist")
        # os.makedirs(outvizfolder, exist_ok=True)
        # pbar = tqdm.tqdm(total=len(tss_imgs_us)-1)
        # for (ts_idx, ts_us) in enumerate(tss_imgs_us):
        #     if ts_idx == len(tss_imgs_us) - 1:
        #         break
            
        #     if DELTA_MS is None:
        #         evs_idx = np.where((evs[:, 2] >= ts_us) & (evs[:, 2] < tss_imgs_us[ts_idx+1]))[0]
        #     else:
        #         evs_idx = np.where((evs[:, 2] >= ts_us) & (evs[:, 2] < ts_us + DELTA_MS*1e3))[0]
                
        #     if len(evs_idx) == 0:
        #         print(f"no events in range {ts_us*1e-3} - {tss_imgs_us[ts_idx+1]*1e-3} milisecs")
        #         continue
        #     evs_batch = np.array(evs[evs_idx, :]).copy()


        #     img = render(evs_batch[:, 0], evs_batch[:, 1], evs_batch[:, 3], H, W)
        #     imfnmae = os.path.join(outvizfolder, f"{ts_idx:06d}_dist.png")
        #     cv2.imwrite(imfnmae, img)

        #     rect = rectify_map[evs_batch[:, 1].astype(np.int32), evs_batch[:, 0].astype(np.int32)]
        #     img = render(rect[:, 0], rect[:, 1], evs_batch[:, 3], H, W)
            
        #     imfnmae = imfnmae.split(".")[0] + ".png"
        #     cv2.imwrite(os.path.join(outvizfolder, imfnmae), img)

        #     pbar.update(1)
        # ############ [end DEBUG] viz undistorted events

        #保存IMU数据
        imu1topic='/davis_left/imu'
        all_imu1=read_imu_from_rosbag(bag, imu1topic)
        write_imu(all_imu1,os.path.join(indir, f"{side}_imu_data.csv"))

        print(f"Finshied processing {indir}\n\n")
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP Stereo HKU data in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    roots = []
    for root, dirs, files in os.walk(args.indir):
        for f in files:
            if f.endswith(".bag"):
            # if f=="HKU_aggressive_rotation.bag": #debug used
                p = os.path.join(root, f"{f.split('.')[0]}")
                #如果存在，先删除
                if os.path.exists(p):
                    os.system(f"rm -rf {p}")
                os.makedirs(p, exist_ok=True)#创建文件夹（对于每个都创建一个文件夹）

                if p not in roots:
                    roots.append(p)

    
    cors = 2 #4
    assert cors <= 9
    roots_split = np.array_split(roots, cors)

    processes = []
    for i in range(cors):
        p = multiprocessing.Process(target=process_seq_hku, args=(roots_split[i].tolist(),))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    print(f"Finished processing all Stereo HKU scenes")


