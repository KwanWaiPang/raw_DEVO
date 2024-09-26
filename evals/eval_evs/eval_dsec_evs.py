import os
import torch
from devo.config import cfg

# 处理服务器中evo的可视化问题
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

import sys
sys.path.append('/home/gwp/raw_DEVO')#要导入

from utils.load_utils import load_gt_us, dsec_evs_iterator
from utils.eval_utils import assert_eval_config, run_voxel
from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.viz_utils import viz_flow_inference

H, W = 480, 640

@torch.no_grad()  #表示该函数不会计算梯度（由于是导入网络权重的）
def evaluate(config, args, net, train_step=None, datapath="", split_file=None, 
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, timing=False, side='left', viz_flow=False, sota_comparison=False):
    dataset_name = "dsec_evs"
    assert side == "davis346" or side == "dvxplorer"

    if side == "dvxplorer":
        H, W = 480, 640
    else:
        H, W = 260, 346

    # 若配置文件为空，则使用默认配置文件
    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")
    
    # 读取场景的名称    
    scenes = open(split_file).read().split()
    print("the number of scenes is", len(scenes),"the input scenes are: ", scenes, "the type of camera is: ", side)

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        for trial in range(trials):
            # estimated trajectory
            datapath_val = os.path.join(datapath, scene)

            # run the slam system
            # 通过调用davis240c_evs_iterator来将事件数据进行打包处理
            traj_est, tstamps, flowdata = run_voxel(datapath_val, config, net, viz=viz, 
                                          iterator=dsec_evs_iterator(datapath_val, side=side, stride=stride, timing=timing, H=H, W=W),
                                          timing=timing, H=H, W=W, viz_flow=viz_flow)

            # load  traj（这应该是获取gt trajectory的值,从txt文件中读取,注意读入的为秒）
            tss_traj_s, traj_hf = load_gt_us(os.path.join(datapath_val, f"trajectory/stamped_groundtruth2.txt"))

            import numpy as np
            t0_us = np.loadtxt(os.path.join(datapath_val, f"t0_us.txt"), delimiter=',', usecols=0)
            tss_traj_us = tss_traj_s*1e6 - t0_us#减去t0_us的时间

            # do evaluation （进行验证）
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            # 通过log_results函数来记录结果(用evo评估定位的精度)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride,
                                                                   max_diff_sec=1.0,
                                                                   _n_to_align=-1,
                                                                   expname=f"{scene}_{side}"#args.expname
                                                                   )
            
            if sota_comparison: #如果需要与sota进行比较
                tss_gt_s, traj_gt = load_gt_us(os.path.join(datapath_val, f"trajectory/stamped_groundtruth_alignment.txt"))#用对齐后的轨迹
                tss_esvoaa_s, traj_esvoaa = load_gt_us(os.path.join(datapath_val, f"trajectory/stamped_traj_estimate_ours.txt"))#ESVOAA的轨迹
                tss_esvo_s, traj_esvo = load_gt_us(os.path.join(datapath_val, f"trajectory/stamped_traj_estimate_ori.txt"))#ESVO的轨迹
                tss_gt_us = tss_gt_s*1e6 - t0_us#减去t0_us的时间
                tss_esvoaa_us = tss_esvoaa_s*1e6 - t0_us#减去t0_us的时间
                tss_esvo_us = tss_esvo_s*1e6 - t0_us#减去t0_us的时间
                from utils.eval_utils import plot_four_trajectory
                pdfname = f"results/{dataset_name}/{scene}.pdf"
                plot_four_trajectory((traj_est, tstamps/1e6), (traj_esvoaa, tss_esvoaa_us/1e6), (traj_esvo, tss_esvo_us/1e6), (traj_gt, tss_gt_us/1e6), 
                            f"Comparison DEVO with ESVO and ESVO_AA in {scene}",
                            pdfname, align=True, correct_scale=True, max_diff_sec=1.0)
            
            if viz_flow:
                viz_flow_inference(outfolder, flowdata)
            
        print(scene, sorted(results_dict_scene[scene]))

    # write output to file with timestamp
    write_raw_results(all_results, outfolder)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__': 
    import argparse
    # 导入一系列参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/eval_dsec.yaml")#参数文件
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--sota_comparison', action="store_true") #是否与sota进行比较
    parser.add_argument('--val_split', type=str, default="splits/monohku/monohku_val.txt") # 验证集的路径,有它来决定验证的序列
    parser.add_argument('--trials', type=int, default=5)# 试验次数
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--side', type=str, default="dvxplorer")
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="")

     # 解析命令行参数并将结果赋值给args。
    args = parser.parse_args()
    assert_eval_config(args)# 检查几个配置的参数是否合理

    # args.config就是VO的配置文件，通过merge_from_file函数将配置文件中的内容合并到程序的配置对象cfg中（cfg 是一个配置对象，用来存储程序运行时所需的各种配置参数）
    cfg.merge_from_file(args.config)
    print("Running eval_dsec_evs.py with config...")
    print(cfg) 

    torch.manual_seed(1234)# 设置随机种子

    # 人为设置一些参数
    args.save_trajectory = True
    args.plot = True #人为设置为True
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz,timing=args.timing, \
                        stride=args.stride, side=args.side, viz_flow=args.viz_flow, sota_comparison=args.sota_comparison)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
