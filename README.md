
<div align="center">
<h1>测试 （3DV 2024）Deep Event Visual Odometry</h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://github.com/tum-vision/DEVO">Original Github Page</a>
  </h3>

</div>





# 测试Stereo-HKU数据集
~~~
conda activate raw_devo

python scripts/pp_davis240c.py --indir=/media/lfl-data2/davis240c/

python evals/eval_evs/eval_hku_evs.py --datapath=/home/gwp/DEVO/datasets/HKU_dataset --weights="/home/gwp/DEVO/DEVO.pth" --stride=1 --trials=1 --expname=gwphku
~~~

# 测试davis240c数据集
~~~
conda activate raw_devo
python evals/eval_evs/eval_davis240c_evs.py --datapath=/media/lfl-data2/davis240c/ --weights="/home/gwp/DEVO/DEVO.pth" --stride=1 --trials=1 --expname=davis240c

CUDA_VISIBLE_DEVICES=0 python evals/eval_evs/eval_davis240c_evs.py \
--datapath=/media/lfl-data2/davis240c/ \
--weights=/home/gwp/DEVO/DEVO.pth \
--stride=1 \
--trials=1


CUDA_VISIBLE_DEVICES=0 python evals/eval_evs/eval_davis240c_evs.py \
--datapath=/media/lfl-data2/davis240c/ \
--weights=/media/lfl-data2/DEVO_base_2gpu_ckp/240000.pth \
--stride=1 \
--trials=1

~~~

# 测试Mono-HKU数据集
~~~
conda activate raw_devo

python scripts/pp_mono_hku.py --indir=/media/lfl-data2/Mono_HKU/

python evals/eval_evs/eval_monohku_evs.py --datapath=/media/lfl-data2/Mono_HKU/ --weights="/home/gwp/DEVO/DEVO.pth" --stride=1 --trials=1 --expname=mono_hku


python evals/eval_evs/eval_monohku_evs.py --datapath=/media/lfl-data2/flying_sequence/ --val_split=splits/monohku/monohku_val_flying.txt --weights="/home/gwp/DEVO/DEVO.pth" --stride=1 --trials=1 --expname=mono_hku


~~~



# 处理stereo-HKU数据集
~~~

conda activate raw_devo

CUDA_VISIBLE_DEVICES=2 python evals/eval_evs/eval_hku_evs.py --datapath=/media/lfl-data2/Stereo_HKU/ --weights=/home/gwp/DEVO/DEVO.pth --val_split=splits/hku/hku_val.txt --trials=5


<!-- python scripts/pp_stereo_hku.py --indir=/media/lfl-data2/Steroe_HKU/ -->
~~~

# 处理Vector数据集
～～～
conda activate raw_devo

<!-- python scripts/pp_vector_rosbag.py --indir=/media/lfl-data2/VECtor/ -->
～～～

# 处理FPV数据集
～～～
conda activate raw_devo

<!-- python scripts/pp_fpv.py --indir=/media/lfl-data2/UZH-FPV/ -->

CUDA_VISIBLE_DEVICES=0 python evals/eval_evs/eval_fpv_evs.py \
--datapath=/media/lfl-data2/UZH-FPV/ \
--weights=/media/lfl-data2/DEVO_base_2gpu_ckp/240000.pth \
--stride=1 \
--trials=1


CUDA_VISIBLE_DEVICES=3 python evals/eval_evs/eval_fpv_evs.py --datapath=/media/lfl-data2/UZH-FPV/ --weights=/home/gwp/DEVO/DEVO.pth --val_split=splits/fpv/fpv_val_debug.txt --trials=5

～～～

# 处理MVSEC数据集
～～～
conda activate raw_devo

<!-- python scripts/pp_mvsec_rosbag.py --indir=/media/lfl-data2/MVSEC/ -->
～～～

# 处理DSEC数据集
～～～
conda activate raw_devo

<!-- python scripts/pp_dsec.py --indir=/media/lfl-data2/DSEC/ -->
～～～

# 处理ECMD数据集
～～～
conda activate raw_devo

<!-- python scripts/pp_ecmd.py --indir=/media/lfl-data2/ECMD/ -->
～～～

# 测试vector数据集
～～～
conda activate raw_devo

CUDA_VISIBLE_DEVICES=2 python evals/eval_evs/eval_vector_evs.py --datapath=/media/lfl-data2/VECtor_h5/ --weights=/home/gwp/DEVO/DEVO.pth --val_split=splits/vector/vector_val.txt --trials=1
～～～


# 测试TUM-VIE数据集

```bash
conda activate raw_devo

CUDA_VISIBLE_DEVICES=3 python evals/eval_evs/eval_tumvie_evs.py --datapath=/media/lfl-data2/TUM-VIE/ --weights=/home/gwp/DEVO/DEVO.pth --val_split=splits/tumvie/tumvie_val_small.txt --trials=5
```