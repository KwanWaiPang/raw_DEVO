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

~~~

# 处理stereo-HKU数据集
~~~

conda activate raw_devo

python scripts/pp_stereo_hku.py --indir=/media/lfl-data2/Steroe_HKU/
~~~

# 处理Vector数据集
～～～
conda activate raw_devo

python scripts/pp_vector_rosbag.py --indir=/media/lfl-data2/VECtor/
～～～

# 处理FPV数据集
～～～
conda activate raw_devo

python scripts/pp_fpv.py --indir=/media/lfl-data2/UZH-FPV/

CUDA_VISIBLE_DEVICES=0 python evals/eval_evs/eval_fpv_evs.py \
--datapath=/media/lfl-data2/UZH-FPV/ \
--weights=/media/lfl-data2/DEVO_base_2gpu_ckp/240000.pth \
--stride=1 \
--trials=1

～～～

# 处理MVSEC数据集
～～～
conda activate raw_devo

python scripts/pp_mvsec_rosbag.py --indir=/media/lfl-data2/MVSEC/
～～～

# 处理DSEC数据集
～～～
conda activate raw_devo

python scripts/pp_dsec.py --indir=/media/lfl-data2/DSEC/
～～～

# 处理ECMD数据集
～～～
conda activate raw_devo

python scripts/pp_ecmd.py --indir=/media/lfl-data2/ECMD/
～～～