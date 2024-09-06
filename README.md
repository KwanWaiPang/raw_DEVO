#测试Stereo-HKU数据集
~~~
conda activate raw_devo
python evals/eval_evs/eval_hku_evs.py --datapath=/home/gwp/DEVO/datasets/HKU_dataset --weights="/home/gwp/DEVO/DEVO.pth" --stride=1 --trials=1 --expname=gwphku
~~~