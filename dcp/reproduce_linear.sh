MPATH="./Checkpoints/dcp_v1/models/model.best.t7"
python eval_linear.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="identity"
python eval_linear.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="identity"

MPATH="./Checkpoints/mah_dcp_v1/models/model.best.t7"
python eval_linear.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="identity"
python eval_linear.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="identity"



MPATH="./Checkpoints/dcp_v2/models/model.best.t7"
python eval_linear.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="transformer"
python eval_linear.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="transformer"

MPATH="./Checkpoints/mah_dcp_v2/models/model.best.t7"
python eval_linear.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="transformer"
python eval_linear.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="transformer"