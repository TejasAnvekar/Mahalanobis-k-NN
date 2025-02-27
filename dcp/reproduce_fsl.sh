##V1
#DCP
MPATH="./Checkpoints/dcp_v1/models/model.best.t7"
python eval_fsl.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="identity" --k_way 5 --m_shot 10 --n_query 20
python eval_fsl.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="identity" --k_way 5 --m_shot 10 --n_query 20

python eval_fsl.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="identity" --k_way 5 --m_shot 20 --n_query 40
python eval_fsl.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="identity" --k_way 5 --m_shot 20 --n_query 40


python eval_fsl.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="identity" --k_way 10 --m_shot 10 --n_query 20
python eval_fsl.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="identity" --k_way 10 --m_shot 10 --n_query 20

python eval_fsl.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="identity" --k_way 10 --m_shot 20 --n_query 40
python eval_fsl.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="identity" --k_way 10 --m_shot 20 --n_query 40




#MDCP
MPATH="./Checkpoints/mah_dcp_v1/models/model.best.t7"
python eval_fsl.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="identity" --k_way 5 --m_shot 10 --n_query 20
python eval_fsl.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="identity" --k_way 5 --m_shot 10 --n_query 20

python eval_fsl.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="identity" --k_way 5 --m_shot 20 --n_query 40
python eval_fsl.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="identity" --k_way 5 --m_shot 20 --n_query 40


python eval_fsl.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="identity" --k_way 10 --m_shot 10 --n_query 20
python eval_fsl.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="identity" --k_way 10 --m_shot 10 --n_query 20

python eval_fsl.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="identity" --k_way 10 --m_shot 20 --n_query 40
python eval_fsl.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="identity" --k_way 10 --m_shot 20 --n_query 40





##V2
#DCP
MPATH="./Checkpoints/dcp_v2/models/model.best.t7"
python eval_fsl.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="transformer" --k_way 5 --m_shot 10 --n_query 20
python eval_fsl.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="transformer" --k_way 5 --m_shot 10 --n_query 20

python eval_fsl.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="transformer" --k_way 5 --m_shot 20 --n_query 40
python eval_fsl.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="transformer" --k_way 5 --m_shot 20 --n_query 40


python eval_fsl.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="transformer" --k_way 10 --m_shot 10 --n_query 20
python eval_fsl.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="transformer" --k_way 10 --m_shot 10 --n_query 20

python eval_fsl.py --dataset="modelnet40" --model="dcp" --model_path=$MPATH --pointer="transformer" --k_way 10 --m_shot 20 --n_query 40
python eval_fsl.py --dataset="scanobjectnn" --model="dcp" --model_path=$MPATH --pointer="transformer" --k_way 10 --m_shot 20 --n_query 40




#MDCP
MPATH="./Checkpoints/mah_dcp_v2/models/model.best.t7"
python eval_fsl.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="transformer" --k_way 5 --m_shot 10 --n_query 20
python eval_fsl.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="transformer" --k_way 5 --m_shot 10 --n_query 20

python eval_fsl.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="transformer" --k_way 5 --m_shot 20 --n_query 40
python eval_fsl.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="transformer" --k_way 5 --m_shot 20 --n_query 40


python eval_fsl.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="transformer" --k_way 10 --m_shot 10 --n_query 20
python eval_fsl.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="transformer" --k_way 10 --m_shot 10 --n_query 20

python eval_fsl.py --dataset="modelnet40" --model="mdcp" --model_path=$MPATH --pointer="transformer" --k_way 10 --m_shot 20 --n_query 40
python eval_fsl.py --dataset="scanobjectnn" --model="mdcp" --model_path=$MPATH --pointer="transformer" --k_way 10 --m_shot 20 --n_query 40