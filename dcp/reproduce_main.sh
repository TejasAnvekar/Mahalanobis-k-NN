python main.py --exp_name=dcp_v1_unseen --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --unseen=True
python main_mah.py --exp_name=mah_dcp_v1_unseen --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --unseen=True


python main.py --exp_name=dcp_v2_unseen --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --unseen=True
python main_mah.py --exp_name=mah_dcp_v2_unseen --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --unseen=True