
python main.py \
--pretrain_model 'Sage' --finetune_model 'Sage' --device 'cpu' \
--source_val_frac 0.1 --rs 2023 \
--pretrain_embedding 256 --pretrain_hidden_feats [256,256] --pretrain_batch_size 2048 --pretrain_epoch 10 \
--pretrain_sample_neighbor [-1,-1] \
--finetune_epoch 10 --finetune_batch_size 1024 \
--recall_level 2 --min_recall 10