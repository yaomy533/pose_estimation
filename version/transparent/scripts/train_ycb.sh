CUDA_VISIBLE_DEVICES=0 python train.py \
--workers 10 \
--log_file '01' \
--resume_posenet '' \
--dataset_root '/root/Source/ymy_dataset/YCB-V' \
--dataset 'ycb' \
--config_file 'config/ycb_config.py' \
--out_root '/root/Source/ymy_dataset/trained/instance/ycb-v'