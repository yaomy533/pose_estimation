CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --workers 10 \
    --log_file '30' \
    --resume_posenet '/root/Source/ymy_dataset/trained/instance/cleargrasp/30/pose_model_236_0.00659505330696632.pth' \
    --dataset_root '/root/Source/ymy_dataset/cleargrasp' \
    --dataset 'cleargrasp' \
    --config_file 'config/cleargrasp_config.py' \
    --out_root '/root/Source/ymy_dataset/trained/instance/cleargrasp' \
    --batch_size 8

