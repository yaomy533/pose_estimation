CUDA_VISIBLE_DEVICES=1 python train.py \
    --log_file '24' \
    --cls_type 'all' \
    --workers 15 \
    --resume_posenet '' \
    --config_file 'config/linemod/lm_v3_1.py' \
#    --backbone_oly \
#    --eval_mode \
#    --start_epoch 6 \
#    --debug \
#    --retrain \
#    --dataset 'lm-bop'
#    --eval_mode \
