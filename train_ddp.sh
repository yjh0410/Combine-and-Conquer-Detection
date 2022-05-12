# 2 GPUs
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d voc \
                                                    -v ccdet_r50 \
                                                    --ema \
                                                    --fp16 \
                                                    --eval_epoch 10
