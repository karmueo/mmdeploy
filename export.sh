#!/bin/bash
mv $CONDA_PREFIX/lib/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6.bak
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$(pwd)/onnxruntime-linux-x64-gpu-1.8.1/lib:$LD_LIBRARY_PATH
python3 tools/deploy.py \
    configs/mmaction/video-recognition/video-recognition_onnxruntime_static.py \
    /home/tl/work/mmaction2/configs/recognition/uniformerv2/uniformerv2-base-p16-bird_uav_finetune.py \
    /home/tl/work/mmaction2/output/uniformerv2/epoch_50.pth \
    ./bird-3.20-2-target_8.mp4 \
    --work-dir mmdeploy_models/mmaction/uniformerv2/ort \
    --device cuda:0 \
    --dump-info

# mv $CONDA_PREFIX/lib/libstdc++.so.6.bak $CONDA_PREFIX/lib/libstdc++.so.6