gpu=false
network=lstm

feat_train=/export/b17/jcho/emotion_recognition/IEMOCAP/v2_20180307/data/pathNlab/8k_downsampled/mfcc_pitch/fold1/emo4/feats_train_emo4
feat_val=/export/b17/jcho/emotion_recognition/IEMOCAP/v2_20180307/data/pathNlab/8k_downsampled/mfcc_pitch/fold1/emo4/feats_cv_emo4
utt2emo=/export/b17/jcho/emotion_recognition/IEMOCAP/v2_20180307/data/pathNlab/8k_downsampled/mfcc_pitch/fold1/emo4/utt2emo_4emo

. utils/parse_options.sh
#echo 1stage
if [ ${gpu} == true ];then
    use_gpu=--gpu
fi
#echo 2stage
bash sub_gpu.sh \
    modules/ser_train.py \
    ${use_gpu} \
    --network ${network} \
    --feats-scp ${feat_train} \
    --feats-scp-val ${feat_val} \
    --utt2emo ${utt2emo} \
    --num-process 0
