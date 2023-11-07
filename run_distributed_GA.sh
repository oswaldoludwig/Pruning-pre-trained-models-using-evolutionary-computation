#!/bin/bash

cd /asr/users/oswaldo_ludwig/E2E_ASR/train_Cerence_data/w2v/GA_pruning

echo 'defining binds...'
export SINGULARITY_BIND="/opt/slurm,/asr/users/oswaldo_ludwig,/home/halboth/scratch/wav2vec/data,/home/halboth/scratch/wav2vec/data/speech/datateam,/home/halboth/scratch/wav2vec/data/speech/mnt/asr/am/data/train/ENU/mobile_vlingo,/home/halboth/scratch/wav2vec/data_enu,/asr/users/oswaldo_ludwig/E2E_ASR/train_Cerence_data/knowledge_distillation2"

echo 'calling GA model...'
srun --mem=48G singularity exec container_for_w2v.sif python /asr/users/oswaldo_ludwig/E2E_ASR/train_Cerence_data/w2v/GA_pruning/distributed_GA.py --data="./paper_abstract_TTS.hrl" --soundDir="./wav_files_paper_abstract/" --SelectPressure=4 --PopSize=30 --subsampling=0.8 >> distributed.log
