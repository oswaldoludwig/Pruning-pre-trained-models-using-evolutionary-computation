#!/bin/bash

cd /asr/users/oswaldo_ludwig/E2E_ASR/train_Cerence_data/w2v/GA_pruning

echo 'defining binds...'
export SINGULARITY_BIND="/opt/slurm,/asr/users/oswaldo_ludwig"

echo 'calling GA model...'
srun --mem=48G singularity exec container_for_w2v.sif python /asr/users/oswaldo_ludwig/E2E_ASR/train_Cerence_data/w2v/GA_pruning/distributed_GA.py --data="./paper_abstract_TTS.hrl" --soundDir="./wav_files_paper_abstract/" --SelectPressure=4 --PopSize=130 --subsampling=0.8 >> distributed.log
