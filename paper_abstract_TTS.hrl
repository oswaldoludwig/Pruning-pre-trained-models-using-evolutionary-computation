#head;hrl;2.0;utf-8#source reference_paper
#ref#audio_input#speaker#transcription
head#test
ref#file0.wav#CNTTS#COMPRESSING WAV2VEC2 FOR EMBEDDED APPLICATIONS
ref#file1.wav#CNTTS#Wav2vec2 self-supervised multilingual training learns speech units common to multiple languages, leading to better generalization capacity.
ref#file2.wav#CNTTS#However, Wav2vec2 is larger than other E2E ASR models such as the Conformer ASR.
ref#file3.wav#CNTTS#Therefore, the objective of this work is to reduce the Wav2vec footprint by pruning lines from the intermediate dense layers of the encoder block, since they represent about two thirds of the encoder parameters.
ref#file4.wav#CNTTS#We apply Genetic Algorithms to solve the combinatorial optimization problem associated with pruning, which means running many copies of the Wav2vec2 decoder in parallel using multiprocessing on a computer grid, so an effort was made to optimize the GA for good performance with few CPUs.
ref#file5.wav#CNTTS#The experiments show a small absolute word error rate damage of 0.21 percent for a pruning of 40 percent and compare this value with those of the usual L1-norm pruning and model restructuring by singular value decomposition.
ref#file6.wav#CNTTS#We proposed a new neuroevolution-based method to solve thecombinatorial optimization problem associated with pruningand compared it with the usual L1-norm pruning and SVD-based model restructuring.
ref#file7.wav#CNTTS#This method can be applied to any pretrained Transformer-based model to preserve as much information as possible from its pre-training.
ref#file8.wav#CNTTS#Here, the idea isto preserve Wav2vec2 generalization capacity, which is animportant feature for applications in an extensive languageportfolio that includes poorly resourced languages in the cor-pus linguistics sense.
ref#file9.wav#CNTTS#The proposed GA-based pruning required 62 CPUs forabout a day to find the optimal pruning setup. The experimental results support our method, showing a small relative damage of 1.26 percent for a pruning of about 40 percent of the model parameters. 
ref#file10.wav#CNTTS#In general, our experiments indicate that focusing on the intermediate FC layers is a good way to achieve a high compression ratio with little impact on performance, an equally mall relative damage of 1.17 percent was achieved by restructuring these layers by SVD, which takes just a few seconds tocompute using a SciPy library.   
ref#file11.wav#CNTTS#However, due to its more serial nature, the resulting restructured model tends not to beas computationally efficient at runtime as the pruned model,which allows for better parallelism. 
