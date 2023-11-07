# This script implements a GA to prune the W2V2 model using multi-CPU processing on a computer grid.
# To do this, it generates and runs a Shel script to distribute copies of the W2V2 decoder with different pruning setups/chromosomes.
# Author: Oswaldo Ludwig
# In the case of using this code (or pieces of this codebase) cite:
# O. Ludwig and T. Claes, "Compressing Wav2vec2 for Embedded Applications,"
# 2023 IEEE 33rd International Workshop on Machine Learning for Signal Processing (MLSP), Rome, Italy, 2023, pp. 1-6, doi: 10.1109/MLSP55844.2023.10285964.

import numpy as np
import os
import time
import subprocess as commands
import argparse

parser = argparse.ArgumentParser(description='Running GA pruning for wav2vec.')
parser.add_argument('--PopSize', type=int, default=100, help="Size of the GA population")
parser.add_argument("--SelectPressure", type=float, default=3.0, help="Selective pressure in the range [2, 6]")
parser.add_argument('--nHeads', type=int, default=32, help="Number of heads")
parser.add_argument('--nLayers', type=int, default=24, help="Number of layers")
parser.add_argument('--nPruned', type=int, default=20, help="Number of pruned structures")
parser.add_argument('--nGenerations', type=int, default=70, help="Number of GA generations")
parser.add_argument('--initialChrom', default=[21,28,29,8,9,18,20,14,26,12,17,6,16,7,27,30,15,11,3,23,15,12,26,23,10,20,11,28,24,13,5,30,29,25,7,14,31,1,4,22,23,28,26,30,11,7,19,14,16,29,24,22,27,1,5,15,6,0,20,10,0,4,23,17,11,12,1,25,7,24,9,29,6,10,30,13,5,15,16,2,11,25,4,22,6,29,27,2,10,15,12,24,7,9,30,13,0,16,23,3,19,0,9,22,5,6,28,3,8,16,4,18,14,30,2,7,24,17,11,15,8,20,9,28,7,26,4,17,19,31,27,30,13,10,23,6,21,2,22,11,1,31,16,0,13,29,19,21,4,14,2,15,22,30,6,12,28,10,20,11,28,14,20,25,15,3,0,29,1,13,17,19,6,11,8,16,22,21,24,30,1,14,0,23,8,25,26,10,16,17,28,15,13,3,11,30,12,31,27,22,17,14,19,16,21,8,23,2,31,27,29,4,1,9,5,12,3,7,13,26,20,4,9,26,19,27,11,28,24,29,8,13,16,30,7,18,3,6,17,31,31,28,17,14,7,29,22,27,1,18,16,23,26,24,9,13,6,20,21,4,24,14,7,11,9,29,13,2,25,18,6,26,1,17,15,0,30,8,31,21,12,9,21,29,19,3,1,15,6,27,0,16,2,30,18,4,13,22,14,8,5,23,4,6,11,17,24,27,22,25,15,13,19,21,0,2,12,14,20,7,4,30,23,19,27,8,9,21,31,13,10,24,7,11,0,17,12,26,15,25,2,24,0,25,7,14,30,5,28,4,31,6,18,3,23,16,21,22,1,20,0,26,6,23,13,22,20,4,16,15,27,10,2,21,14,11,25,19,5,17,7,30,19,3,11,17,21,20,6,23,0,4,9,14,16,15,26,22,12,1,10,31,22,9,28,21,12,8,24,4,15,16,5,19,23,7,27,2,29,26,14,15,19,1,17,26,11,0,5,8,25,2,10,18,3,7,31,6,4,28,17,22,13,20,29,16,8,18,1,9,30,31,24,4,28,23,19,3,27,6,7,30,21,19,9,10,25,2,4,12,16,28,14,0,6,8,5,17,22,1], help="Initialization from best L1-norm pruning for rapid convergence")
parser.add_argument('--workDir', default='/asr/users/oswaldo_ludwig/E2E_ASR/train_Cerence_data/w2v/GA_pruning', help="The work directory")
parser.add_argument('--bindings', default='/opt/slurm,/asr/users/oswaldo_ludwig,/home/halboth,/mnt/asr/scratch/work/halboth,/home/halboth,/mnt/asr/am/data/train,/home/halboth/test_cd,/datateam,/asr/users/halboth,/home/halboth,/asr/build/users/halboth,/asr/build/users/halboth,/mnt,/opt/slurm,/var/run/munge,/lm,/asr/oneasr_cloud,/asr/test,/datateam,/asr/am/data/train,/asr/build', help="Container bindings")
parser.add_argument('--container', default='/mnt/users/oswaldo_ludwig/E2E_ASR/train_Cerence_data/w2v/GA_pruning/container_for_w2v.sif', help="Path to Container")
parser.add_argument('--w2vCheckpoint', default="./checkpoint_HuggingFace/huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english/resolve/main/", help="The path to the W2V2 checkpoint")
parser.add_argument('--data', default="/asr/users/oswaldo_ludwig/E2E_ASR/train_Cerence_data/knowledge_distillation2/student_test/all_shuf.hrl", help="The training data")
parser.add_argument('--soundDir', default="/mnt/asr/scratch/work/halboth/wav2vec/data/speech/", help="Path to sound files directory")
parser.add_argument('--subsampling', type=float, default=1.0, help="Subsampling rate in the interval (0, 1]")

args = parser.parse_args()

pop_size = args.PopSize
pressure = args.SelectPressure
n_heads = args.nHeads
n_layers = args.nLayers
n_pruned_heads = args.nPruned
tot_gen = args.nGenerations
L1_pruning = args.initialChrom
granularity = n_heads
best_chr = L1_pruning

# Defining string of commands to generate the master file used in parallelizing the GA population assessment process:
head = '#!/bin/bash \ncd '
head += args.workDir
head += ' \nexport SINGULARITY_BIND="'
head += args.bindings + '"'
head += ' \n\n[[ -n "${!SLURM_*}" ]] && unset ${!SLURM_*} || true\n\n'
slurm = '/opt/slurm/bin/srun --mem=32G singularity exec '
slurm += args.container
slurm += ' '
cmd0 = "python ./test_w2v.py --pruned_idx="
cmd1 = " --model="
cmd1 += args.w2vCheckpoint
cmd1 += " --idx_individual="
cmd2 = " --inputFilepath="
cmd2 += args.data
cmd2 += " --soundFiledir="
cmd2 += args.soundDir
cmd2 += " --outputFilepath=train.res"
cmd2 += " --subsampling=" + str(args.subsampling)
cmd3 = " --maxnbest=1 &> test_w2v.log"

def toString(b):
    txt = ''
    for i in b:
        txt+= str(i) + ','
    return (txt[0:-1])

def generate_chromosome(n_heads, n_layers, n_pruned_heads):
   chromosome = ''
   for l in range(n_layers):
      sub_chromosome = ''
      for h in range(n_pruned_heads):
         gene = str(int(np.random.rand() * n_heads - 0.01))
         while gene in sub_chromosome:
            gene = str(int(np.random.rand() * n_heads - 0.01))
         sub_chromosome += gene + ','
      chromosome += sub_chromosome
   return(chromosome)

def gen_init_pop(pop_size, n_heads, n_layers, n_pruned_heads):
   population = np.zeros((pop_size, (n_pruned_heads * n_layers)))
   for k in range(pop_size):
      population[k,:] = np.fromstring(generate_chromosome(n_heads, n_layers, n_pruned_heads), dtype=int, sep=',')
   return(population.astype(int))

def Eval(population):
   # here I create a Shell script with all the commands to parallelize the population evaluation on the grid
   Cmd = head
   for k in range(pop_size):
      individual = str(k) + " & \n"
      chromosome =  toString(population[k,:])
      Cmd += slurm + cmd0 + chromosome + " --granularity=" + str(granularity) + cmd1 + str(k) + cmd2 + cmd3 + individual
   Cmd = Cmd[0:-3]
   with open("population_eval.sh", "w") as text_file:
      text_file.write(Cmd)
   os.system('bash ' + "population_eval.sh")
   fitness = np.zeros(pop_size)
   for k in range(pop_size):
      time_count = 0
      while (time_count < 7200) and not(os.path.isfile('fitness' + str(k) + '.npy')):  #  waiting 7200 sec (2h) to deal with eventual GPU shortage
         time.sleep(1)
         time_count += 1
      if not(os.path.isfile('fitness' + str(k) + '.npy')):
         fitness[k] = 1  #  a big WER for avoiding using this individual during cross-over
      else:
         try:
            fitness[k] = np.load('fitness' + str(k) + '.npy')
         except:
            print('Another corrupted file...')
            fitness[k] = 1
         os.system('rm ' + 'fitness' + str(k) + '.npy')
   print(min(fitness))
   status, output = commands.getstatusoutput("squeue -u oswaldo.ludwig")
   while len(output.split())>9:
      jobID = output.split()[8]
      os.system('scancel ' + jobID)
      status, output = commands.getstatusoutput("squeue -u oswaldo.ludwig")
   return(fitness)

# *******************
# Here starts the GA
# *******************

# generating the initial population:

population = gen_init_pop(pop_size, n_heads, n_layers, n_pruned_heads)
population[20] = np.array(L1_pruning)

# loop over generations:

fitness = np.zeros(pop_size)
count = 0
status = ''
best_fitness = 10000
for gen in range(tot_gen):
    # evaluating the fitness of all individuals:
    fitness = Eval(population)
    # ranking the individuals by their fitness:
    rank = np.argsort(fitness)
    status = 'generation '+ str(gen) + ', mean fitness: ' + str(np.mean(fitness))  + ', std fitness: ' + str(np.std(fitness)) + ', best fitness: ' + str(np.min(fitness))
    print(status, flush=True)
    # selecting individuals for crossover:
    new_population = - np.ones(population.shape)
    for i in range(pop_size):
                        v1 = np.random.rand()
                        p1 = min(int((pop_size -1) * (np.exp(pressure * v1) - 1) / (np.exp(pressure) - 1)),(pop_size -1))
                        v2 = np.random.rand()
                        p2 = min(int((pop_size -1) * (np.exp(pressure * v2) - 1) / (np.exp(pressure) - 1)),(pop_size -1))
                        while p1 == p2:
                            v2 = np.random.rand()
                            p2 = min(int((pop_size -1) * (np.exp(pressure * v2) - 1) / (np.exp(pressure) - 1)),(pop_size -1))
                        # applying the crossover operation with the constraints (not repeting the head indexes per layer):
                        count = 0
                        for l in range(n_layers):
                           init_count = count
                           piece1 = population[rank[p1],count:(count+n_pruned_heads)]
                           piece2 = population[rank[p2],count:(count+n_pruned_heads)]
                           for p in range(n_pruned_heads):
                                r = np.random.rand()
                                if (r > 0.5) and (not( population[rank[p1],count] in new_population[i,init_count:count])):
                                    new_population[i,count] = population[rank[p1],count]
                                else:
                                    if not( population[rank[p2],count] in new_population[i,init_count:count]):
                                       new_population[i,count] = population[rank[p2],count]
                                    else:
                                       rand_try = int(n_heads * np.random.rand())
                                       while rand_try in new_population[i,init_count:count]:
                                          rand_try = int(n_heads * np.random.rand())
                                       new_population[i,count] = rand_try
                                count += 1
    # replacing the old population:
    pop0 = population[rank[0]]
    population = new_population
    if gen == 10:
       population[10] = np.array(best_chr)
    if min(fitness) <= best_fitness:
       bestChrom = pop0  #  for elitism
       best_fitness = min(fitness) # if it is evolving, no elitism is needed
    else:
       print('Best individual degraded, so elitism will be applied to rescue the best individual...')
       population[-1] = bestChrom  # if degraded, elitism is applied
    print('Best chromosome = ' + str(bestChrom), flush=True)
