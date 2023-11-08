# This script adapts the HuggingFace Wav2vec2 code to edit fully connected layers and replace them with their pruned versions
# Author: Oswaldo Ludwig


import sys
sys.stdout.flush()
import os
import torch
import transformers
import time
import torch.nn as nn
import codecs
import argparse
from timeit import default_timer as timer
import json
import numpy as np
import psutil
import pickle
import wave
from collections import OrderedDict

num_samples = 750
dic_index = list(range(0,32))

def wer(r, h):
    if (len(h) == 0) or (len(r) == 0):
       return(np.float32(1))
    else:
       d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
       d = d.reshape((len(r)+1, len(h)+1))
       for i in range(len(r)+1):
           for j in range(len(h)+1):
               if i == 0:
                   d[0][j] = j
               elif j == 0:
                   d[i][0] = i

       # computation:
       for i in range(1, len(r)+1):
           for j in range(1, len(h)+1):
               if (r[i-1] == h[j-1]):
                   d[i][j] = d[i-1][j-1]
               else:
                   substitution = d[i-1][j-1] + 1
                   insertion    = d[i][j-1] + 1
                   deletion     = d[i-1][j] + 1
                   d[i][j] = min(substitution, insertion, deletion)
       return (float(d[len(r)][len(h)])/float(len(r)))

def _printMemory(msg):
    p = psutil.Process()
    vms = getattr(p.memory_full_info(), 'vms')
    sys.stdout.flush()

def read_wav_file(file_path):
    try:
        wav_file = wave.open(file_path, 'rb')
        # Read the frames from the wave file
        frames = wav_file.readframes(-1)
        # Convert the byte string to a NumPy array
        audio_data = np.frombuffer(frames, dtype=np.int16)
        return audio_data
    except FileNotFoundError:
        print(f"No file found at {file_path}")
        return None

def dict_from_hrl(file_path):
    dictionary_list = []
    with open(file_path, 'r') as file:
        file.readline()
        headers = file.readline().strip().split('#')[1:]  # read the second line as headers
        file.readline()
        for line in file:
            values = line.strip().split('#')
            dictionary = {header: value for header, value in zip(headers, values)}
            dictionary_list.append(dictionary)
    return dictionary_list

BAD_SCORE = 100000

class Vocabulary(object):

    def __init__(self, tokenizer, wordlistFilepath):
        self._tokenizer = tokenizer
        self._wordlist = set()
        self._prefixes = set()
        with open(wordlistFilepath, 'r') as fd:
            for line in fd:
                word = line.strip().upper()+' '
                self._wordlist.add(word)
                for i in range(1, len(word) + 1):
                    self._prefixes.add(word[:i])

    def translate(self, ids):
        return self._tokenizer.convert_ids_to_tokens(ids)

    def score(self, i, history):
        partialtokens = self._tokenizer.convert_ids_to_tokens(history+[i])
        if partialtokens[-1] == self._tokenizer.eos_token:
            partialtokens[-1] = self._tokenizer.word_delimiter_token
        partialstr = self._tokenizer.convert_tokens_to_string(partialtokens)
        lastspace = partialstr.rstrip().rfind(' ')
        if partialtokens[-1] == self._tokenizer.word_delimiter_token:
            partialstr += ' '
        if lastspace != -1:
            partialstr = partialstr[lastspace+1:]
        if partialstr in self._prefixes:
            return 0
        else:
            return BAD_SCORE

# beam search
def beam_search_decoder(data, maxhypos, lm=None):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                scorej = score
                if lm: scorej += lm.score(j, seq)
                candidate = [seq + [j], scorej - np.log(row[j])]
                if candidate[1] < BAD_SCORE:
                    all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select maxhypos best
        sequences = ordered[:maxhypos]
    return sequences


class Wav2Vec(object):
    def __init__(self, args):
        self._name = args.model
        self._maxhypos = args.accuracy
        self._nbest = args.maxnbest
        self._memlog = args.logMemory
        self._reloadNbr = args.reload
        self._wordlistFilepath = args.wordlistFilepath

        # load pretrained model
        self._processor = None
        self._model = None
        self._processed = 0
        self._vocab = None
        self._reload()

    def _reload(self):
        del self._processor
        del self._model
        import torch
        import transformers
        self._processor = transformers.Wav2Vec2Processor.from_pretrained(self._name)
        self._model = transformers.Wav2Vec2ForCTC.from_pretrained(self._name, output_hidden_states=True)

        if self._wordlistFilepath:
            self._vocab = Vocabulary(self._processor.tokenizer, self._wordlistFilepath)

        self._processed = 0
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model.to(self._device)

    def process(self, entry):
        if self._reloadNbr and self._processed % self._reloadNbr:
            self._reload()
        filename = entry['audio_input']
        #filename = entry['speechfile']
        if self._memlog: print('')  #  print('process %s' % filename, file=sys.stdout)
        if self._memlog: _printMemory('>>>')
        # pad input values and return pt tensor
        inference_start = timer()

        # here is how to get the transcription for calculating the WER
        sys.stdout.flush()
        input_values = self._processor(entry['audio_input'], sampling_rate=16000, return_tensors="pt").input_values.to(self._device).float()
        if self._memlog: _printMemory('... input')

        results = OrderedDict()

        # retrieve logits
        with torch.no_grad():
            logits = self._model(input_values).logits
            X = input_values.cpu().detach().numpy()
            logits = torch.nn.functional.softmax(logits.float(), dim=-1)
            Logits = np.array(logits.cpu().detach().numpy())

        del entry['audio_input']
        self._processed += 1
        return (X, Logits) # utt

def main():
    parser = argparse.ArgumentParser(description='Running wav2vec inference.')
    parser.add_argument('-m', '--model', required=True, help='Name of the model (automatically downloaded)')
    parser.add_argument('-i',"--inputFilepath", required=True, help="input hrl file")
    parser.add_argument('-o',"--outputFilepath", required=True, help="output res file")
    parser.add_argument('-s',"--soundFiledir", help="The sound file directory")
    parser.add_argument('-f',"--soundFileFormat", default="wav", help="The sound file format")
    parser.add_argument('-b','--accuracy', type=int, default=10, help="Maximum number of beam search hypotheses")
    parser.add_argument('-n', '--maxnbest', type=int, default=1, help="Maximum number of n-best results")
    parser.add_argument('--wordlistFilepath', help="vocabulary wordlist file")
    parser.add_argument("--logMemory", action='store_true', default=False, help="Print memory logging")
    parser.add_argument('--numberOfLinesPerTest', type=int, default=None, help="Number of lines to process")
    parser.add_argument("--reload", type=int, default=0, help="Modules reload frequency")
    parser.add_argument('--version', action='version', version='%(prog)s 2.0.0')
    parser.add_argument('--pruned_idx', help="A list of indexes of pruned indexes. It must be a multiple of 24, as the model has 24 layers")
    parser.add_argument('--idx_individual', type=int, default=10, help="index of current individual")
    parser.add_argument('--granularity', type=int, default=32, help="the pruning granularity")
    parser.add_argument('--saveModel', type=int, default=0, help="0 for evaluating without saving, 1 for saving the pruned model")
    parser.add_argument('--subsampling', type=float, default=1.0, help="Subsampling rate in the interval (0, 1]")

    args = parser.parse_args()
    n_heads = args.granularity
    print("Granularity set to " + str(n_heads))

    if not(args.soundFiledir == ''):

      reader = dict_from_hrl(args.inputFilepath)
      wav2vec = Wav2Vec(args)

      print('get the list of tensors to be prunned...')
      # Here I get the list of tensors to be prunned:

      if not(args.pruned_idx == 'None'):
         pruning_indexes = list(np.fromstring(args.pruned_idx.replace(".0", ""), dtype=int, sep=','))
         n_pruned_idx = int(len(pruning_indexes)/24)

      else:
         pruning_indexes = []
         n_pruned_idx = 0

      # Generating the dictionary of characters:

      processor = transformers.Wav2Vec2Processor.from_pretrained(args.model)
      tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
      tock = []
      for t in dic_index:
        tock.append(processor.decode(t))

      token_dict = dict(zip(dic_index, tock))
      with open('token_dictionary', 'wb') as handle:
         pickle.dump(token_dict, handle)

      # HERE IS HOW TO PRUNE THE W2V2 MODEL:

      Params = list(wav2vec._model.named_parameters())
      for count_layer in range(24):
            idxs = pruning_indexes[(count_layer * n_pruned_idx) : ((count_layer + 1) * n_pruned_idx)]
            lines = []
            for idx in idxs:
               lines += list(range((idx * int(4096/n_heads)), ((idx + 1) * int(4096/n_heads))))
            for name, Tensor in Params:
                if name == "wav2vec2.encoder.layers." + str(count_layer) + ".feed_forward.intermediate_dense.weight":
                   # getting the weight:
                   weight = Tensor.cpu().detach().numpy()
                   # pruning lines:
                   weight = np.delete(weight, lines, 0)
                   # Attributing this pruned weight back:
                   wav2vec._model.wav2vec2.encoder.layers[count_layer].feed_forward.intermediate_dense.weight = nn.Parameter(torch.from_numpy(weight))

                if name == "wav2vec2.encoder.layers." + str(count_layer) + ".feed_forward.intermediate_dense.bias":
                   # getting the weight:
                   bias = Tensor.cpu().detach().numpy()
                   # pruning:
                   bias = np.delete(bias, lines, 0)
                   # Attributing this weight back:
                   wav2vec._model.wav2vec2.encoder.layers[count_layer].feed_forward.intermediate_dense.bias = nn.Parameter(torch.from_numpy(bias))

                if name == "wav2vec2.encoder.layers." + str(count_layer) + ".feed_forward.output_dense.weight":
                   # getting the weight:
                   weight = Tensor.cpu().detach().numpy()
                   # pruning:
                   weight = np.delete(weight, lines, 1)
                   # Attributing this weight back:
                   wav2vec._model.wav2vec2.encoder.layers[count_layer].feed_forward.output_dense.weight = nn.Parameter(torch.from_numpy(weight))

      count = 0

      print('SAMPLING THE DATA FOR PRUNED MODEL EVAL...')
      # HERE IS WHERE I SAMPLE THE DATA FOR PRUNED MODEL EVAL:
      avg_WER = 0

      for entry in reader:
        #composing the mini-batches:
        if np.random.rand() < args.subsampling:  #  sampling with sampling_prob probability
            # here the modification to avoid PyTorch preprocessing:
            entry['audio_input'] = read_wav_file(args.soundFiledir + "/" + entry['audio_input'])
            target = entry['transcription']
            print('The target transcription is:')
            print(target)
            X_processed, Y = wav2vec.process(entry)
            pred_ids = np.argmax(Y, axis=-1)[0]  #  Greedy decoder to be fast, as we aren't using LM
            transcription = tokenizer.decode(pred_ids, output_word_offsets=True).text
            print("The predicted transcription is:")
            print(transcription)
            WER = wer(list(target.lower().split(" ")), list(transcription.lower().split(" ")))
            avg_WER += WER
            print("WER:")
            print(WER)
            count += 1

        if count > num_samples:
            break
      print('DATA SAMPLING DONE.')
      avg_WER = avg_WER/count

      print('avg_WER:')
      print(avg_WER)
      np.save('fitness' + str(args.idx_individual) + '.npy', avg_WER)

      if args.saveModel==1:
         print(dir(wav2vec._model))
         try:
            os.makedirs("./pruned_w2v")
         except:
            print('Checkpoint directory for pruned model already exists')
         wav2vec._model.save_pretrained("./pruned_w2v")


if __name__ == '__main__':
    main()
