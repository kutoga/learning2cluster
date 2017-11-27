"""
Script to generate list of speakers (speaker corpus).

Based on previous work of Gerber, Lukic and Vogt.
"""
import os
from random import randint

# Parameters
NUM_OF_SPEAKERS = 630
distinct = True  # if true, check if chosen speaker is distinct with list "FORBIDDEN_SPEAKERS"
num_speakers_clustering = 40
OUTPUT_FILE = '../../data/speaker_lists/speakers_40_not_clustering_vs_reynolds2.txt'
FORBIDDEN_SPEAKERS = '../../data/speaker_lists/speakers_40_clustering_vs_reynolds.txt'

speakers = []
oldSpeaker = ''

for root, directories, filenames in os.walk('../../data/training/TIMIT/'):
    for filename in filenames:
        if '_RIFF.WAV' in filename and oldSpeaker != root[-5:]:
            oldSpeaker = root[-5:]
            speakers.append(root[-5:])

y = []

if distinct:
    print('Generate distinct list.')
    not_to_use = []
    with open(FORBIDDEN_SPEAKERS, 'rb') as f:
        not_to_use = f.read().splitlines()

    while len(y) < num_speakers_clustering:
        idx = randint(0, len(speakers) - 1)
        speaker = speakers.pop(idx)
        if speaker not in not_to_use:
            y.append(speaker)
else:
    print('Generate probably non-distinct list')
    for i in range(NUM_OF_SPEAKERS):
        idx = randint(0, len(speakers))
        y.append(speakers.pop(idx))

female = 0
male = 0
with open(OUTPUT_FILE, 'wb') as f:
    for speaker in y:
        f.write(speaker)
        f.write('\n')
        if speaker[-5:-4] == 'M':
            male += 1
        else:
            female += 1

print('List successfully generated. %d women, %d men' % (female, male))
