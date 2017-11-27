"""
Script to check, if the speakers in the two given files are distinct.

Takes two arguments, file 1 and file 2. The files must be text files, with one speaker name per line.

python checker.py file1 file2
"""
import sys

with open(sys.argv[1], 'rb') as f:
    f1 = f.readlines()

with open(sys.argv[2], 'rb') as f:
    f2 = f.readlines()

identical_speakers = 0

for speaker_name in f1:
    if speaker_name in f2:
        identical_speakers += 1

if identical_speakers > 0:
    print('There are ' + str(identical_speakers) + ' identical speakers.')
else:
    print('Files are distinct.')
