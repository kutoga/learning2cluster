from os import path
import librosa
from ast import literal_eval
from subprocess import call
import re
import itertools
import numpy as np

#############################################################################
# This tool may be used to play TIMIT snippets. The snippets specification
# can be copied from the HTML generated report (see Image/AudioDataProvider)
# Possible snippet specifications:
# FGMB0 [CONCAT] [1555-1655]
# FBAS0 [SA1] [200-300]
#############################################################################

# Configuration
TIMIT_dir = 'E:\\tmp\\test\\TIMIT'
TIMIT_list_files_name = 'list_files.txt'

# Load the list of all available audio files
list_files_file = path.join(TIMIT_dir, TIMIT_list_files_name)
with open(list_files_file, 'r') as fh:
    data = fh.read()
data = literal_eval(data)

# Define some audio functions
ffplay_exe = "C:\\Users\\bmeier\\Anaconda2\pkgs\\ffmpeg-2.7.0-0\\Scripts\\ffplay.exe"
sample_rate = 16000
step_size = 160
tmp_wav_file = 'E:\\tmp\\tmp.wav'
ms2st = lambda ms: ms * 160
def get_audio_content(file, ms_start=None, ms_end=None):
    y, sr = librosa.load(file, sr=sample_rate)
    if ms_start != None and ms_end != None:
        start = ms2st(ms_start)
        end = ms2st(ms_end)
        y_res = y[start:end]
    else:
        y_res = y
    return y_res
# def get_file_length_ms(file):
#     y, sr = librosa.load(file, sr=sample_rate)
#     return len(y) / 160
def play_file(file, ms_start, ms_end):
    y_res = get_audio_content(file, ms_start, ms_end)
    librosa.output.write_wav(tmp_wav_file, sr=sample_rate, y=y_res, norm=True)
    call([ffplay_exe, '-autoexit', tmp_wav_file])
def concat_and_play_files(files, ms_start, ms_end):
    y = np.asarray(list(itertools.chain(*[get_audio_content(file) for file in files])))
    start = ms2st(ms_start)
    end = ms2st(ms_end)
    y_res = y[start:end]
    librosa.output.write_wav(tmp_wav_file, sr=sample_rate, y=y_res, norm=True)
    call([ffplay_exe, '-autoexit', tmp_wav_file])

# While true: Parse the input and play the audio
while True:
    snippet_spec = input("Please input a snippet specification (e.g. 'FGMB0 [CONCAT] [1555-1655]' or 'FBAS0 [SA1] [100-200]'):\n")
    # snippet_spec = 'FGMB0 [CONCAT] [1555-1655]'

    # Parse the specification
    m = re.search('^([^ ]+) \[([^\]]+)\] \[(\d+)-(\d+)].*$', snippet_spec)
    if m is None:
        print("Invalid snippet specification. Abort.")
        break
    speaker = m.group(1)
    file = m.group(2)
    ms_start = int(m.group(3))
    ms_end = int(m.group(4))

    if not speaker in data:
        print("Could not find speaker. Abort.")
        break

    files = sorted(data[speaker])

    if file == 'CONCAT':
        files = list(map(lambda f: path.join(TIMIT_dir, f), files))
        f_play = lambda: concat_and_play_files(files, ms_start, ms_end)
    else:
        files = list(filter(lambda f: f.endswith('/' + file + '.WAV'), files))
        if len(files) != 1:
            print("Could not find the target file '{}'.".format(file))
            break
        f_play = lambda:play_file(path.join(TIMIT_dir, files[0]), ms_start, ms_end)

    while True:
        f_play()
        if input('Play again? y/[n]\n') != 'y':
            break

