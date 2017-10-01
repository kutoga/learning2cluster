
# TODO
# Requirements:
# - Output can be a 2D image
# - Output can be a 1D image
# - Allow different audio data types (spectrogram types; maybe use the AUdioHelper from VT2 as an argument for the audio data provider)

from impl.data.image.image_data_provider import ImageDataProvider

class AudioDataProvider(ImageDataProvider):
    def __init__(self):
        super().__init__()

