import librosa
import numpy as np
from math import ceil

from impl.data.misc.helper import progress


class AudioHelper:
    def __init__(self, n_fft=512, sample_rate=16000, intensify_spectrogram=True, coefficients=192,
                 mel_coefficients=128, windows_length=320, normalize_spectrum=True, default_spectrum='normal'): # default_spectrum \in {'normal', 'mel'}
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.intensify_spectrogram = intensify_spectrogram
        self.coefficients = coefficients
        self.mel_coefficients = mel_coefficients
        self.windows_length = windows_length
        self.default_spectrum = default_spectrum
        self.normalize_spectrum = normalize_spectrum

        # Normalization values (obtained through testing): The new range is now in [0, 1]
        self.normalization_mel_offset = -80.
        self.normalization_mel_factor = 95.#6.3
        self.normalization_offset = 0
        self.normalization_factor = 2.2#4.2

    def __normalize(self, spectrogram, offset=0., factor=1., inverse=False):
        if not self.normalize_spectrum:
            return spectrogram
        if inverse:
            spectrogram = spectrogram * factor + offset
        else:
            spectrogram = (spectrogram - offset) / factor
        return spectrogram

    def __normalize_spectrogram(self, spectrogram, inverse=False):
        return self.__normalize(spectrogram, self.normalization_offset, self.normalization_factor, inverse=inverse)

    def __normalize_mel_spectrogram(self, mel_spectrogram, inverse=False):
        return self.__normalize(mel_spectrogram, self.normalization_mel_offset, self.normalization_mel_factor, inverse=inverse)

    def __get_hop_length(self):
        return int(10 * self.sample_rate / 1000)

    @staticmethod
    def normalize_audio_volume(input_file, output_file):
        y, sr = librosa.load(input_file, sr=None)
        librosa.output.write_wav(output_file, y, sr, norm=True)

    def audio_to_default_spectrogram(self, file):
        if self.default_spectrum == 'normal':
            return self.audio_to_spectrogram(file)
        elif self.default_spectrum == 'mel':
            return self.audio_to_mel_spectrogram(file)

    def default_spectrogram_to_audio(self, file, spectrogram, show_progress=True):
        if self.default_spectrum == 'normal':
            return self.spectrogram_to_audio(file, spectrogram, show_progress=show_progress)
        elif self.default_spectrum == 'mel':
            return self.mel_spectrogram_to_audio(file, spectrogram, show_progress=show_progress)

    def audio_to_raw_data(self, file):
        y, sr = librosa.load(file, sr=self.sample_rate)
        return y

    def raw_data_to_audio(self, file, raw, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        librosa.output.write_wav(file, raw, sample_rate, norm=True)

    def audio_to_spectrogram(self, file):
        y, sr = librosa.load(file, sr=self.sample_rate)
        S = librosa.stft(y, self.n_fft, hop_length=self.__get_hop_length(), win_length=self.windows_length)
        #p = np.angle(S)
        spectrogram = np.log1p(np.abs(S[np.newaxis, :, :]))[0, :, :]
        # if self.intensify_spectrogram:
        #     spectrogram = self.intensify(spectrogram)
        spectrogram = spectrogram[:self.coefficients, :]

        # Normalize
        spectrogram = self.__normalize_spectrogram(spectrogram)
        return spectrogram

    def audio_to_mel_spectrogram(self, file):
        # <test> Just for testing in parallel:
        # y, sr = librosa.load(file, sr=self.sample_rate)
        # print("librosa.stft(y, {}, hop_length={})".format(
        #     self.n_fft, self.__get_hop_length()
        # ))
        # S = librosa.stft(y, self.n_fft, hop_length=self.__get_hop_length())
        # S0 = np.abs(S[np.newaxis, :, :])[0, :, :] ** 2
        # S0_l = np.log1p(S0)
        # S0_e = np.exp(S0_l) - 1
        #
        # # S0 == S0_e
        # mel_basis = self.get_mel_basis()
        # S0_m = mel_spectrogram = np.dot(mel_basis, S0_e)
        #
        # </test>

        y, sr = librosa.load(file, sr=self.sample_rate)

        # Replace "librosa.feature.melspectrogram" by the following function
        def melspectrogram(y, sr, n_fft, hop_length, n_mels, window_length):
            # Based on:
            # librosa.feature.melspectrogram
            # librosa.core._spectrogram
            # Just added "win_length=window_length" (everything else should be original)

            S = None
            power = 2.0

            if S is not None:
                # Infer n_fft from spectrogram shape
                n_fft = 2 * (S.shape[0] - 1)
            else:
                # Otherwise, compute a magnitude spectrogram from input
                S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=window_length)) ** power

            # Build a Mel filter
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mels)

            return np.dot(mel_basis, S)

        mel_spectrogram = melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.__get_hop_length(),
                                         n_mels=self.mel_coefficients, window_length=self.windows_length)
        # mel_spectrogram2 = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.__get_hop_length(),
        #                                                  n_mels=self.mel_coefficients)

        if self.intensify_spectrogram:
            mel_spectrogram = self.intensify(mel_spectrogram)
        mel_spectrogram = mel_spectrogram[:self.mel_coefficients, :]

        # Normalize
        mel_spectrogram = self.__normalize_mel_spectrogram(mel_spectrogram)
        return mel_spectrogram

    def get_mel_basis(self):
        mel_basis = librosa.filters.mel(self.sample_rate, self.n_fft, n_mels=self.mel_coefficients)[:self.mel_coefficients, :self.coefficients]
        return mel_basis

    def spectrogram_to_mel_spectrogram(self, spectrogram):
        mel_basis = self.get_mel_basis()

        # Denormalize
        spectrogram = np.copy(spectrogram)
        spectrogram = self.__normalize_spectrogram(spectrogram, inverse=True)
        spectrogram = np.exp(spectrogram) - 1
        spectrogram = spectrogram ** 2

        mel_spectrogram = np.dot(mel_basis, spectrogram)

        # Intensify
        if self.intensify_spectrogram:
            mel_spectrogram = self.intensify(mel_spectrogram)

        # Normalize
        mel_spectrogram = self.__normalize_mel_spectrogram(mel_spectrogram)

        return mel_spectrogram

    def mel_spectrogram_to_spectrogtam(self, mel_spectrogram):
        mel_basis = self.get_mel_basis()

        # Denormalize
        mel_spectrogram = np.copy(mel_spectrogram)
        mel_spectrogram = self.__normalize_mel_spectrogram(mel_spectrogram, inverse=True)

        mel_spectrogram = np.copy(mel_spectrogram)
        if self.intensify_spectrogram:
            mel_spectrogram = self.intensify(mel_spectrogram, True)

        spectrogram = np.dot(np.transpose(mel_basis), mel_spectrogram)

        # TODO: Maybe we have to do a multiplication with a constant factor. Why? See:
        # https://groups.google.com/d/msg/librosa/E1RvuRB8aDI/vZY8FJ5RDgAJ

        # Post-process spectrogram (set every value at least to 1e-8: Because of numerical reasons we get sometimes very
        # small negative numbers)
        spectrogram = np.maximum(1e-8, spectrogram) ** 0.5
        spectrogram = np.log1p(spectrogram)

        # Normalize
        spectrogram = self.__normalize_spectrogram(spectrogram)

        return spectrogram

    def intensify(self, spectrogram, inverse=False):

        # Use now the power_to_db and db_to_power
        if not inverse:
            return librosa.power_to_db(spectrogram)
        else:
            return librosa.db_to_power(spectrogram)

        # f = 10000
        # c = 1
        # if not inverse:
        #     spectrogram = np.log10(c + f * spectrogram)
        # else:
        #     spectrogram = (np.power(spectrogram, 10) - c) / f
        # return spectrogram

    def spectrogram_to_audio(self, file, spectrogram, show_progress=True):
        a = np.zeros((self.n_fft // 2 + 1, spectrogram.shape[1]), dtype=float)

        # Denormalize
        spectrogram = self.__normalize_spectrogram(spectrogram, inverse=True)

        a[:spectrogram.shape[0], :] = np.exp(spectrogram) - 1

        if show_progress:
            print("Create: {}".format(file))
            def state(percentage):
                progress(percentage, 100)
        else:
            def state(percentage):
                pass

        # This code is supposed to do phase reconstruction
        p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
        count = 500
        hop_length = self.__get_hop_length()
        for i in range(count):
            state(i * 100. / count)
            S = a * np.exp(1j * p)
            x = librosa.istft(S, hop_length=hop_length)
            p = np.angle(librosa.stft(x, self.n_fft, hop_length=hop_length))
        state(100)

        librosa.output.write_wav(file, x, self.sample_rate, norm=True)

    def mel_spectrogram_to_audio(self, file, mel_spectrogram, show_progress=True):
        spectrogram = self.mel_spectrogram_to_spectrogtam(mel_spectrogram)
        self.spectrogram_to_audio(file, spectrogram, show_progress=show_progress)

    def get_settings_str(self):
        return "{}_n{}_m{}".format(self.default_spectrum, self.coefficients, self.mel_coefficients)

    @staticmethod
    def split_spectrogram_into_snippets(spectrogram, length, zero_padding=False):
        snippets = []
        spectrogram_length = spectrogram.shape[1]

        if zero_padding and (spectrogram_length % length != 0):

            # Do zero padding (this is relatively expensive)
            new_length = int(ceil(spectrogram_length * 1. / length)) * length
            new_spectrogram = np.zeros((spectrogram.shape[0], new_length))
            new_spectrogram[:, :spectrogram_length] = spectrogram
            spectrogram = new_spectrogram
            length = new_length

        # Create now the snippets
        snippet_count = int(spectrogram_length / length)

        # Create all snippets
        for i in range(snippet_count):
            snippets.append(spectrogram[:, (i * length):((i + 1)*length)])

        return snippets

    @staticmethod
    def select_random_spectrograms_snippet(spectrograms, length, zero_padding=False):

        # If multiple spectrograms are given they all must have the same LENGTH
        # At least one spectrogram must be given

        spectrogram = spectrograms[0]
        spec_length = spectrogram.shape[1]

        # Is the input spectrogram smaller than the required snippet?
        if spec_length < length:
            if zero_padding:
                results = []
                for spectrogram in spectrograms:
                    result = np.zeros((spectrogram.shape[0], length), dtype=spectrogram.dtype)
                    result[:, :spec_length] = spectrogram
                    results.append(result)
                return results
            else:
                return None # not possible

        # Select an index
        start = np.random.choice(spec_length + 1 - length)
        results = []
        for spectrogram in spectrograms:
            results.append(spectrogram[:, start:(start + length)])
        return results

    @staticmethod
    def select_random_spectrogram_snippet(spectrogram, length, zero_padding=False):
        return AudioHelper.select_random_spectrograms_snippet([spectrogram], length, zero_padding)[0]

    @staticmethod
    def select_random_raw_data_snippet(raw, length, zero_padding=False):

        # Check if the input is smaller than the requested snippet
        if raw.shape[0] < length:
            if zero_padding:
                res = np.zeros((length,), dtype=raw.dtype)
                res[:raw.shape[0]] = raw
                return res
            else:
                return raw

        # The input has at least the size of the requested snippet
        start = np.random.choice(raw.shape[0] + 1 - length)
        return raw[start:(start + length)]

def main():
    pass

if __name__ == '__main__':
    main()
