import numpy as np

from .channel import Channel
from scipy.special import erf


class BpskAwgnChannel(Channel):
    def __init__(self, SINR, is_dB=True):
        super().__init__()
        if is_dB:
            self._SINR_dB = SINR
            self._SINR = np.power(10.0, SINR / 10.0)
        else:
            self._SINR = SINR
            self._SINR_dB = 10 * np.log10(SINR)

        self._BER = (1 - erf(np.sqrt(self._SINR))) / 2

        self._erasure_prob = - (self._BER * np.log(self._BER) + (1.0 - self._BER) * np.log(1.0 - self._BER))

        self._zero_LLR = np.log((1 - self._BER) / self._BER)
        self._one_LLR = np.log(self._BER / (1 - self._BER))

    def get_suffix(self):
        """

        :return:
        """
        return 'SINR={}_dB'.format(self._SINR_dB)

    def get_erasure_prob(self):
        """

        :return:
        """
        return self._erasure_prob

    def get_llr(self, out_symbol):
        """

        :param out_symbol:
        :return:
        """
        return self._one_LLR if out_symbol == 1 else self._zero_LLR

    def get_ber(self):
        """

        :return:
        """
        return self._BER

    def modulate(self, to_message):
        """

        :param to_message:
        :return:
        """
        return np.array([-1.0 if bit == 0 else 1.0 for bit in to_message], dtype='float128')

    def demodulate(self, from_message):
        """

        :param from_message:
        :return:
        """
        return np.array([0 if symbol <= 0.0 else 1 for symbol in from_message], dtype='uint8')

    def transmit(self, message):
        """

        :param message:
        :return:
        """
        noise_std = 1 / np.sqrt(2 * self._SINR)
        noise = noise_std * np.random.randn(len(message))

        return np.array(message + noise, dtype='float128')
