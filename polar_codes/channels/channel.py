from abc import ABCMeta, abstractmethod


class Channel(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_suffix(self):
        pass

    @abstractmethod
    def get_erasure_prob(self):
        pass

    @abstractmethod
    def get_llr(self, out_symbol):
        pass

    @abstractmethod
    def get_ber(self):
        pass

    @abstractmethod
    def modulate(self, to_message):
        pass

    @abstractmethod
    def demodulate(self, from_message):
        pass

    @abstractmethod
    def transmit(self, message):
        pass