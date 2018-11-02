import numpy as np
import os.path

from numpy import arctanh, tanh
from sys import exit

from .channels.channel import Channel

from . import PACKAGE_DIRECTORY

CHECK_NODE_TANH_THRES = 44


class PolarCode:
    """ Implements the concept of polar code.

    Any polar code is defined with the length of codeword N = 2 ** n and positions of K information bits.
    To define a polar code one should provide exponent an N, number of information bits K.

    Also, to define a code instance one should specify which method should be used to construct
    the code (Polarization Weight (PW),. The positions of the information bits will be obtained depending on the chosen
    method.

    In addition, one should define which decoder to use (e.g. Successive Cancellation (SC)).

    Attributes:
        n: An integer exponent of the two which defines the codeword length (N = 2 ** n);

        K: An integer number of information bits;

        N: An integer length of codeword;

        construction_methods: A dict {name:method} of available construction methods;

        decoding_methods: A dict {name:method} of available decoders;

        info_bits_positions: A tuple of information bit indices in the codeword;

        frozen_bits_positions: A tuple of frozen bits (equal to 0) indices in the codeword.

    """
    CRC_polynomials = {
        8: np.asarray([1, 1, 1, 0, 1, 0, 1, 0, 1], dtype='uint8'),
        16: np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype='uint8'),
    }

    def __init__(self, n, K, channel, construction_method, CRC_len=0):
        """
        Initializes the attributes of the BscPolarCode class instance and constructs it.

        :param n:
        :param K:
        :param channel:
        :param construction_method:
        :param CRC_len:
        """
        self._n = n
        self._K_minus_CRC = K
        self._channel = channel
        self._construction_method = construction_method
        self._CRC_len = CRC_len

        if not (CRC_len in PolarCode.CRC_polynomials.keys() or CRC_len == 0):
            print('Wrong length of CRC was passed to the {} constructor.'
                  'Supported CRC lengths are 0 and {}'.format(self.__class__.__name__,
                                                              PolarCode.CRC_polynomials.keys()))
        self._K = self._K_minus_CRC + self._CRC_len

        if not isinstance(self._channel, Channel):
            print('Wrong channel was passed to the {} constructor. '
                  'An instance of {} is required, while only {} was given'.format(self.__class__.__name__,
                                                                                  Channel.__name__,
                                                                                  self._channel.__class__.__name__))
            exit(-1)

        self._N = 2 ** self._n

        if self._K > self._N:
            print('Wrong parameters were passed to the {} constructor. '
                  'Pretending to have K = {} and CRC length = {} while N = {}'.format(self.__class__.__name__,
                                                                                      self._K - self._CRC_len,
                                                                                      self._CRC_len,
                                                                                      self._N))

        self._construction_methods = {
            'PW': self._pw_construction,
            'IBEC': self._independent_bec_construction,
            'DBEC': self._dependent_bec_construction,
        }

        self._decoding_methods = {
            'SSC': self._slow_sc_decode,
            'FSC': self._fast_sc_decode,
            'TVD': self._tal_vardy_decode,
            'SCL': self._scl_decode,
        }

        self._info_bits_positions = None
        self._frozen_bits_positions = None
        self._construct(self._construction_method)

        # Attributes required for the simple Tal-Vardy decoder
        self._p_arr = None
        self._b_arr = None

        # Attributes required for the list Tal-Vardy decoder
        self._inactive_path_indices = None
        self._active_path = None
        self._array_pointer_p = None
        self._array_pointer_c = None
        self._path_index_to_array_index = None
        self._inactive_array_indices = None
        self._array_reference_count = None

    def _construct(self, construction_method):
        """
        Constructs the code, i.e. defines which bits are informational and which are frozen.

        Two behaviours are possible:
        1) If there is previously saved data with the sorted indices of channels for given N, QBER
        and construction method, it loads this data and uses it to define sets of informational and frozen bits.
        2) Otherwise, it calls the preferred method from the dict of methods to construct the code. Then, it
        saves sorted indices of channels and finally defines sets of informational and frozen bits.

        :param construction_method: A string defining which of the construction method to use;
        :return: void.
        """
        try:
            # Define the name where the dumped data should be stored
            construction_path = self._get_construction_path()
            construction_name = construction_path + '{}.npy'.format(construction_method)

            # If the file with data exists, load ordered_channels
            if os.path.isfile(construction_name):
                construction = np.load(construction_name)
            # Otherwise, obtain construction and save it to the file
            else:
                construction = self._construction_methods[construction_method]()

                if not os.path.exists(construction_path):
                    os.makedirs(construction_path)
                np.save(construction_name, construction)

            # Take first K bits as informational
            self._info_bits_positions = tuple(sorted(construction[:self._K]))
            # Others are frozen
            self._frozen_bits_positions = tuple(set(range(0, self._N)) - set(self._info_bits_positions))
        except KeyError as wrong_key:
            print('Unable to construct a {}: '
                  'there is no {} construction method'.format(self.__class__.__name__, wrong_key))
            exit(-1)

    def _get_construction_path(self):
        """
        Returns the path to a file containing a construction of the code (i.e. indices of bits in codeword
        sorted in the descending order of their "quality". The path depends on codeword length and the
        chosen construction method. All construction are stored in the package folder.

        :return: A string with absolute path to the code construction.
        """
        construction_path = PACKAGE_DIRECTORY
        construction_path += '/polar_codes_constructions/'
        construction_path += 'N={}/'.format(self._N)
        construction_path += '{}/'.format(self._channel.get_suffix())

        return construction_path

    def _pw_construction(self):
        """
        Implements the PW method of polar code construction. For each polarized channel it
        calculates the beta-expansion of its index i \in (0, N-1). The polarized channel used to transmit
        information bits  are those K of them which have the highest value of calculated
        beta-expansion (also called Polarizarion Weight).

        :return: An array of polarized channel indices sorted from the best to the worst for information
        transmission
        """
        # Parameter used to build the expansions. Allows achieving the optimal performance with value close
        # to 2 ^ (1/4).
        beta = np.power(2, 0.25)

        # Calculate polar weight for each of the polarized channel.
        polar_weights = [PolarCode.beta_expand(i, beta) for i in range(0, self._N)]

        # Sort all the channels in the descending order of their PW since the higher PW is, the more
        # suitable channel is to transmit the information.
        construction = [_[0] for _ in sorted(enumerate(polar_weights), key=lambda tup:tup[1])][::-1]

        return construction

    @staticmethod
    def beta_expand(idx, beta):
        """
        Calculates the beta-expansion of integer idx. Beta-expansion is the value of polynomial with coefficients
        come from the binary representation of idx calculated in the point beta.

        :param idx: An integer index of the polarized channel;
        :param beta: A double value which defines at which point the value of polynomial should be calculated;
        :return: beta_expansion -- A double value of the polynomial in the point beta.
        """
        beta_expansion = 0.0
        exponent = 0

        while idx:
            beta_expansion += np.power(beta, exponent) * (idx & 1)
            idx >>= 1
            exponent += 1

        return beta_expansion

    def _independent_bec_construction(self):
        """
        Implements the BEC method of polar construction. This method was decribed in the original Arikan's paper
        and it is based on how the Bhattacharya Z-parameters evolve during the polarization.

        :return: An array of polarized channel indices sorted from the best to the worst for information
        transmission
        """
        # Calculate Z-parameter for each of N polarized channels.
        bhatt_z_array = np.array([PolarCode.bhatt_z(i, self._N) for i in range(self._N)])

        # Sort all the channels in the ascending order of their Z-parameter since the lower Z-parameter is,
        # the more suitable channel is to transmit the information.
        construction = [x[0] for x in sorted(enumerate(bhatt_z_array), key=lambda tup: tup[1])]

        return construction

    def _dependent_bec_construction(self):
        """
        Implements the BEC method of polar construction. This method was decribed in the original Arikan's paper
        and it is based on how the Bhattacharya Z-parameters evolve during the polarization. Differs from the previous
        method since it initializes the first BEC channel with the probability of erasure equal to QBER instead of 0.5.

        :return: An array of polarized channel indices sorted from the best to the worst for information
        transmission
        """
        # Calculate Z-parameter for each of N polarized channels.
        erasure_prob = self._channel.get_erasure_prob()
        bhatt_z_array = np.array([PolarCode.bhatt_z(i, self._N, erasure_prob) for i in range(self._N)])

        # Sort all the channels in the ascending order of their Z-parameter since the lower Z-parameter is,
        # the more suitable channel is to transmit the information.
        construction = [_[0] for _ in sorted(enumerate(bhatt_z_array), key=lambda tup: tup[1])]

        return construction

    @staticmethod
    def bhatt_z(i, N, init_value=0.5):
        """
        Recursively calculates the value of Bhattacharya Z-parameter for the i-th out of N polarized channel.
        The calculation is based on the relations obtained in the original Arikan's paper. The recursion starts
        from the value for one channel out of one which is equal to 0.5.

        :param i: An integer index of a polarized channel;
        :param N: An integer number of polarized channels;
        :param init_value: A double initial value of the erasure probability of first non-polarized BEC channel;
        :return: Double value of the Bhattacharya Z-parameter for the given channel.
        """
        if i == 0 and N == 1:
            return init_value
        else:
            if i % 2 == 0:
                return 2 * PolarCode.bhatt_z(i / 2, N / 2, init_value) - np.power(PolarCode.bhatt_z(i / 2, N / 2, init_value), 2)
            else:
                return np.power(PolarCode.bhatt_z((i - 1) / 2, N / 2, init_value), 2)

    def get_message_info_bits(self, u_message):
        """
        Returns the information bits from the u message based on the internal code structure.

        :param u_message: An integer array of bits from which the information bits should be obtained;
        :return: An integer array of information bits.
        """
        return u_message[np.array(self._info_bits_positions)][0:self._K_minus_CRC]

    def get_message_frozen_bits(self, u_message):
        """
        Returns the frozen bits from the u message based on the internal code structure.

        :param u_message: An integer array of bits from which the frozen bits should be obtained;
        :return: An integer array of frozen bits.
        """
        return u_message[np.array(self._frozen_bits_positions)]

    def extend_info_bits(self, info_bits, frozen_bits=None):
        """
        Interleaves K informational bits and their CRC with frozen bits such that a u_message is obtained.

        :param info_bits:
        :param frozen_bits:
        :return: u_message — result of extension.
        """
        if len(info_bits) != self._K_minus_CRC:
            print('Unable to encode message of {} info bits instead of {}'.format(len(info_bits), self._K))
            exit(-1)

        # Create empty message of length N.
        u_message = np.zeros(self._N, dtype='uint8')

        # If some special frozen bits values are provided, set them in the u_message.
        if frozen_bits is not None:
            if len(frozen_bits) != self._N - self._K:
                print('Unable to encode message with {} vs. {} frozen bits'.format(len(frozen_bits), self._N - self._K))
                exit(-1)

            u_message[np.array(self._frozen_bits_positions)] = frozen_bits

        # Set the values of information bits in the u_message.
        u_message[np.array(self._info_bits_positions)] = np.concatenate([info_bits, self._calculate_CRC(info_bits)])

        return u_message

    def encode(self, info_bits, frozen_bits=None):
        """
        Encodes K information bits into the N bits of the codeword message by padding
        them with frozen bits and by using the polar transform after that (resembles FFT).

        :param info_bits: An integer array of K information bits;
        :param frozen_bits: An array of bits which should be set as frozen during the encoding (None if they
        are treated all zero as in the original Arikan's paper);
        :return: x_message -- result of encoding.
        """
        u_message = self.extend_info_bits(info_bits, frozen_bits)

        # Apply the polar transform to the u_message.
        x_message = PolarCode.polar_transform(u_message)

        return x_message

    def _calculate_CRC(self, info_bits):
        """

        :param info_bits:
        :return:
        """
        if self._CRC_len == 0:
            return np.asarray([])
        else:
            padded_info_bits = np.concatenate([info_bits, np.zeros(self._CRC_len, dtype='uint8')])

            while len(padded_info_bits[0:self._K_minus_CRC].nonzero()[0]):
                cur_shift = (padded_info_bits != 0).argmax(axis=0)
                padded_info_bits[cur_shift: cur_shift + self._CRC_len + 1] ^= PolarCode.CRC_polynomials[self._CRC_len]

            return padded_info_bits[self._K_minus_CRC:]

    @staticmethod
    def polar_transform(u_message):
        """
        Implements the polar transform on the given message in a recursive way (defined in Arikan's paper).

        :param u_message: An integer array of N bits which are to be transformed;
        :return: x_message -- result of the polar transform.
        """
        u_message = np.array(u_message)

        if len(u_message) == 1:
            x_message = u_message
        else:
            u1u2 = u_message[::2] ^ u_message[1::2]
            u2 = u_message[1::2]

            x_message = np.concatenate([PolarCode.polar_transform(u1u2), PolarCode.polar_transform(u2)])
        return x_message

    def decode(self, message, frozen_bits=None, decoding_method='FSC', list_size=None):
        """
        Calls the preferred method from the dict of methods to construct the code.

        :param message: An integer array of N bits which are to be decoded;
        :param frozen_bits: An array of bits which should be counted as frozen during the decoding (None if they
        are treated all zero as in the original Arikan's paper);
        :param decoding_method: A string defining which of the construction method to use;
        :param list_size:
        :return: An integer array of N decoded bits (u message, not x).
        """
        try:
            if decoding_method == 'SCL':
                return self._scl_decode(message, frozen_bits, list_size)
            else:
                return self._decoding_methods[decoding_method](message, frozen_bits)
        except KeyError as wrong_key:
            print('Unable to decode message: no {} method was provided'.format(wrong_key))
            exit(-1)

    def _slow_sc_decode(self, y_message, frozen_bits=None):
        """
        Implements the Successive Cancellation (SC) decoder described by Arikan. In particular, it just calculates
        the LLR for each polarized channel based on the received bits and previous decoded bits and compares it to zero.
        Decoding is conducted in LLR domain since it is more resilient to float overflows. However, this decoding
        function is slow since it has O(N ^ 2) computational complexity.

        IMPORTANT: This decoding function was written first as proof-of-concept and now it is obsolete. Advanced
        decoding function 'fast_sc_decode' should be used.

        :param y_message: An integer array of N bits which are to be decoded;
        :param frozen_bits: An array of bits which should be counted as frozen during the decoding (None if they
        are treated all zero as in the original Arikan's paper);
        :param list_size:
        :return: An integer array of N decoded bits (u message, not x).
        """
        y_message = np.array(y_message, dtype='uint8')

        u_est = np.zeros(self._N, dtype='uint8')

        # The values of frozen bits are all known
        if frozen_bits is not None:
            if len(frozen_bits) != self._N - self._K:
                print('Unable to decode message with {} vs. {} frozen bits'.format(len(frozen_bits), self._N - self._K))
                exit(-1)

            u_est[np.array(self._frozen_bits_positions)] = frozen_bits

        # For each information bit we should calculate the LLR of corresponding polarized channel.
        for idx in self._info_bits_positions:
            llr = self._slow_llr(idx, self._N, y_message, u_est[:idx])
            u_est[idx] = 0 if llr > 0 else 1

        return u_est

    def _slow_llr(self, i, N, y, u_est):
        """
        Recursively calculates the Log-Likelihood Ratio (LLR) of i-th polarized channel out of N based on given message
        y and previous estimated bits. The recursive formulas follows from the Arikan paper.

        IMPORTANT: Since it calculates LLR recursively for each polarized channel without usage of previously
        calculated values, it has poor computational performance. Advanced LR calculation function 'fast_lr'
        should be used.

        :param i: An integer index of polarized channel;
        :param N: An integer number of polarized channels;
        :param y: An integer array of bits which are used while decoding;
        :param u_est: An integer array of previously decoded bits;
        :return: The double value of LLR for the current polarized channel.
        """
        if len(y) != N:
            print('Unable to calculate LLR: y vector has length of {} instead of {}'.format(len(y), N))
            exit(-1)
        if len(u_est) != i:
            print('Unable to calculate LLR: u_est vector has length of {} instead of {}'.format(len(u_est), i))
            exit(-1)

        # Trivial case of one polarized channel out of one.
        if i == 0 and N == 1:
            llr = self._channel.get_llr(y[i])

        else:
            if i % 2 == 0:
                llr_1 = self._slow_llr(i // 2,
                                       N // 2,
                                      y[:(N // 2)],
                                      (u_est[::2] ^ u_est[1::2])[:(i // 2)])
                llr_2 = self._slow_llr(i // 2,
                                       N // 2,
                                      y[N // 2:],
                                      u_est[1::2][:(i // 2)])

                llr = self._llr_check_node_operation(llr_1, llr_2)
            else:
                llr_1 = self._slow_llr((i - 1) // 2,
                                       N // 2,
                                      y[:(N // 2)],
                                      (u_est[:-1:2] ^ u_est[1:-1:2])[:(i - 1 // 2)])
                llr_2 = self._slow_llr((i - 1) // 2,
                                       N // 2,
                                      y[N // 2:],
                                      u_est[1::2][:((i - 1) // 2)])

                llr = llr_2 + ((-1) ** u_est[-1]) * llr_1

        return np.float128(llr)

    def _fast_sc_decode(self, y_message, frozen_bits=None):
        """
        Implements the Successive Cancellation (SC) decoder described by Arikan. In particular, it calculates
        the LLR for each polarized channel based on the received bits and previous decoded bits and compares it to zero.
        Decoding is conducted in LLR domain since it is more resilient to float overflows. Since this function
        makes use of previously calculated LLR values, it calculates only N * (1 + log(N)) LLR values, thus its
        computational complexity is O(N * (1 + log(N))) and therefore it is quite efficient.

        :param y_message: An integer array of N bits which are to be decoded;
        :param frozen_bits: An array of bits which should be counted as frozen during the decoding (None if they
        are treated all zero as in the original Arikan's paper);
        :param list_size:
        :return: An integer array of N decoded bits (u message, not x).
        """

        u_est = np.full(self._N, -1)

        # An array which shows for each LLR value out of N * (1 + log(N)) whether it was calculated
        is_calc_llr = [False] * self._N * (self._n + 1)

        # An array which stores values for N * (1 + log(N)) LLRs
        llr_array = np.full(self._N * (self._n + 1), 0.0, dtype=np.longfloat)

        for i in range(self._N):
            # Call the function to calculate LLR for i-th out of N polarized channels
            llr = self._fast_llr(i, y_message, u_est[:i], llr_array, is_calc_llr)

            if i in self._frozen_bits_positions:
                u_est[i] = frozen_bits[self._frozen_bits_positions.index(i)] if frozen_bits is not None else 0
            else:
                u_est[i] = 0 if llr > 0 else 1

        return u_est

    def _fast_llr(self, i, y, u_est, llr_array, is_calc_llr):
        """
        Recursively calculates N * (1 + log(N)) values of Log-Likelihood Ratio (LLR) which are required to
        calculate the LLRs for N polarized channels. The basic idea is to split the problem of calculating N LLRs
        to the two problems of calculating N/2 LLRs. In original Arikan's paper these LLRs are placed on a graph,
        but this function <<stretches>> this graph and stores all LLRs in the linear array. That is why such functions
        as "get_problem_i" and  "get_descendants" are used.

        :param i: An integer index of one LLR out of N * (1 + log(N)) to be calculated
        :param y:  An integer array of N bits which are to be decoded;
        :param u_est: An integer array of previously decoded bits;
        :param llr_array: A double array of both already calculated or not LLR values;
        :param is_calc_llr: A boolean array of indicators of LLRs being calculated;
        :return: void (all the calculated LLRs will be placed in the llr_array).
        """
        # If the LLR value is not already calculated, we have to calculate it.
        if not is_calc_llr[i]:
            # We define which to which index in the current subproblem this LLR corresponds.
            problem_i = self._get_problem_i(i)
            # We define the size of the current subproblem based on passed to the function.
            N = len(y)

            if problem_i == 0 and N == 1:
                # For subproblem of size one the calculation of LLR is rather trivial.
                llr_array[i] = self._channel.get_llr(y[0])
            else:
                # Otherwise, we start the recursion

                # First, we check the consistence of the passed arguments
                if len(u_est) != problem_i:
                    print('Unable to calculate LLR: u_est vector has length of {} instead of {}'.format(len(u_est), i))
                    exit(-1)

                # We obtain the indices of the LLRs in the llr_array which should be used to calculate current LLR.
                left_desc, right_desc = self._get_descendants(i)

                # Two descendants LLRs are combined in a different way based on the parity of current subproblem index.
                if (problem_i % 2) == 0:
                    llr_1 = self._fast_llr(left_desc,
                                          y[:(N // 2)],
                                          (u_est[::2] ^ u_est[1::2])[:(problem_i // 2)],
                                           llr_array,
                                           is_calc_llr)
                    llr_2 = self._fast_llr(right_desc,
                                          y[(N // 2):],
                                          u_est[1::2][:(problem_i // 2)],
                                           llr_array,
                                           is_calc_llr)

                    llr_array[i] = self._llr_check_node_operation(llr_1, llr_2)
                else:
                    llr_1 = self._fast_llr(left_desc,
                                          y[:(N // 2)],
                                          (u_est[:-1:2] ^ u_est[1:-1:2])[:(problem_i // 2)],
                                           llr_array,
                                           is_calc_llr)
                    llr_2 = self._fast_llr(right_desc,
                                          y[(N // 2):],
                                          u_est[1::2][:(problem_i // 2)],
                                           llr_array,
                                           is_calc_llr)

                    llr_array[i] = llr_2 + ((-1) ** u_est[-1]) * llr_1

            # After having combined the descending LLRs, we now have this LLR calculated.
            is_calc_llr[i] = True
        return llr_array[i]

    def _get_problem_i(self, i):
        """
        Given the absolute index in the llr_array, it calculates to which index in the subproblem current index relates.

        :param i: An integer index of LLR in the llr_array;
        :return: Integer index of LLR in its related subproblem.
        """
        # First, we find which slice of graph corresponds the provided i.
        slice_idx = i // self._N

        # Then, since the size of subproblem for this index is defined with the number of slice, we can calculate it.
        modulus = 2 ** (self._n - slice_idx)

        # Finally, we return the remainder of i divided by the subproblem size
        return i % modulus

    def _get_descendants(self, i):
        """
        Given the absolute index in the llr_array, it calculates absolute indices of the left and right descendants
        of the current i in the current subproblem.

        :param i: An integer index of LLR in the llr_array;
        :return: Integer indices of the left and right descendants of the current i in the current subproblem.
        """
        # First, we find which slice of graph corresponds the provided i.
        slice_idx = i // self._N

        # Second, we find which index this i has in the related slice.
        slice_i = i - slice_idx * self._N

        # We calculate the size of subproblem for the current slice.
        subproblem_len = 2 ** (self._n - slice_idx)

        # We calculate it which index in the slice the related subproblem starts.
        subproblem_start = (slice_i // subproblem_len) * subproblem_len

        # We calculate which index has i in its related subproblem.
        subproblem_i = i % subproblem_len

        # The left descendant is (subproblem_i / 2) closer to the subproblem starting index in the next slice.
        left_desc = (slice_idx + 1) * self._N + subproblem_start + (subproblem_i // 2)
        # The right descendant is by length of next slice subproblems farther from the left descendant.
        right_desc = left_desc + 2 ** (self._n - slice_idx - 1)

        return left_desc, right_desc

    @staticmethod
    def _llr_check_node_operation(llr_1, llr_2):
        """
        Approximates the check node operation which combines LLRs in accordance with the numerical restrictions
        of the NumPy. Since np.tanh(x) = 1.0 for such x that abs(x) > 22, for all the cases when both arguments
        have absolute value greater than 44, we shall somehow approximate the check node operation function.
        Simple calculations show that the used approximation has the highest error rate of 1% and thus is is quite
        precise.

        :param llr_1: A double value of the first LLR;
        :param llr_2: A double value of the second LLR;
        :return: The exact or approximate double result of the check-node approximation.
        """
        # As it is said in the preface, if both arguments are higher than threshold value, the function should.
        # be approximated.
        if abs(llr_1) > CHECK_NODE_TANH_THRES and abs(llr_2) > CHECK_NODE_TANH_THRES:
            if llr_1 * llr_2 > 0:
                # If both LLRs are of one sign, we return the minimum of their absolute values.
                return min(abs(llr_1), abs(llr_2))
            else:
                # Otherwise, we return an opposite to the minimum of their absolute values.
                return -min(abs(llr_1), abs(llr_2))
        else:
            return 2 * arctanh(tanh(llr_1 / 2, dtype=np.float128) * tanh(llr_2 / 2, dtype=np.float128))

    def _tal_vardy_decode(self, y_message, frozen_bits=None):
        """
        Implements the SC decoder using the notations from the Tal and Vardy paper. This method will be further
        extended to the Tal-Vardy list decoder.

        :param y_message: An integer array of N bits which are to be decoded;
        :param frozen_bits: An array of bits which should be counted as frozen during the decoding (None if they
        are treated all zero as in the original Arikan's paper);
        :return: An integer array of N decoded bits (u message, not x).
        """

        self._p_arr = np.zeros((self._n + 1, self._N, 2), dtype=np.longfloat)
        self._b_arr = np.zeros((self._n + 1, self._N), dtype=np.uint8)

        vec_out_bit_prob = np.vectorize(self._out_bit_prob)
        self._p_arr[0, :, 0] = vec_out_bit_prob(y_message, 0)
        self._p_arr[0, :, 1] = vec_out_bit_prob(y_message, 1)

        for phi in range(self._N):
            self._recursively_calc_p_arr(self._n, phi)
            if phi in self._frozen_bits_positions:
                self._b_arr[self._n, phi] = frozen_bits[self._frozen_bits_positions.index(phi)] if frozen_bits is not None else 0
            else:
                self._b_arr[self._n, phi] = 1 if self._p_arr[self._n, phi, 1] > self._p_arr[self._n, phi, 0] else 0

            if (phi % 2) == 1:
                self._recursively_update_b_arr(self._n, phi)

        return self.polar_transform(self._b_arr[0, :])

    def _recursively_calc_p_arr(self, l, phi):
        if l != 0:
            psi = phi // 2
            arr_idx = self.idx
            if (phi % 2) == 0:
                self._recursively_calc_p_arr(l - 1, psi)

            for br in range(2 ** (self._n - l)):
                if (phi % 2) == 0:
                    self._p_arr[l, arr_idx(l, phi, br), 0] = 0.5 * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br)][0] \
                                                             * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br + 1)][0] \
                                                             + 0.5 * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br)][1] \
                                                             * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br + 1)][1]

                    self._p_arr[l, arr_idx(l, phi, br), 1] = 0.5 * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br)][1] \
                                                             * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br + 1)][0] \
                                                             + 0.5 * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br)][0] \
                                                             * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br + 1)][1]
                else:
                    u = self._b_arr[l, arr_idx(l, phi - 1, br)]

                    self._p_arr[l, arr_idx(l, phi, br)][0] = 0.5 * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br)][u ^ 0] \
                                                             * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br + 1)][0]

                    self._p_arr[l, arr_idx(l, phi, br)][1] = 0.5 * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br)][u ^ 1] \
                                                             * self._p_arr[l - 1, arr_idx(l - 1, psi, 2 * br + 1)][1]

    @staticmethod
    def idx(l, phi, br):
        return phi + (2 ** l) * br

    def _recursively_update_b_arr(self, l, phi):
        if (phi % 2) == 1:
            psi = phi // 2
            arr_idx = self.idx

            for br in range(2 ** (self._n - l)):
                self._b_arr[l - 1, arr_idx(l - 1, psi, 2 * br)] = self._b_arr[l, arr_idx(l, phi - 1, br)] \
                                                                  ^ self._b_arr[l, arr_idx(l, phi, br)]

                self._b_arr[l - 1, arr_idx(l - 1, psi, 2 * br + 1)] = self._b_arr[l, arr_idx(l, phi, br)]

            if (psi % 2) == 1:
                self._recursively_update_b_arr(l - 1, psi)

    def _out_bit_prob(self, output_bit, input_bit):
        return self._channel.get_ber() if output_bit ^ input_bit else (1.0 - self._channel.get_ber())

    def _initialize_data_structures(self, L):
        """

        :param L:
        :return:
        """
        self._inactive_path_indices = []
        self._active_path = None
        self._array_pointer_p = [[] for _ in range(self._n + 1)]
        self._array_pointer_c = [[] for _ in range(self._n + 1)]
        self._path_index_to_array_index = None
        self._inactive_array_indices = [[] for _ in range(self._n + 1)]
        self._array_reference_count = None

        self._path_index_to_array_index = np.zeros((self._n + 1, L), dtype=np.uint8)
        self._array_reference_count = np.zeros((self._n + 1, L), dtype=np.uint8)

        for lam in range(self._n + 1):
            for s in range(L):
                self._array_pointer_p[lam].append(np.full(((2 ** (self._n - lam)), 2), -1.0))
                self._array_pointer_c[lam].append(np.zeros(((2 ** (self._n - lam)), 2), dtype=np.uint8))

                self._inactive_array_indices[lam].append(s)

        self._active_path = np.zeros(L, dtype=bool)
        for l in range(L):
            self._inactive_path_indices.append(l)

    def _assign_initial_path(self):
        """

        :return: Integer index of initial path
        """
        l = self._inactive_path_indices.pop()
        self._active_path[l] = True

        for lam in range(self._n + 1):
            s = self._inactive_array_indices[lam].pop()
            self._path_index_to_array_index[lam, l] = s
            self._array_reference_count[lam, s] = 1

        return l

    def _clone_path(self, l):
        """

        :param l: An integer index of path to clone
        :return: Integer index of copy
        """
        l_dash = self._inactive_path_indices.pop()
        self._active_path[l_dash] = True

        for lam in range(self._n + 1):
            s = self._path_index_to_array_index[lam, l]
            self._path_index_to_array_index[lam, l_dash] = s
            self._array_reference_count[lam, s] += 1

        return l_dash

    def _kill_path(self, l):
        """

        :param l: An integer index of path to kill
        :return:
        """
        self._active_path[l] = False
        self._inactive_path_indices.append(l)
        for lam in range(self._n + 1):
            s = self._path_index_to_array_index[lam, l]
            self._array_reference_count[lam, s] -= 1

            if self._array_reference_count[lam, s] == 0:
                self._inactive_array_indices[lam].append(s)

    def _get_array_pointer_p(self, lam, l):
        """

        :param lam: An integer number of layer
        :param l: An integer path index
        :return: Reference to the corresponding probability pair array
        """

        s = self._path_index_to_array_index[lam, l]
        if self._array_reference_count[lam, s] == 1:
            s_dash = s
        else:
            s_dash = self._inactive_array_indices[lam].pop()
            self._array_pointer_p[lam][s_dash] = np.copy(self._array_pointer_p[lam][s])
            self._array_pointer_c[lam][s_dash] = np.copy(self._array_pointer_c[lam][s])
            self._array_reference_count[lam, s] -= 1
            self._array_reference_count[lam, s_dash] = 1
            self._path_index_to_array_index[lam, l] = s_dash

        return self._array_pointer_p[lam][s_dash]

    def _get_array_pointer_c(self, lam, l):
        """

        :param lam: An integer number of layer
        :param l: An integer path index
        :return: Reference to the corresponding bit pair array
        """

        s = self._path_index_to_array_index[lam, l]
        if self._array_reference_count[lam, s] == 1:
            s_dash = s
        else:
            s_dash = self._inactive_array_indices[lam].pop()
            self._array_pointer_c[lam][s_dash] = np.copy(self._array_pointer_c[lam][s])
            self._array_pointer_p[lam][s_dash] = np.copy(self._array_pointer_p[lam][s])
            self._array_reference_count[lam, s] -= 1
            self._array_reference_count[lam, s_dash] = 1
            self._path_index_to_array_index[lam, l] = s_dash

        return self._array_pointer_c[lam][s_dash]

    def _recursively_calc_p(self, lam, phi, L):
        """

        :param lam: An integer index of current layer
        :param phi: An integer index of current phase
        :param L: An integer size of decoding list
        :return: Void
        """

        if lam == 0:
            return
        psi = phi // 2

        if (phi % 2) == 0:
            self._recursively_calc_p(lam - 1, psi, L)

        sgm = 0.0
        for l in range(L):
            if self._active_path[l]:
                p_curr = self._get_array_pointer_p(lam, l)
                p_prev = self._get_array_pointer_p(lam - 1, l)
                c_curr = self._get_array_pointer_c(lam, l)

                for br in range(2 ** (self._n - lam)):
                    if (phi % 2) == 0:
                        p_curr[br, 0] = 0.5 * p_prev[2 * br, 0] * p_prev[2 * br + 1, 0] \
                                        + 0.5 * p_prev[2 * br, 1] * p_prev[2 * br + 1, 1]
                        sgm = max(sgm, p_curr[br, 0])

                        p_curr[br, 1] = 0.5 * p_prev[2 * br, 1] * p_prev[2 * br + 1, 0] \
                                        + 0.5 * p_prev[2 * br, 0] * p_prev[2 * br + 1, 1]
                        sgm = max(sgm, p_curr[br, 1])
                    else:
                        u = c_curr[br, 0]
                        p_curr[br, 0] = 0.5 * p_prev[2 * br, u] * p_prev[2 * br + 1, 0]
                        sgm = max(sgm, p_curr[br, 0])

                        p_curr[br, 1] = 0.5 * p_prev[2 * br, u ^ 1] * p_prev[2 * br + 1, 1]
                        sgm = max(sgm, p_curr[br, 1])

        for l in range(L):
            if self._active_path[l]:
                p_curr = self._get_array_pointer_p(lam, l)
                for br in range(2 ** (self._n - lam)):
                    p_curr[br, 0] /= sgm
                    p_curr[br, 1] /= sgm

    def _recursively_update_c(self, lam, phi, L):
        """

        :param lam: An integer index of current layer
        :param phi: An integer index of current phase
        :param L: An integer size of decoding list
        :return: Void
        """

        if (phi % 2) == 1:
            psi = phi // 2

            for l in range(L):
                if self._active_path[l]:
                    c_curr = self._get_array_pointer_c(lam, l)
                    c_prev = self._get_array_pointer_c(lam - 1, l)

                    for br in range(2 ** (self._n - lam)):
                        c_prev[2 * br][psi % 2] = c_curr[br][0] ^ c_curr[br][1]
                        c_prev[2 * br + 1][psi % 2] = c_curr[br][1]

            if psi % 2 == 1:
                self._recursively_update_c(lam - 1, psi, L)

    def _continue_paths_unfrozen_bit(self, phi, L):
        """

        :param phi: An integer index of current phase
        :param L: An integer size of decoding list
        :return: Void
        """
        prob_forks = np.zeros(2 * L, dtype=float)
        i = 0
        for l in range(L):
            if self._active_path[l]:
                p_curr = self._get_array_pointer_p(self._n, l)
                prob_forks[l] = p_curr[0, 0]
                prob_forks[l + L] = p_curr[0, 1]
                i += 1
            else:
                prob_forks[l] = -1
                prob_forks[l + L] = -1

        sorted_prob_forks = sorted(enumerate(prob_forks), key=lambda tup: -tup[1])
        rho = min(2 * i, L)
        cont_forks = np.zeros((L, 2), dtype=bool)
        for i in range(rho):
            cont_forks[sorted_prob_forks[i][0] % L, sorted_prob_forks[i][0] // L] = True
        for l in range(L):
            if self._active_path[l]:
                if not cont_forks[l][0] and not cont_forks[l][1]:
                    self._kill_path(l)

        for l in range(L):
            if not cont_forks[l][0] and not cont_forks[l][1]:
                continue
            c_curr = self._get_array_pointer_c(self._n, l)
            if cont_forks[l][0] and cont_forks[l][1]:
                c_curr[0][phi % 2] = 0
                l_dash = self._clone_path(l)
                c_curr = self._get_array_pointer_c(self._n, l_dash)
                c_curr[0][phi % 2] = 1
            elif cont_forks[l][0]:
                c_curr[0][phi % 2] = 0
            else:
                c_curr[0][phi % 2] = 1

    def _scl_decode(self, y_message, frozen_bits=None, L=32):
        """
        Implements the SC decoder using the notations from the Tal and Vardy paper. This method will be further
        extended to the Tal-Vardy list decoder.

        :param y_message: An integer array of N bits which are to be decoded;
        :param frozen_bits: An array of bits which should be counted as frozen during the decoding (None if they
        are treated all zero as in the original Arikan's paper);
        :param L: An integer size of decoding list;
        :return: An integer array of N decoded bits (u message, not x).
        """
        self._initialize_data_structures(L)
        l = self._assign_initial_path()
        p_zero = self._get_array_pointer_p(0, l)

        for br in range(self._N):
            p_zero[br, 0] = self._out_bit_prob(y_message[br], 0)
            p_zero[br, 1] = self._out_bit_prob(y_message[br], 1)

        for phi in range(self._N):
            self._recursively_calc_p(self._n, phi, L)

            if phi in self._frozen_bits_positions:
                for l in range(L):
                    if self._active_path[l]:
                        c_curr = self._get_array_pointer_c(self._n, l)
                        c_curr[0, phi % 2] = frozen_bits[self._frozen_bits_positions.index(phi)] if frozen_bits is not None else 0
            else:
                self._continue_paths_unfrozen_bit(phi, L)

            if (phi % 2) == 1:
                self._recursively_update_c(self._n, phi, L)

        l_dash = 0
        p_dash = 0
        decoding_list = []
        is_CRC_present = False

        for l in range(L):
            if self._active_path[l]:
                path_output = self.polar_transform(self._get_array_pointer_c(0, l)[:, 0])
                path_output_info_bits = path_output[list(self._info_bits_positions)]

                if np.array_equal(self._calculate_CRC(path_output_info_bits[:self._K_minus_CRC]),
                                  path_output_info_bits[self._K_minus_CRC:]):
                    is_CRC_present = True
                    c_curr = self._get_array_pointer_c(self._n, l)
                    p_curr = self._get_array_pointer_p(self._n, l)
                    decoding_list.append(path_output)
                    if p_dash < p_curr[0, c_curr[0, 1]]:
                        l_dash = l
                        p_dash = p_curr[0, c_curr[0, 1]]

        if not is_CRC_present:
            return None

        c_zero = self._get_array_pointer_c(0, l_dash)
        return self.polar_transform(c_zero[:, 0])
