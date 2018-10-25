import numpy as np

from polar_codes.polar_code import PolarCode
from polar_codes.channels.bpsk_awgn_channel import BpskAwgnChannel


def main():
    channel = BpskAwgnChannel(-1.0)

    n = 8
    K = 115
    code = PolarCode(n=n, K=K, construction_method='PW', channel=channel, CRC_len=8)

    u_message = np.asarray([0 if np.random.random_sample() > 0.5 else 1 for _ in range(0, K)], dtype='uint8')
    print('In message = {}'.format(u_message))

    x_message = code.encode(u_message)
    print('Message sent to a channel = {}'.format(x_message))

    to_message = channel.modulate(x_message)
    from_message = channel.transmit(to_message)
    y_message = channel.demodulate(from_message)

    print('Message to decode: {}'.format(y_message))

    ssc_u_est_message = code.decode(y_message, decoding_method='SSC')
    fsc_u_est_message = code.decode(y_message, decoding_method='FSC')
    tvd_u_est_message = code.decode(y_message, decoding_method='TVD')
    scl_u_est_message = code.decode(y_message, decoding_method='SCL', list_size=32)

    print('SSC out message = {}, equal to in = {}'.format(ssc_u_est_message,
                                                          np.array_equal(u_message, ssc_u_est_message)))
    print('FSC out message = {}, equal to in = {}'.format(fsc_u_est_message,
                                                          np.array_equal(u_message, fsc_u_est_message)))
    print('TVD out message = {}, equal to in = {}'.format(tvd_u_est_message,
                                                          np.array_equal(u_message, tvd_u_est_message)))
    print('SCL out message = {}, equal to in = {}'.format(scl_u_est_message,
                                                          np.array_equal(u_message, scl_u_est_message)))


if __name__ == '__main__':
    main()