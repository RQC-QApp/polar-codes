# Polar codes
This repository is devoted to the Python implementation of Polar codes.

## Motivation
When I faced the need to use Polar codes in my Master's thesis, I hadn't found any open-source Python implementation (but now I there are some of them).
That was quite a surprise since I just wanted to download a library and start my research.  
Hence, it took me to implement Polar codes myself.
Now, I want to help anybody who will further have a need in Python implementation and I start this project.
This is my first open-source project, I don't have too much experience, but hope it will go well. 
Comments, remarks and pull requests are welcome!

## What is inside

Currently this repository is work-in-progress. 
In particular, the following funcionalty is implemented, but comments are still are to be written and some features shall be tested.

- [x] **Polar code** — a principal class which is responsible for encoding and decoding.
 It has the following public methods.
    - [x] **construct** — Polar codes are channel-dependent and thus they can be constructed in many ways.
    This class will have the following construction methods.
        - [x] **PW** — a construction method which arranges virtual channels according to their polar weights (PW stands for **P**olar **W**eights).
        - [x] **IBEC** — a construction method which arranges polar transformed Binary Erasure Channels (BEC) according to their Bhattacharya Z-parameter.
         The erasure probability of BECs is set to `0.5` (IBEC stands for **I**ndependent **B**inary **E**rasure **C**hannel construction).
         - [x] **DBEC** — a construction method which arranges polar transformed Binary Erasure Channels (BEC) according to their Bhattacharya Z-parameter.
         The erasure probability of BECs is chosen such that initial channel has the same capacity as the virtual BEC (DBEC stands for **D**ependent **B**inary **E**rasure **C**hannel construction).
    - [x] **encode** — implements encoding algorithm which complements informational bits with frozen ones and applies a polar transform to the resulting codeword.
    - [x] **decode** — up to now there are several decoders proposed for Polar codes which have different computational efficiency and error-correction ability.
    This method simply call a specified decoder
        - [x] **Slow Successive Cancellation (SSC)** — a simple successive-cancellation decoder initially proposed by Arikan. 
        It employs no optimization and thus has O(N^2) complexity.
        - [x] **Fast Successive Cancellation (FSC)** — an advanced version of successive cancellation decoder, also proposed by Arikan, has O(NlogN) complexity.
        - [x] **Tal-Vardy (TVD)** — an implementation of SC decoder from Tal-Vardy paper. Was added to get used to Tal-Vardy notion. 
        - [x] **List decoder + CRC (SCL)** — a list version of SC decoder proposed by Tal-Vardy. Its efficiency increases with list size. 
        Moreover, if a CRC was added to an initial codeword it may be used just like in Tal-Vardy's paper.
        **Thorough testing is required**!
- [x] **Channel** — Polar codes are channel-specific codes and I decided to pass to each instance of Polar code an instance of underlying channel. 
A "Channel" is an abstract class forcing all its derivatives to implement several methods.
- [x] **BscChannel** — an implementation of Channel methods for case of Binary Symmetric Channel (BSC).
- [x] **BpskAwgnChannel** — an implemetation of Chanel methods for case of AWGN channel combined with BPSK.

All algorithms are based on papers, references to them will be added further.

## How to use

The package itself is located in "polar_codes" directory. 
Thus, you may copy it and then import a PolarCode class as follows.
    
    from polar_codes.polar_code import PolarCode
    
Probaly, it is not as pretty as one may want and I will fix it further.