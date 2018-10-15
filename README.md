# Polar codes
This repository is devoted to the Python implementation of Polar codes.

## Motivation
When I faced the need to use Polar codes in my Master's thesis, I hadn't found any open-source Python implementation.
That was quite a surprise since I just wanted to download a library and start my research.  
Hence, it took me to implement Polar codes myself.
Now, I want to help anybody who will further have a need in Python implementation and I start this project.
This is my first open-source project, I don't have too much experience, but hope it will go well. 
Comments, remarks and pull requests are welcome!

## What is inside

Currently this repository is work-in-progress. However, perfectly it should contain the following:

- [ ] **Polar code** — a principal class which is responsible for encoding and decoding.
 It has the following public methods.
    - [ ] **construct** — Polar codes are channel-dependent and thus they can be constructed in many ways.
    This class will have the following construction methods.
        - [ ] **PW** — a construction method which arranges channel according to their polar weights (PW stands for **P***olar **W**eights).
        - [ ] **IBEC** — a construction method which arranges polar transformed Binary Erasure Channels (BEC) according to their Bhattacharya Z-parameter.
         The erasure probability of BECs is set to `0.5` (IBEC stands for **I**ndependent **B**inary **E**rasure **C**hannel construction).
         - [ ] **DBEC** — a construction method which arranges polar transformed Binary Erasure Channels (BEC) according to their Bhattacharya Z-parameter.
         The erasure probability of BECs is chosen such that initial channel has the same capacity as the virtual BEC (DBEC stands for **D**ependent **B**inary **E**rasure **C**hannel construction).
    - [ ] **encode** — implements encoding algorithm which complements informational bits with frozen ones and applies a polar transform to the resulting codeword.
    - [ ] **decode** — 
        - [ ] **Successive Cancellation (SC)** —
        - [ ] **Tal-Vardy (TVD)** —
        - [ ] **List decoder** — 
        - [ ] **List decoder + CRC** — 
- [ ] **Channel** — Polar codes are channel-specific codes and thus each instance.
- [ ] **BscChannel** — 
- [ ] **BpskAwgnChannel** — 

All algorithms are based on papers, references to them will be added further.