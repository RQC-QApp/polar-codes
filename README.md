# Polar codes
This repository is devoted to the Python implementation of polar codes.

## Motivation
When I faced the need to use polar codes in my Master's thesis, I hadn't found any open-source Python implementations of Polar codes.
That was quite a surprise since I just wanted to download a library and start my research.  
Hence, it took me to implement Polar codes myself.
Now, I want to help anybody who will further have a need in Python implementation and I start this project.
This is my first open-source project, I don't have too much experience, but hope it will go well. 
Comments, remarks and pull requests are welcome!

## What is inside

Currently this repository is work-in-progress. However, perfectly it should contain the following:

**Classes**
- [ ] *Polar code* — a principal class which will have the most important `encode` and `decode` methods.
    - [ ] construct — 
        - [ ] 
    - [ ] encode — imple encoding algorithm which complements informational bits with frozen ones and applies a polar transform to the resulting codeword.
    - [ ] decode — 
        - [ ] Successive Cancellation (SC) —
        - [ ] Tal-Vardy (TVD) —
        - [ ] List decoder — 
        - [ ] List decoder + CRC — 
- [ ] *Channel* — Polar codes are channel-specific codes and thus each instance.
- [ ] *BscChannel* — 
- [ ] *BpskAwgnChannel* — 
