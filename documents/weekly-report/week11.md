# Week 11 (From Oct. 2 to Oct. 6)
**Summary of the week**  
1. Literature review on modeling time-series sensor error. Plan on the next step experiement.
2. Select the autoregressive input-output hidden markov model (AIO-HMM) as the perception error model. 
## Weekly outcome
- Mainly focus on the three papers:  
1. [On the Simulation of Perception Erors in Autonomous Vehicles](https://arxiv.org/pdf/2302.11919.pdf)
2. [Statistical sensor mod-
elling for autonomous driving using autoregressive input-output hmms](https://ieeexplore.ieee.org/document/8569592)
3. [Car that Knows Before You Do:
Anticipating Maneuvers via Learning Temporal Driving Models](https://arxiv.org/pdf/1504.02789.pdf)

- The first paper use the conditional autoregressive model to model the perception error in the surrounding zone, which is partition as the polar corrdinate.
- The second paper use the AIO-HMMs to model the perception error, given the nature of the AIOHMM relax the conditional indepenece assuption between the outputs, which is suitable to model the highly autoregressive data for our case.
- The method used in the second paper is inspired from the thrid paper.
- However, currently, there is not Python implementation of the AIO-HMMs. So have to implement the metdho by myself.
- Summarize how I will modeling the perception error to Andrea, not hear from his reply for a weeks. 
## Next week task
- Implement the AIO-HMM in Python.