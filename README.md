# DeepSIC
A deep learning based soft interference cancellation symbol detector, based on the paper:

N. Shlezinger, R. Fu, and Y. C. Eldar. 

“**DeepSIC: Deep Soft Interference Cancellation for Multiuser MIMO Detection**”. 

*arXiv preprint*, arXiv:2002.03214, 2020.


## Repository content
The implementation of DeepSIC consists of two functions:

-  GetDeepSICNet - generate and train DeepSIC MIMO detector. Training is carried out in a sequential manner (see **Sequential Training** in the above reference).
  
-  s_fDetDeepSIC - use trained model to detect symbols, returns BER.
  
A code example for evaluating ViterbiNet can be found in the script DeepSIC_Test1.m

This code requires Matlab with deep learning toolbox.
