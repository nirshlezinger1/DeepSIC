# DeepSIC
A deep learning based soft interference cancellation symbol detector, based on the paper:

N. Shlezinger, R. Fu, and Y. C. Eldar. 

“**ViterbiNet: A deep learning based Viterbi algorithm for symbol detection**”. 

*arXiv preprint*, arXiv:2002.03214, 2020.


## Repository content
The implementation of DeepSIC consists of two functions:

-  GetDeepSICNet - generate and train DeepSIC MIMO detector.
  
-  s_fDetDeepSIC - use trained model to detect symbols, returns BER.
  
A code example for evaluating ViterbiNet can be found in the script DeepSIC_Test1.m

This code requires Matlab with deep learning toolbox.
