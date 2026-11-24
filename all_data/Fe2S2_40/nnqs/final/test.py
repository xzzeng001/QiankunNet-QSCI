import numpy as np

aa=np.load('fe2s2_cdfci_0.0_107437784states.npz')

for ii in aa:
    print('ii',aa['ci_states'])
