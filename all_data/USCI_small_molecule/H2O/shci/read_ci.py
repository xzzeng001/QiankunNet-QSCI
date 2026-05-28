import numpy as np

# read the ci coefficients
f=np.loadtxt('ci.txt',dtype=np.str_)

# determinate the number of orbitals and determinant
norb=len(f[0])-2
ndet=len(f[:,0])

#psi_str=''
for i in range(ndet):
    psi_str=''
    for j in range(norb):
        sk=f[i,j+2]
        if sk.strip() == '2':
            psi_str += '11'
        elif sk.strip() == '0':
            psi_str += '00'
        elif sk.strip() == 'a':
            psi_str += '10'
        elif sk.strip() == 'b':
            psi_str += '01'
        else:
            print('read error string!!!!')
            exit(0)

    rr=f[i,1]
#    print('rr and type(rr): ',rr,type(rr))
    with open('ci_and_coeff.txt','a') as ff:
        np.savetxt(ff,[[i, rr, psi_str]],fmt='%s    %s    %s')

ci_probs=np.zeros([ndet],dtype=complex)
ci_states=np.zeros([ndet,2*norb],dtype=int)
for i in range(ndet):
    ii=0
    for j in range(norb):
        sk=f[i,j+2]
        if sk.strip() == '2':
            ci_states[i][ii] = 1
            ci_states[i][ii+1] = 1
        elif sk.strip() == '0':
            ci_states[i][ii] = -1
            ci_states[i][ii+1] = -1
        elif sk.strip() == 'a':
            ci_states[i][ii] = 1
            ci_states[i][ii+1] = -1
        elif sk.strip() == 'b':
            ci_states[i][ii] = -1
            ci_states[i][ii+1] = 1
        else:
            print('read error string!!!!')
            exit(0)

        ii+=2

    rr=f[i,1]
    ci_probs[i]=complex(rr)

#    print('ci_states and ci_probs: ',ci_states[i],ci_probs[i])

np.savez('ham_shci.npz',ci_probs=ci_probs,ci_states=ci_states)


