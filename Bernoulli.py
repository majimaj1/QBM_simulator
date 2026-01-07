import numpy as np
M=8
N=10
epoch_train=1000
peaks = np.random.randint(0, 1023, M, dtype=int)
data = np.random.randint(0, 1024, epoch_train, dtype=int)
HDist=np.zeros((M,epoch_train),dtype=int)
train_data=np.zeros((2,epoch_train),dtype=float)
p=0.9
for i in range(0,M):
    diff=np.bitwise_xor(data,peaks[i])
    HDist[i,:]=np.bitwise_count(diff)
Pvk= (1/M)*np.pow(p,N-HDist)*np.pow(1-p,HDist)
Pvdata=np.sum(Pvk,axis=0)
np.save("points.npy",data)
np.save("Pvdata.npy",Pvdata)
