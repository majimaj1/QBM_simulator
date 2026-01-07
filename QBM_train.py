#这是一个用于训练QBM的程序
import numpy as np
import matplotlib as plt
N=10
M=8
Iteration=20
Pvdata=np.load("Pvdata.npy")
points=np.load("points.npy")
sigma_z=np.array([[1,0],
                  [0,-1]])
sigma_x=np.array([[0,1],
                  [1,0]])
b=np.random.uniform(-1,1,size=N)
w=np.random.uniform(-1,1,size=(N,N))
