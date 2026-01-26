import cupy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
N=10
M=8
Iteration=20
Pvdata=np.load("Pvdata.npy")
points=np.load("points.npy")
sigma_z=np.array([[1,0],
                  [0,-1]])
sigma_x=np.array([[0,1],
                  [1,0]])

I=np.identity(2)
b=np.load("b.npy")
w=np.load("w.npy")
H=np.zeros((np.power(2,N).item(),np.power(2,N).item()))
for i in range(0,N):
    for j in range(i,N):
        w[j,i]=w[i,j]

class QBM:
    def __init__(self,N,M,Iteration, points, Pv, b ,w,gamma=1):
        self.N=N
        self.M=M
        self.Iteration=Iteration
        self.gamma=gamma
        self.b=b
        self.w=w
        self.x=self.wb_to_f(self.gamma,self.b,self.w)
        self.x_operator=self.matrix_list()
        self.ops=[np.identity(2) for _ in range(self.N)]
        self.binary_list=self.binary_all()
        self.points=points
        self.Pv=Pv
        self.binary_points=[self.ret_binary(k) for k in points]
        self.lam=[self.lambda_v(num) for num in self.binary_points]
        self.state=[self.state_v(num) for num in self.binary_points]
        self.loss_list=[]
    def state_v(self,num):
        T = np.array([1, 0])
        F = np.array([0, 1])
        if num[0] == 1:
            lam = T
        if num[0] == -1:
            lam = F
        for k in num[1:]:
            if k == 1:
                lam = np.kron(lam, T)
            if k == -1:
                lam = np.kron(lam, F)
        return lam
    def lambda_v(self,num):
        T=np.array([[1,0],
                  [0,0]])
        F=np.array([[0,0],
                  [0,1]])
        if num[0]==1:
            lam=T
        if num[0]==-1:
            lam=F
        for k in num[1:]:
            if k ==1:
                lam=np.kron(lam,T)
            if k==-1:
                lam = np.kron(lam, F)
        return lam
    def wb_to_f(self,gamma,b,w):
        N=self.N
        f = np.hstack([b, w.reshape(-1), gamma])
        return f
    def f_to_b_w_gamma(self,f):
        N = self.N
        b = f[0:N]
        w = f[N:N + N * N].reshape(N, N)
        gamma =  float(f[N + N*N])
        print(w)
        return b,w,gamma
    def matrix_list(self):
        N=self.N
        d=np.power(2,10).item()
        gamma_list=[None]*N
        gamma_list_re=np.empty((N,d,d))
        b_list=[None]*N
        b_list_re=np.empty((N,d,d))
        w_list=[[None]*N for _ in range(0,N)]
        w_list_re=np.empty((N*N,d,d))
        f_list=np.empty((2*N+N*N,d,d))
        sigma_z = np.array([[1, 0],
                            [0, -1]])
        sigma_x = np.array([[0, 1],
                            [1, 0]])
        I = np.identity(2)
        grid= [[None]*N for _ in range(0,N)]

        for i in range(0,N):
            ops_x=[np.identity(2) for _ in range(self.N)]
            ops_z=[np.identity(2) for _ in range(self.N)]
            ops_x[i]=sigma_x
            ops_z[i]=sigma_z
            res_x=ops_x[0]
            res_z=ops_z[0]
            for xi in ops_x[1:]:
                res_x=np.kron(res_x,xi)
            for zi in ops_z[1:]:
                res_z=np.kron(res_z,zi)
            gamma_list[i]=res_x
            gamma_list_re[i]=res_x
            b_list[i]=res_z
            b_list_re[i]=res_z
            for j in range(0,N):
                ops_z2=[np.identity(2) for _ in range(self.N)]
                ops_z2[j]=sigma_z
                res_z2=ops_z2[0]
                for z2i in ops_z2[1:]:
                    res_z2=np.kron(res_z2,z2i)
                w_list[i][j]=res_z@res_z2
                w_list_re[i*N+j]=res_z@res_z2
            f_list[0:N]=b_list_re[:]
            f_list[N:N*N+N]=w_list_re[:]
            f_list[N*N+N:]=gamma_list_re[:]
        return f_list
    def cal_H(self,x):
        """
        计算H的值
        :param gamma:
        :param b:
        :param w:
        :return:
        """
        d=np.power(2,10).item()
        x_operator=np.empty((N+N*N+1,d,d))
        x_operator[0:N+N*N]=self.x_operator[0:N+N*N]
        x_operator[N+N*N]=np.sum(self.x_operator[N+N*N:2*N+N*N])
        H=np.tensordot(x,x_operator,(0,0))

        return H
    def ret_binary(self,num):
        stor = num
        ejz = np.full(shape=10, fill_value=-1)
        for i in range(9, -1, -1):
            if stor >= np.power(2, i).item():
                ejz[i] = 1
                stor = stor - np.power(2, i).item()
        return ejz
    def binary_all(self):
        all=np.zeros(shape=(10,1024))
        for i in range(0,1024):
            all[:,i]=self.ret_binary(i)
        return all
    def cal_trHv(self,state,x,operator,H):
        trace=0
        aver=state.T@H@state
        grad=state.T@operator@state
        trace+=np.exp(aver)*grad
        print(aver)
        return trace

    def cal_average(self,x,operator,H):
        aver=0
        P=np.zeros(np.power(2,self.N).item())
        eigvals,U=np.linalg.eigh(H)
        l_min=np.min(eigvals)
        scales=eigvals-l_min
        U_dagger=np.conjugate(U).T
        operator_trans=U_dagger@operator@U
        for i in range(0,1024):
            aver+=operator_trans[i,i]*np.exp(-scales[i])

        aver*=np.exp(-l_min)
        return aver
    def intergrade_aver(self,x,operator1,operator2,H):
        aver=0
        diagnal_H=np.identity(np.power(2, self.N).item())
        eigvals, U = np.linalg.eigh(H)
        for i,k in enumerate(eigvals):
            diagnal_H[i,i]=k
        U_dagger = np.conjugate(U).T
        operator1_trans = U_dagger @ operator1 @ U
        operator2_trans = U_dagger @ operator2 @ U
        def intergrd(tau):
            first_e=np.exp(-tau*diagnal_H)
            second_e=np.exp((tau-1)*diagnal_H)
            eff_matrix=operator1_trans@first_e@operator2_trans@operator2_trans
            trace=0
            for i in range(0,1024):
                trace+=eff_matrix[i,i]
            return trace
        aver,_=quad(intergrd,0,1)
        return aver

    def lossfunction(self,x):
        L=0
        x=np.asarray(x)
        I=np.identity(1024)
        H = self.cal_H(x)
        Z=self.cal_average(x,I,H)
        for i,k in enumerate(self.Pv):
            L+=-(k*np.log(self.cal_trHv(self.state[i],x,I,H)/Z))
        return float(L.get())
    def grad_loss(self,x):
        x=np.asarray(x)
        grad=0
        cc=0
        grad_f=np.zeros(np.size(x))
        I = np.identity(1024)
        H = self.cal_H(x)
        Z = self.cal_average(x, I,H)
        x_list=self.matrix_list()
        for n, theta in enumerate(x_list[0:N+N*N]):
            grad=0
            print("start intergrate")
            trall=self.intergrade_aver(x,I,theta,H)
            print(trall)
            for i, k in enumerate(self.Pv):
                grad += k * (self.cal_trHv(self.state[i],x,theta,H)/self.cal_trHv(self.state[i],x,I,H)-trall/Z)

            print(grad)
            grad_f[n]=grad

        grad=0
        x_sum=np.sum(x_list[N+N*N:(2+N)*N])
        trxall=self.intergrade_aver(x, I, x_sum,H)
        for v, k in enumerate(self.Pv):
            grad += k * (self.cal_trHv(self.state[v],x, x_sum,H) / self.cal_trHv(self.state[v],x,I,H) - trxall / Z)
        grad_f[-1]=grad
        return grad_f.get()
    def callback(self,xk):
        self.loss_list.append(self.lossfunction(xk))
    def BFGS(self):
        x=self.x.copy().get()
        grad=self.grad_loss
        loss=self.lossfunction
        callback=self.callback
        res= minimize(loss,x,method="BFGS",jac=grad,callback=callback,options={'maxiter':25,"disp":True})
        return res
    def pl(self):
        iteration=np.arange(1,26,1)
        plt.plot(iteration,self.loss_list)
        plt.show()
b=np.full(shape=N,fill_value=0.01)
w=np.full(shape=(N,N),fill_value=0.01)
QBM_b=QBM(N,M,Iteration,points,Pvdata,b,w)
QBM_b.BFGS()
QBM_b.pl()