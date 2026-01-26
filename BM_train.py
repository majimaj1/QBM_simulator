import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
N=10
M=8
Iteration=20
Pvdata=np.load("Pvdata.npy")
points=np.load("points.npy")
b=np.load("b.npy")
w=np.load("w.npy")
Ez = np.zeros(np.pow(2, N))
P = np.zeros(np.pow(2, N))
class Bm:
    """
    定义BM类
    """
    def __init__(self, b, w, Pvdata, datalist, iteration, eta=np.power(10.0,-3)):
        """
        :param b: 电子本身自旋的能量系数（随机初始化【-1,1】）
        :param w: 两个电子自旋的耦合能量系数（随机初始化【-1，1】）
        :param Ez: 状态的能量(目前全是0)
        :param P: 状态的概率（目前全是零）
        :param eta: 学习率
        :param loss: 记录loss function变化
        :param binary_datalist: 将traindata的态转化为二进制表示
        """
        self.b = b
        self.w = w
        self.Ez, self.P = self.cal_Ez_P(self.b,self.w)
        self.Pvdata = Pvdata
        self.datalist = datalist
        self.iteration = iteration
        self.eta=eta
        self.loss=[]
        self.binary_datalist=self.binary_num(self.datalist)
        self.binary_all=self.binary_num(np.arange(np.pow(2,N)))
        self.eta_cos=eta
        self.H=np.identity(int(N+N*(N+1)/2))
        self.yk=np.zeros(int(N+N*(N+1)/2))
        self.sk=np.zeros(int(N+N*(N+1)/2))
    def Pm(self,P):
        Pmod = np.zeros(len(self.Pvdata))
        for i, k in enumerate(self.datalist):
            Pmod[i] = P[k]
        return Pmod
    def neu_num(self,num):
        """
        将排列组合编号处理成二元神经元的组合，用向量表示，神经元有两种状态：{1，-1}，
        比如排列组合编号为1的神经元组合就是【1，-1，-1，-1，-1，-1，-1，-1，-1，-1】###注意这里的数字顺序和二进制是倒过来的，第一个是个位
        :param num: 组合编号
        :return: 矩阵
        """
        stor = num
        ejz = np.full(shape=10, fill_value=-1)
        for i in range(9, -1, -1):
            if stor >= np.pow(2, i):
                ejz[i] = 1
                stor = stor - np.pow(2, i)
        return ejz
    def binary_num(self,datalist):
        b_list=np.zeros((N, np.size(datalist)))
        for i,k in enumerate(datalist):
            b_list[:,i]=self.neu_num(k)[:]
        return b_list

    def cal_couple_term(self,state,w):
        """
        用于计算两个神经元间的相互作用带来的能量
        :param w:相互作用项的系数
        :param N:神经元个数规定前五个是
        :return: E: 耦合项的能量
        """
        E = 0
        for i in range(9, -1, -1):
            for j in range(i, -1, -1):
                E += state[i] * state[j] * w[i, j]
        return E
    def cal_Ez_P(self, b, w):
        """
        通过计算当前的分布下的能量来确定每个状态具有的概率
        :return:
        """
        Ez = np.zeros(np.pow(2, N))
        P = np.zeros(np.pow(2, N))
        for i in range(0,np.pow(2, N)):
            state = self.neu_num(i)
            Ez[i] = -b @ state - self.cal_couple_term(state,w)
            P[i] = np.power(np.e, -Ez[i])
        Z = np.sum(P)
        P = P/Z
        return Ez, P
    def delta_x_to_b_w(self,alpha,pk):
        delta_b=np.zeros(N)
        delta_w=np.zeros((N,N))
        count=0

        for i in range(0,N):
            delta_b[i]=alpha*pk[i]
            for j in range(i,N):
                delta_w[i,j]=alpha*pk[N+count]
                count+=1
        return delta_b, delta_w
    def x_to_bw(self,x):
        c=0
        for i in range(0,N):
            b[i]=x[i]
            for j in range(i,N):
                w[i,j]=x[N+c]
                c+=1
        return b,w

    def lossfunction_KL_x(self,x):
        b,w=self.x_to_bw(x)
        P=self.cal_Ez_P(b,w)[1]
        loss=np.sum(self.Pvdata @ (np.log(self.Pvdata)-np.log(self.Pm(P))))
        return loss



    def lossfunction_KL(self,P):
        """
        计算lossfunction: Kullback-Leibler divergence
        :param Pvdata: 训练数据度的概率
        :param datalist: 训练数据的状态
        """
        loss = np.sum(self.Pvdata @ (np.log(self.Pvdata)-np.log(self.Pm(P))))
        print(loss)
        return loss
    def updata_grad_down(self):
        """
        通过梯度下降算法更新所有的参数和分布
        :return:
        """
        delta_b = np.zeros(N)
        delta_w = np.zeros((N,N))
        zazb_Pv = np.zeros(np.size(self.Pvdata))
        zazb_P = np.zeros(np.size(self.P))
        div_b = np.zeros(N)
        div_w = np.zeros((N,N))

        for i in range(0,N):
            div_b[i]=-(self.Pvdata@self.binary_datalist[i,:]-np.sum(self.Pvdata)*self.P@self.binary_all[i,:])

            for j in range(i, N):
                zazb_Pv[:]=self.binary_datalist[i,:]*self.binary_datalist[j,:]
                zazb_P[:]=self.binary_all[i, :]*self.binary_all[j, :]
                div_w[i,j]=-(self.Pvdata@zazb_Pv-np.sum(self.Pvdata)*(self.P@zazb_P))
        delta_b=-self.eta*div_b
        delta_w = -self.eta*div_w
        self.b+=delta_b
        self.w+=delta_w
        self.Ez, self.P = self.cal_Ez_P(self.b,self.w)
        self.loss.append(self.lossfunction_KL(self.P))
    def grad_f(self,x):
        co=0
        cc=0
        gd_f=np.zeros(int(N + N * (N + 1) / 2))
        b=np.zeros(N)
        w=np.zeros((N,N))
        for i in range(0,N):
            b[i]=x[i]
            for j in range(i,N):
                w[i,j]=x[N+cc]
                cc+=1
        P=self.cal_Ez_P(b,w)[1]
        for i in range(0,N):
            gd_f[i]=(self.Pvdata@self.binary_datalist[i,:]-self.Pm(P)@self.binary_datalist[i,:])
            for j in range(i,N):
                zazb_Pv = self.binary_datalist[i,:]*self.binary_datalist[j,:]
                zazb_P = self.binary_all[i,:]*self.binary_all[j,:]
                gd_f[co+N]=(self.Pvdata@zazb_Pv-np.sum(self.Pvdata)*(P@zazb_P))
                co+=1
        return gd_f


    def update_BFGS(self):
        """
        使用BFGS下降lossfunction
        :return:
        """
        delta_b = np.zeros(N)
        delta_w = np.zeros((N, N))
        zazb_Pv = np.zeros(np.size(self.Pvdata))
        zazb_P = np.zeros(np.size(self.P))
        div_b = np.zeros(N)
        div_b2 = np.zeros(N)
        div_w = np.zeros((N, N))
        div_w2 = np.zeros((N, N))
        P=np.zeros(np.pow(2, N))
        bk=np.zeros(N)
        wk=np.zeros((N,N))
        div_f = np.zeros(int(N + N * (N + 1) / 2))
        div_f2=np.zeros(int(N + N * (N + 1) / 2))
        pk = np.zeros(int(N + N * (N + 1) / 2))
        alpha = 10
        rho = 0.9
        c1 = np.power(10.0, -2)
        c2 = 0.9
        count = 0
        count2=0
        f1=0
        f2=0
        xk=np.zeros(int(N + N * (N + 1) / 2))
        xk1=np.zeros(int(N + N * (N + 1) / 2))
        sk=np.zeros(int(N + N * (N + 1) / 2))
        yk=np.zeros(int(N + N * (N + 1) / 2))

        for i in range(0,N):
            div_b[i]=(self.Pvdata@self.binary_datalist[i,:]-self.Pm(self.P)@self.binary_datalist[i,:])
            div_f[i] = div_b[i]
            xk[i]=self.b[i]
            for j in range(i, N):
                zazb_Pv[:]=self.binary_datalist[i,:]*self.binary_datalist[j,:]
                zazb_P[:]=self.binary_all[i, :]*self.binary_all[j, :]
                div_w[i,j]=(self.Pvdata@zazb_Pv-np.sum(self.Pvdata)*(self.P@zazb_P))
                div_f[N + count] = div_w[i, j]
                xk[N+count]=self.w[i,j]
                count += 1
        f1=self.lossfunction_KL(self.P)
        pk = -self.H @ div_f
        print('PK=',pk)
        bk=self.b+self.delta_x_to_b_w(alpha,pk)[0]
        wk=self.w+self.delta_x_to_b_w(alpha,pk)[1]
        P=self.cal_Ez_P(bk,wk)[1]
        f2=self.lossfunction_KL(P)
        for i in range(0,N):
            div_b2[i]=(self.Pvdata@self.binary_datalist[i,:]-self.Pm(P)@self.binary_datalist[i,:])
            div_f2[i] = div_b2[i]
            xk[i]=bk[i]
            for j in range(i, N):
                zazb_Pv[:]=self.binary_datalist[i,:]*self.binary_datalist[j,:]
                zazb_P[:]=self.binary_all[i, :]*self.binary_all[j, :]
                div_w2[i,j]=(self.Pvdata@zazb_Pv-np.sum(self.Pvdata)*(P@zazb_P))
                div_f2[N + count2] = div_w2[i, j]
                xk[N+count2]=wk[i,j]
                count2 += 1
        count=0
        count2=0


        while True:
            if f2<=f1+c1*alpha*(div_f@pk) and div_f2@pk>=c2*div_f@pk:
                break
            if f2<=f1+c1*alpha*(div_f@pk) and alpha<0.01:
                break
            print(f"f2={f2}, f1+c1*alpha*(div_f@pk)={f1+c1*alpha*(div_f@pk)}")
            print(f"div_f2@pk={div_f2@pk}, c2*div_f@pk={c2*div_f@pk}")
            alpha=rho*alpha
            print(alpha)
            bk = self.b + self.delta_x_to_b_w(alpha, pk)[0]
            wk = self.w + self.delta_x_to_b_w(alpha, pk)[1]
            P = self.cal_Ez_P(bk, wk)[1]
            f2 = self.lossfunction_KL(P)
            for i in range(0, N):
                div_b2[i] = (self.Pvdata @ self.binary_datalist[i, :] - self.Pm(P) @ self.binary_datalist[i, :])
                div_f2[i] = div_b2[i]
                xk1[i]=bk[i]
                for j in range(i, N):
                    zazb_Pv[:] = self.binary_datalist[i, :] * self.binary_datalist[j, :]
                    zazb_P[:] = self.binary_all[i, :] * self.binary_all[j, :]
                    div_w2[i, j] = (self.Pvdata @ zazb_Pv - np.sum(self.Pvdata) * (P @ zazb_P))
                    div_f2[N + count2] = div_w2[i, j]
                    xk1[N+count2]=wk[i,j]
                    count2 += 1
            count2 = 0
        sk=xk1-xk
        yk=div_f2-div_f
        rk=1/(yk@sk)
        self.b=bk
        self.w=wk
        self.H=(np.identity(int(N+N*(N+1)/2))-rk*(np.outer(sk,yk)))@self.H@(np.identity(int(N+N*(N+1)/2))-rk*(np.outer(sk,yk)))+rk*np.outer(sk,sk)
        self.Ez, self.P = self.cal_Ez_P(self.b, self.w)
        self.loss.append(self.lossfunction_KL(self.P))
    def update_BFGS_sci(self):
        x0=np.zeros(int(N+N*(N+1)/2))
        c=0
        for i in range(0,N):
            x0[i]=self.b[i]
            for j in range(i,N):
                x0[i+c]=self.w[i,j]
                c+=1

        res=minimize(fun=self.lossfunction_KL_x,x0=x0,method='BFGS',jac=self.grad_f,options={'gtol':1e-3,'maxiter':25, 'disp':True})



    def train_grad_down(self):
        """
        训练网络把训练数据集跑iteration次
        :return:
        """
        for i in range(0,self.iteration):
            print("iteration:",i)
            self.updata_grad_down()
            self.eta_cos=self.eta*np.cos(np.pi*i/(500))
        plt.plot(np.arange(self.iteration),self.loss)
        plt.show()
    def train_BFGS(self):
        for i in range(0,self.iteration):
            print("iteration:",i)
            self.update_BFGS()
        plt.plot(np.arange(self.iteration),self.loss)
        plt.show()

BM_test=Bm(b, w, Pvdata, points, 25)
BM_test.train_grad_down()
np.save("b.npy",BM_test.b)
np.save("w.npy",BM_test.w)
bk=np.load("b.npy")
wk=np.load("w.npy")
#BM_test_BFGS=Bm(bk,wk,Pvdata,points, iteration=25)
#BM_test_BFGS.train_BFGS()
BM_test.update_BFGS_sci()