import NormalInvWish as ni
import numpy as np
from scipy.stats import multivariate_normal, beta, invgamma, bernoulli
from scipy.stats import t as student_t
from math import log, exp
import numpy.linalg as nl

G = 17603
S = 5
T = 11
L = 3
data = np.loadtxt("Normalized.txt", delimiter=',').reshape((G, S, T))
e_m = np.loadtxt("e_m.txt", delimiter=',')

__metaclass__=type
class timeRNA:
    def __init__(self):
        self.h_m = np.zeros((G, S))
        self.b_v = np.zeros((L, 1))
        self.SIGMA_m = np.identity(L)
        self.sigmasq_v = [1, ] * G
        self.beta_m = np.ones((G, S, L))
        self.p_v = [1/2, ] * G
        self.p_alpha = 0.2017988
        self.p_beta = 0.2412342
        self.t_mu = -2.881022
        self.t_nu = 1.560685
        self.t_tausq = 0.1816126
    def h_update(self):
        mean_0 = (0, ) * T
        for g in range(G):
            var_0 = np.identity(T) * self.sigmasq_v[g]
            mvn_0 = multivariate_normal(mean=mean_0, cov=var_0)
            mean_1 = e_m.dot(self.b_v).reshape(T)
            var_1 = var_0 + e_m.dot(self.SIGMA_m).dot(e_m.T)
            mvn_1 = multivariate_normal(mean=mean_1, cov=var_1)
            for s in range(S):
                logp = mvn_0.logpdf(data[g][s].T) + log(1 - self.p_v[g]), mvn_1.logpdf(data[g][s].T) + log(self.p_v[g])
                p0, p1 = [ exp(z - max(logp)) for z in logp ]
                self.h_m[g][s] = bernoulli.rvs(p1/(p1 + p0))
    def beta_update(self):
        self.beta_m[self.h_m == 0] = 0
        inv_SIGMA_m = nl.inv(self.SIGMA_m)
        for g in range(G):
            var_m = nl.inv(e_m.T.dot(e_m) / self.sigmasq_v[g] + inv_SIGMA_m)
            for s in range(S):
                if self.h_m[g][s] == 1:
                    mean_v = var_m.dot(np.add(inv_SIGMA_m.dot(self.b_v), e_m.T.dot(data[g][s].reshape((11, 1))) / self.sigmasq_v[g])).reshape(L)
                    self.beta_m[g][s] = np.random.multivariate_normal(mean=mean_v, cov=var_m)
    def b_SIGMA_update(self):
        beta_all = []
        for g in range(G):
            for s in range(S):
                if self.h_m[g][s] == 1:
                    beta_all.append(self.beta_m[g][s].tolist())
        beta_all = np.array(beta_all)
        mean_beta = sum(beta_all) / beta_all.shape[0]
        sub_beta = beta_all - mean_beta
        S_m = np.zeros((L, L))
        for l1 in range(L):
            for l2 in range(l1, L):
                S_m[l1][l2] = S_m[l2][l1] = sum(sub_beta[:, l1] * sub_beta[:, l2])
        # for beta in beta_all:
        #     temp = np.subtract(beta, self.b_v)
        #     S_m = np.add(S_m, temp.T.dot(temp))
#        self.SIGMA_m = ni.sample_invwishart(nl.inv(S_m), len(beta_all) - 1)
        self.SIGMA_m = ni.sample_invwishart(S_m, len(beta_all) - 1)
        self.b_v = np.random.multivariate_normal(mean=mean_beta, cov=self.SIGMA_m /len(beta_all)).reshape((3, 1))
    def p_update(self):
        suc = [sum(h) for h in self.h_m]
        self.p_v = [beta.rvs(self.p_alpha + s, self.p_beta + S - s) for s in suc]
    def sigmasq_update(self):
        for g in range(G):
            mean_m = e_m.dot(self.beta_m[g].T).T
            scale = sum(sum((data[g] - mean_m)**2)) / 2
            shape = S * T / 2
            prop = invgamma.rvs(a=shape, scale=scale)
            old = self.sigmasq_v[g]
            var_prop = np.identity(T) * prop
            var_old = np.identity(T) * old
            log_1 = sum([multivariate_normal.logpdf(x, mean=m, cov=var_prop) for (m, x) in zip(mean_m, data[g])]) - sum([multivariate_normal.logpdf(x, mean=m, cov=var_old) for (m, x) in zip(mean_m, data[g])])
            log_2 = student_t.logpdf(log(prop), loc=self.t_mu, scale=self.t_tausq, df=self.t_nu) - student_t.logpdf(log(old), loc=self.t_mu, scale=self.t_tausq, df=self.t_nu) + log(old) - log(prop)
            log_3 = invgamma.logpdf(old, a=shape, scale=scale) - invgamma.logpdf(prop, a=shape, scale=scale)
            p = min(1, exp(log_1 + log_2 + log_3))
            self.sigmasq_v[g] = prop if bernoulli.rvs(p) else old
    # def MCMCupdate(self, burn=1):
    #     for g in range(burn):
    #         self.h_update()
    #         self.beta_update()
    #         self.b_SIGMA_update()
    #         self.p_update()
    #         self.sigmasq_update()

timed = timeRNA()
burn = 2
for g in range(burn):
    timed.h_update()
    timed.beta_update()
    timed.b_SIGMA_update()
    timed.p_update()
    timed.sigmasq_update()

print(timed.h_m.shape)
print(timed.beta_m.shape)
print(timed.b_v.shape)
print(timed.SIGMA_m.shape)
print(len(timed.p_v))
print(len(timed.sigmasq_v))
