'''
Solving the DSGE in the paper

I only replicate the information sufficiency and IRFs. I take the Bayesian estimation for calibration as give, since it's not the focus of the paper.
'''

# === PACKAGES AND IMPORTS =======

import numpy as np
import pandas as pd
from fredapi import Fred
from scipy.linalg import ordqz
from numpy.linalg import solve, lstsq, cond
import matplotlib.pyplot as plt

FRED_API_KEY = '5ca03acdfbe96dfd8b54a70edb9f2e5a'
fred = Fred(api_key=FRED_API_KEY)

Y = fred.get_series("GDPC1")
C = fred.get_series("PCECC96")
I = fred.get_series("GPDIC1")
df = pd.concat([Y, C, I], axis=1)
df.columns = ["Y", "C", "I"]
df = df.dropna()

# ==== PARAMETERS ====

zeta = 2.07
chi = 5.5
hgreek = 0.53
sigmaf = 3.98
thetaw = 0.87
thetap = 0.88
gammapi = 1.003
gammay = 0.0044
mup = 0.3
muw = 0.05
alpha = 0.19
Gamma = 1
psi = 0.22
delta = 0.025
beta = 0.99
rhor = 0.57
rho = 0.96
rhoq = 0.19
rhod = 0.68
rhop = 0.81
rhow = 0.95
rhog = 0.98
psip = 0.49
psiw = 0.96
Sigmae = 0.98
Sigmav = 1.28
Sigmaq = 0.26
Sigmad = 4.84
Sigmap = 0.14
Sigmaw = 0.40
Sigmag = 0.52

# I don't see the solution of the model so I just calibrate the steady state to real data
# The values for steady state C/Y, I/Y taken from the US data.
# The value for the steady state capital share of income is, in this model alpha times 1/markup. 
# Given the description of the model, the markup is 1+mup

CY = (df["C"] / df["Y"]).mean() # I take from FED
IY = 1 - CY - psi # Psi is the calibrated G/Y. 
RkK_over_PY = alpha * 1/(1+mup)

# === DEFINITIONS =====

'''
The log linearization and dynamics of the model is provided in the paper, with:
- 13 log linear approximated equations
- 7 exogenous shock processes
- 1 monetary policy rule

I will first write the system in linear rational expectations form:

X0 xt = X1 xt-1 + X2 Ext+1 + X3 xit

xt will have all the state and control variables. ut will have all the structural shocks. 

Then I will solve it into a reduced form, so I can generate data, and fit the VARs.

With that reduced form, I can initialize all the variables and shock lags to 0 and simulate the data, where I will fit the different VAR specifications.

I define the vector x_t as follows, with the corresponding indexes

lambdahatt (0) // chatt (1) // deltaat (2) // rt (3) // rkt (4) // pit (5) // phihatt (6) // dt (7) // zetahatt (8) // ut (9) // mt (10) // whatt (11) // khatt (12) // nt (13) // kbart (14) // yhatt (15) // gt (16) // mpt (17) // mwt (18) // qt (19) // Tt (20) // epswt (21) // epst (22) // epst_1 (23) // epst_2 (24) // epst_3 (25) // epspt (26)

And the vector xit as follows, with the corresponding indexes

epsqt (0) // epsgt (1) // epswt (2) // vt (3) // epst (4) // epspt (5) // epsdt (6)
'''

nx = 27  # dimension of x_t
nu = 7   # dimension of u_t

# Structural (LRE) matrices
X0 = np.zeros((nx, nx))
X1 = np.zeros((nx, nx))
X2 = np.zeros((nx, nx))
X3 = np.zeros((nx, nu))

'''
There are 27 state variables, and 7 innovations
I need 27 equations to solve the system
'''

'''
EQUATION 1 (Taylor rule):
rt - (1-rhor)*gammapi*pit - (1-rhor)*gammay*yhatt - qt = rhor*rt-1
'''

# x_t coefficients
X0[0, 3]  = 1.0
X0[0, 5]  = -(1.0 - rhor) * gammapi
X0[0, 15] = -(1.0 - rhor) * gammay
X0[0, 19] = -1.0   # -qt on the LHS

# x_{t-1} coefficients
X1[0, 3]  = rhor

# E_t[x_{t+1}] coefficients -> none
# shock (xit) coefficients -> none (policy shock enters in the q_t process equation)

'''
EQUATION 2 (Marginal utility / habit):

lambdahatt
+? ((Gamma**2 + hgreek**2 * beta) / ((Gamma - hgreek*beta)*(Gamma - hgreek))) * chatt // Using negative sign. I think its a mistake.
+ (hgreek*Gamma / ((Gamma - hgreek*beta)*(Gamma - hgreek))) * deltaat
=
(hgreek*Gamma / ((Gamma - hgreek*beta)*(Gamma - hgreek))) * chatt(-1)
+ (hgreek*beta*Gamma / ((Gamma - hgreek*beta)*(Gamma - hgreek))) * Ext+1[chatt]
+ (hgreek*beta*Gamma / ((Gamma - hgreek*beta)*(Gamma - hgreek))) * Ext+1[deltaat]
'''

# common denominator
den = (Gamma - hgreek*beta) * (Gamma - hgreek)

# coefficients
coef_ct      = (Gamma**2 + hgreek**2 * beta) / den
coef_ct_lag  = (hgreek * Gamma) / den
coef_ct_lead = (hgreek * beta * Gamma) / den

# x_t coefficients (X0)  [row 1 = Equation 2]
X0[1, 0] = 1.0                 # lambdahatt
X0[1, 1] = -coef_ct            # chatt ??? // Yes. Changing sign vs paper. I think there was a typo.
X0[1, 2] = coef_ct_lag         # deltaat

# x_{t-1} coefficients (X1)
X1[1, 1] = coef_ct_lag          # chatt(-1)

# E_t[x_{t+1}] coefficients (X2)
X2[1, 1] = coef_ct_lead         # Ext+1[chatt]
X2[1, 2] = coef_ct_lead         # Ext+1[deltaat]

# shocks (X3) -> none

'''
EQUATION 3 (Bond Euler):

lambdahatt - rt = Ext+1[lambdahatt] - Ext+1[deltaat] - Ext+1[pit]
'''

# x_t coefficients (X0)  [row 2 = Equation 3]
X0[2, 0] = 1.0     # lambdahatt
X0[2, 3] = -1.0    # -rt

# x_{t-1} coefficients (X1) -> none

# E_t[x_{t+1}] coefficients (X2)
X2[2, 0] = 1.0     # Ext+1[lambdahatt]
X2[2, 2] = -1.0    # -Ext+1[deltaat]
X2[2, 5] = -1.0    # -Ext+1[pit]

# shocks (X3) -> none

'''
EQUATION 4 (Tobin's q / value of installed capital):

phihatt
=
((1-delta)*beta/Gamma) * Ext+1[phihatt]
+ (1 - (1-delta)*beta/Gamma) * Ext+1[lambdahatt]
- Ext+1[deltaat]
+ (1 - (1-delta)*beta/Gamma) * Ext+1[rkt]
'''

# coefficients
coef_phi_lead = ((1.0 - delta) * beta) / Gamma
coef_other_lead = 1.0 - coef_phi_lead  # = 1 - (1-delta)*beta/Gamma

# x_t coefficients (X0)  [row 3 = Equation 4]
X0[3, 6] = 1.0    # phihatt

# x_{t-1} coefficients (X1) -> none

# E_t[x_{t+1}] coefficients (X2)
X2[3, 6] = coef_phi_lead        # Ext+1[phihatt]
X2[3, 0] = coef_other_lead      # Ext+1[lambdahatt]
X2[3, 2] = -1.0                 # -Ext+1[deltaat]
X2[3, 4] = coef_other_lead      # Ext+1[rkt]

# shocks (X3) -> none

'''
EQUATION 5 (Investment FOC):

lambdahatt - phihatt - dt + chi*Gamma**2*(1+beta)*zetahatt + chi*Gamma**2*deltaat
=
chi*Gamma**2*zetahatt(-1)
+ beta*chi*Gamma**2 * Ext+1[zetahatt]
+ beta*chi*Gamma**2 * Ext+1[deltaat]
'''

coef_adj = chi * (Gamma**2)

# x_t coefficients (X0)  [row 4 = Equation 5]
X0[4, 0] = 1.0                         # lambdahatt
X0[4, 6] = -1.0                        # -phihatt
X0[4, 7] = -1.0                        # -dt
X0[4, 8] = coef_adj * (1.0 + beta)     # +chi*Gamma^2*(1+beta)*zetahatt
X0[4, 2] = coef_adj                    # +chi*Gamma^2*deltaat

# x_{t-1} coefficients (X1)
X1[4, 8] = coef_adj                    # chi*Gamma^2*zetahatt(-1)

# E_t[x_{t+1}] coefficients (X2)
X2[4, 8] = beta * coef_adj             # beta*chi*Gamma^2 * Ext+1[zetahatt]
X2[4, 2] = beta * coef_adj             # beta*chi*Gamma^2 * Ext+1[deltaat]

# shocks (X3) -> none


'''
EQUATION 6 (Utilization):

rkt - zeta*ut = 0
'''

# x_t coefficients (X0)  [row 5 = Equation 6]
X0[5, 4] = 1.0          # rkt
X0[5, 9] = -zeta        # -zeta*ut

# x_{t-1}, E_t[x_{t+1}], shocks -> none


'''
EQUATION 7 (Marginal cost):

mt - alpha*rkt - (1-alpha)*whatt = 0
'''

# x_t coefficients (X0)  [row 6 = Equation 7]
X0[6, 10] = 1.0             # mt
X0[6, 4]  = -alpha          # -alpha*rkt
X0[6, 11] = -(1.0 - alpha)  # -(1-alpha)*whatt

# x_{t-1}, E_t[x_{t+1}], shocks -> none


'''
EQUATION 8 (Rental rate condition):

rkt - whatt + khatt - nt = 0
'''

# x_t coefficients (X0)  [row 7 = Equation 8]
X0[7, 4]  = 1.0     # rkt
X0[7, 11] = -1.0    # -whatt
X0[7, 12] = 1.0     # +khatt
X0[7, 13] = -1.0    # -nt

# x_{t-1}, E_t[x_{t+1}], shocks -> none


'''
EQUATION 9 (Capital services identity):

khatt - ut + deltaat = kbart(-1)
'''

# x_t coefficients (X0)  [row 8 = Equation 9]
X0[8, 12] = 1.0     # khatt
X0[8, 9]  = -1.0    # -ut
X0[8, 2]  = 1.0     # +deltaat

# x_{t-1} coefficients (X1)
X1[8, 14] = 1.0     # kbart(-1)

# E_t[x_{t+1}], shocks -> none


'''
EQUATION 10 (Capital accumulation, corrected):

kbart + ((1-delta)/Gamma)*deltaat - (1-(1-delta)/Gamma)*dt - zetahatt
=
((1-delta)/Gamma) * kbart(-1)
'''

coef_kbar_lag = (1.0 - delta) / Gamma
coef_d = 1.0 - coef_kbar_lag

# x_t coefficients (X0)  [row 9 = Equation 10]
X0[9, 14] = 1.0                 # kbart
X0[9, 2]  = coef_kbar_lag        # +((1-delta)/Gamma)*deltaat
X0[9, 7]  = -coef_d              # -(1-(1-delta)/Gamma)*dt
# X0[9, 8]  = -1.0                 # -zetahatt as written in the paper.
X0[9, 8]  = -coef_d  # This is how I think it really is

# x_{t-1} coefficients (X1)
X1[9, 14] = coef_kbar_lag        # ((1-delta)/Gamma) * kbart(-1)

# E_t[x_{t+1}], shocks -> none


'''
EQUATION 11 (Production):

yhatt - alpha*khatt - (1-alpha)*nt = 0
'''

# x_t coefficients (X0)  [row 10 = Equation 11]
X0[10, 15] = 1.0             # yhatt
X0[10, 12] = -alpha          # -alpha*khatt
X0[10, 13] = -(1.0 - alpha)  # -(1-alpha)*nt

# x_{t-1}, E_t[x_{t+1}], shocks -> none


'''
EQUATION 12 (Goods-market clearing):

(1-psi)*yhatt - (C/Y)*chatt - (I/Y)*zetahatt - (RkK_over_PY)*ut - gt = 0
'''

# x_t coefficients (X0)  [row 11 = Equation 12]
X0[11, 15] = (1.0 - psi)      # (1-psi)*yhatt
X0[11, 1]  = -CY              # -(C/Y)*chatt
X0[11, 8]  = -IY              # -(I/Y)*zetahatt
X0[11, 9]  = -RkK_over_PY     # -(R^k K)/(P Y) * ut
X0[11, 16] = -1.0             # -gt

# x_{t-1}, E_t[x_{t+1}], shocks -> none


'''
EQUATION 13 (Price Phillips curve):

pit - kappa*mt - kappa*mpt = beta * Ext+1[pit]

where
kappa = ((1-thetap*beta)*(1-thetap)) / thetap
'''

kappa = ((1.0 - thetap * beta) * (1.0 - thetap)) / thetap

# x_t coefficients (X0)  [row 12 = Equation 13]
X0[12, 5]  = 1.0        # pit
X0[12, 10] = -kappa     # -kappa*mt
X0[12, 17] = -kappa     # -kappa*mpt

# E_t[x_{t+1}] coefficients (X2)
X2[12, 5]  = beta       # beta*Ext+1[pit]

# x_{t-1}, shocks -> none


'''
EQUATION 14 (Wage setting / wage Phillips):

(1+kappaw)*whatt + (kappaw*zeta)*nt? NO -> careful with signs:
From the paper:
whatt = 1/(1+beta)*whatt(-1) + beta/(1+beta)*Ext+1[whatt]
        - 1/(1+beta)*(pit + deltaat) + beta/(1+beta)*Ext+1[(pit + deltaat)]
        - kappaw*(whatt - zeta*nt + lambdahatt + kappaw*mwt)

So the LRE row form is:

(1+kappaw)*whatt - (kappaw*zeta)*nt + kappaw*lambdahatt + (kappaw**2)*mwt
+ (1/(1+beta))*pit + (1/(1+beta))*deltaat
=
(1/(1+beta))*whatt(-1)
+ (beta/(1+beta))*Ext+1[whatt]
+ (beta/(1+beta))*Ext+1[pit]
+ (beta/(1+beta))*Ext+1[deltaat]

where
kappaw = ((1-thetaw*beta)*(1-thetaw)) / (thetaw*(1+beta)*(1 + zeta*(1 + 1/muw)))
'''

kappaw = ((1.0 - thetaw * beta) * (1.0 - thetaw)) / (thetaw * (1.0 + beta) * (1.0 + zeta * (1.0 + 1.0/muw)))
a0 = 1.0 / (1.0 + beta)
a1 = beta / (1.0 + beta)

# x_t coefficients (X0)  [row 13 = Equation 14]
X0[13, 11] = 1.0 + kappaw           # (1+kappaw)*whatt
X0[13, 13] = -(kappaw * zeta)       # -(kappaw*zeta)*nt
X0[13, 0]  = kappaw                 # +kappaw*lambdahatt
X0[13, 18] = (kappaw**2)            # +(kappaw^2)*mwt
X0[13, 5]  = a0                     # +(1/(1+beta))*pit
X0[13, 2]  = a0                     # +(1/(1+beta))*deltaat

# x_{t-1} coefficients (X1)
X1[13, 11] = a0                     # (1/(1+beta))*whatt(-1)

# E_t[x_{t+1}] coefficients (X2)
X2[13, 11] = a1                     # (beta/(1+beta))*Ext+1[whatt]
X2[13, 5]  = a1                     # (beta/(1+beta))*Ext+1[pit]
X2[13, 2]  = a1                     # (beta/(1+beta))*Ext+1[deltaat]

# shocks (X3) -> none

'''
EQUATION 15 (News technology shock)

deltaat - Tt = epst-4 - Tt-1

and I treat epst-4 as (epst-3)t-1
'''

# x_t coefficients (X0)
X0[14,2] = 1
X0[14,20] = -1

# x_{t-1} coefficients (X1)
X1[14,25] = 1
X1[14,20] = -1

# E_t[x_{t+1}] coefficients (X2) -> none

# shocks (X3) -> none

'''
EQUATION 16 (Temporary technology shock)

Tt = rho Tt-1 + vt
'''

# x_t coefficients (X0)
X0[15,20] = 1

# x_{t-1} coefficients (X1)
X1[15,20] = rho

# E_t[x_{t+1}] coefficients (X2) -> none

# shocks (X3)
X3[15,3] = np.sqrt(Sigmav) # Multiply the N(0,1) shock by its calibrated stdev

'''
EQUATION 17 (Investment specific technology shock)

dt = rhod dt-1 + epsdt
'''

# x_t coefficients (X0)
X0[16,7] = 1

# x_{t-1} coefficients (X1)
X1[16,7] = rhod

# E_t[x_{t+1}] coefficients (X2) -> none

# shocks (X3)
X3[16,6] = np.sqrt(Sigmad) # Multiply the N(0,1) shock by its calibrated stdev

'''
EQUATION 18 (Price Markup Shock)

mpt = rhop mpt-1 + epspt - psip epspt-1
'''

# x_t coefficients (X0)
X0[17,17] = 1
X0[17,26] = -1 # (take the shock from the state)

# x_{t-1} coefficients (X1)
X1[17,17] = rhop
X1[17,26] = -psip

# E_t[x_{t+1}] coefficients (X2) -> none

# shocks (X3) -> none (taking it from the state)

'''
EQUATION 19 (Wage Markup Shock)

mwt = rhow mwt-1 + epswt - psiw epswt-1
'''

# x_t coefficients (X0)
X0[18,18] = 1
X0[18,21] = -1 # (take the shock from the state)

# x_{t-1} coefficients (X1)
X1[18,18] = rhow
X1[18,21] = -psiw

# E_t[x_{t+1}] coefficients (X2) -> none

# shocks (X3) -> none (taking it from the state)

'''
EQUATION 20 (Government Spending Shock)

gt = rhog gt-1 + epsgt
'''

# x_t coefficients (X0)
X0[19,16] = 1

# x_{t-1} coefficients (X1)
X1[19,16] = rhog

# E_t[x_{t+1}] coefficients (X2) -> none

# shocks (X3)
X3[19,1] = np.sqrt(Sigmag) # Multiply the N(0,1) shock by its calibrated stdev

'''
EQUATION 21 (Monetary Policy Shock)

qt = rhoq qt-1 + epsqt
'''

# x_t coefficients (X0)
X0[20,19] = 1

# x_{t-1} coefficients (X1)
X1[20,19] = rhoq

# E_t[x_{t+1}] coefficients (X2) -> none

# shocks (X3)
X3[20,0] = np.sqrt(Sigmaq) # Multiply the N(0,1) shock by its calibrated stdev

'''
EQUATIONS 22 AND 23 (Buffers for the MA Processes)

epswt (state) = epswt (innovation)
epspt (state) = epspt (innovation)
'''

# x_t coefficients (X0)
X0[21,21] = 1
X0[22,26] = 1

# x_{t-1} coefficients (X1) -> none

# E_t[x_{t+1}] coefficients (X2) -> none

# shocks (X3)
X3[21,2] = np.sqrt(Sigmaw) # Multiply the N(0,1) shock by its calibrated stdev
X3[22,5] = np.sqrt(Sigmap) # Multiply the N(0,1) shock by its calibrated stdev

'''
EQUATIONS 24,25,26 AND 27 (Buffers for the 4th Lag of the Tech Innovation Shock)

epst (state) = epst (innovation)
epst-1 (state t) = epst-1 (epst lag 1)
epst-2 (state t) = epst-2 (epst-1 lag 1)
epst-3 (state t) = epst-3 (epst-2 lag 1)
'''

# x_t coefficients (X0)
X0[23,22] = 1
X0[24,23] = 1
X0[25,24] = 1
X0[26,25] = 1

# x_{t-1} coefficients (X1)
X1[24,22] = 1
X1[25,23] = 1
X1[26,24] = 1

# E_t[x_{t+1}] coefficients (X2) -> none

# shocks (X3)
X3[23,4] = np.sqrt(Sigmae) # Multiply the N(0,1) shock by its calibrated stdev


# === SOLVING THE DSGE =====

def solve_news_model(X0, X1, X2, X3,
                     tol_eig=1e-8,
                     newton_tol_res=1e-10,
                     newton_tol_step=1e-14,
                     newton_max_iter=30):
    """
    Solve the linear RE system

        X0 x_t = X1 x_{t-1} + X2 x_{t+1} + X3 u_t

    and return a stable reduced form

        x_t = F x_{t-1} + G u_t

    First, build an initial F from the QZ decomposition,
    then refine F with a Newton method that enforces

        X0 F = X1 + X2 F^2

    up to numerical precision.
    """

    n = X0.shape[0]
    k = X3.shape[1]

    # -----------------------------
    # 1. QZ step: initial F (your old code)
    # -----------------------------
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-X1,              -X0     ]
    ])

    B = np.block([
        [np.eye(n),        np.zeros((n, n))],
        [np.zeros((n, n)), -X2             ]
    ])

    S, T, alpha, beta, Q, Z = ordqz(
        A, B,
        sort='iuc',      # inside unit circle first
        output='complex'
    )

    eigvals = np.full_like(alpha, np.inf, dtype=np.complex128)
    mask = np.abs(beta) > tol_eig
    eigvals[mask] = alpha[mask] / beta[mask]

    stable_idx = np.where(np.abs(eigvals) < 1 - tol_eig)[0]
    n_stable = len(stable_idx)
    if n_stable < n:
        raise RuntimeError(
            f"Not enough stable eigenvalues to build solution: {n_stable} < {n}"
        )

    # Select n linearly independent stable eigenvectors
    chosen = []
    Z_top = Z[:n, :]
    for j in stable_idx:
        cols = chosen + [j]
        Z1_candidate = Z_top[:, cols]
        rank = np.linalg.matrix_rank(Z1_candidate, tol_eig)
        if rank == len(cols):
            chosen = cols
            if len(chosen) == n:
                break

    if len(chosen) < n:
        raise RuntimeError(
            "Could not find n linearly independent stable eigenvectors; "
            "model likely indeterminate or mis-specified."
        )

    Zs = Z[:, chosen]        # 2n x n
    Z1 = Zs[:n, :]           # n x n
    Z2 = Zs[n:, :]           # n x n

    # Initial policy matrix from QZ
    F = np.real(Z2 @ np.linalg.inv(Z1))

    # -----------------------------
    # 2. Newton refinement on F
    # -----------------------------
    def vec(M):
        return M.reshape(-1, order="F")

    def mat(v):
        return v.reshape((n, n), order="F")

    def R_of(Fmat):
        # residual of the Sylvester-type equation: X0 F = X1 + X2 F^2
        return X0 @ Fmat - X1 - X2 @ (Fmat @ Fmat)

    I_n = np.eye(n)

    Fcur = F.copy()
    Rcur = R_of(Fcur)
    res = np.max(np.abs(Rcur))
    # small diagnostic print (optional)
    print(f"Newton for F: start ||R||_inf = {res:.3e}")

    for it in range(1, newton_max_iter + 1):
        if res < newton_tol_res:
            break

        # Jacobian of R(F) w.r.t. vec(F): J = kron(I, X0) - kron(I, X2 F) - kron(F^T, X2)
        J = np.kron(I_n, X0) - np.kron(I_n, X2 @ Fcur) - np.kron(Fcur.T, X2)

        gv = vec(Rcur)
        try:
            dvec = np.linalg.solve(J, -gv)
        except Exception:
            dvec, *_ = np.linalg.lstsq(J, -gv, rcond=None)

        dF = mat(dvec)
        maxd = np.max(np.abs(dF))
        if maxd < newton_tol_step:
            print(f"Newton: step tiny at iter {it}, ||dF||_inf={maxd:.3e} — stopping")
            break

        # backtracking line search on step length
        step = 1.0
        best_F = Fcur
        best_R = Rcur
        best_res = res
        improved = False

        for _bt in range(11):
            Ftrial = Fcur + step * dF
            Rtrial = R_of(Ftrial)
            res_trial = np.max(np.abs(Rtrial))
            if (res_trial < res * (1 - 1e-12)) or (res_trial < res - 1e-12):
                best_F = Ftrial
                best_R = Rtrial
                best_res = res_trial
                improved = True
                break
            step *= 0.5

        if not improved:
            # accept smallest step even if not strictly improving — just in case
            Fcur = Ftrial
            Rcur = Rtrial
            res = res_trial
            print(f"Newton iter {it:2d}: no improvement, step={step:.3e}, ||R||_inf={res:.3e}")
        else:
            Fcur = best_F
            Rcur = best_R
            res = best_res
            print(f"Newton iter {it:2d}: ||R||_inf={res:.3e}, step={step:.3e}")

        if res < newton_tol_res:
            break

    F = Fcur
    print(f"Newton for F: final ||R||_inf = {res:.3e}")

    # -----------------------------
    # 3. Solve for G from (X0 - X2 F) G = X3
    # -----------------------------
    M = X0 - X2 @ F
    try:
        if np.linalg.cond(M) < 1e12:
            G = np.linalg.solve(M, X3)
        else:
            G = np.linalg.lstsq(M, X3, rcond=None)[0]
    except Exception:
        G = np.linalg.lstsq(M, X3, rcond=None)[0]

    return F, G, eigvals

F, G, eigvals = solve_news_model(X0, X1, X2, X3)

np.savez(
    "dsge_solution.npz",
    F=F,
    G=G
)

# === CHECKING THE DSGE =====

T_horiz = 30

# MIT shock: 1 in period 0 for the epst innovation
shock_idx = 4
shock_size = 1.0

# Dimensions
n = F.shape[0]
k = G.shape[1]

# Storage for simulated path x_0,...,x_T (these are deviations / stationary objects)
x_path = np.zeros((T_horiz + 1, n))

# Start at steady state (zero deviations)
x = np.zeros(n)

# Shock vector u_0
u0 = np.zeros(k)
u0[shock_idx] = shock_size

# Period 0 (impact)
x = F @ x + G @ u0
x_path[0, :] = x

# Periods 1..T (no further shocks)
for t in range(1, T_horiz + 1):
    x = F @ x
    x_path[t, :] = x

# === Extract the model variables (your indices) ===
# NOTE: these are deviations defined as in the paper:
# - chat = log(C_t/A_t) - log(C/A)         :contentReference[oaicite:0]{index=0}
# - yhat = log(Y_t/A_t) - log(Y/A)         :contentReference[oaicite:1]{index=1}
# - ihat (your index 8) corresponds to the stationary investment object (I_t/A_t)
# - n    = log N_t - log N                :contentReference[oaicite:2]{index=2}
# - r    = log R_t - log R                :contentReference[oaicite:3]{index=3}
# - pi   = log(P_t/P_{t-1}) - pi_bar      :contentReference[oaicite:4]{index=4}
# - deltaat in the equilibrium conditions is TFP growth deviation (Δa_t); a_t is a random walk,
#   so to get TFP level you cumulate Δa_t.                   :contentReference[oaicite:5]{index=5}

da   = x_path[:, 2]    # TFP growth deviation (Δa_t)  (your "deltaat")
chat = x_path[:, 1]    # consumption stationary object (ĉ_t)
r    = x_path[:, 3]    # nominal interest rate deviation (r_t)
pi   = x_path[:, 5]    # inflation deviation (π_t)
ihat = x_path[:, 8]    # investment stationary object (î_t / ζ̂_t)
nhrs = x_path[:, 13]   # hours deviation (n_t)
yhat = x_path[:, 15]   # output stationary object (ŷ_t)

# === Build "level" log IRFs for TFP, C, I, Y ===
# TFP level (log) a_t is the cumulative sum of Δa_t (baseline a_{-1}=0 for IRFs)
a = np.cumsum(da)

# Since ĉ_t = log(C_t/A_t) - const, then log C_t IRF = ĉ_t + a_t (constants drop in IRFs)
logC_irf = chat + a
logI_irf = ihat + a
logY_irf = yhat + a

# Hours are already stationary in logs; interest rate and inflation are stationary too.
logH_irf = nhrs
r_irf    = r
pi_irf   = pi

# === Scaling for plots ===
# Real quantities in log deviations → *100 = percent deviation from baseline
TFP_plot = 100.0 * a
C_plot   = 100.0 * logC_irf
I_plot   = 100.0 * logI_irf
Y_plot   = 100.0 * logY_irf
H_plot   = 100.0 * logH_irf

# For r and pi:
# - If you want in percent (log points), use *100
# - If you want annualized percentage points in quarterly data, use *400
annualize = False # In the paper they use quarterly deviations
if annualize:
    r_plot  = 400.0 * r_irf
    pi_plot = 400.0 * pi_irf
else:
    r_plot  = 100.0 * r_irf
    pi_plot = 100.0 * pi_irf

### PLOT ###

h = np.arange(T_horiz + 1)

# Figure 7 layout: 2 rows x 4 columns (7 panels used, last left blank)
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
axes = axes.ravel()

# Panel 1: TFP
axes[0].plot(h, TFP_plot)
axes[0].set_title("TFP")
axes[0].axhline(0, linewidth=0.8)

# Panel 2: Output (GDP)
axes[1].plot(h, Y_plot)
axes[1].set_title("GDP")
axes[1].axhline(0, linewidth=0.8)

# Panel 3: Consumption
axes[2].plot(h, C_plot)
axes[2].set_title("Consumption")
axes[2].axhline(0, linewidth=0.8)

# Panel 4: Investment
axes[3].plot(h, I_plot)
axes[3].set_title("Investment")
axes[3].axhline(0, linewidth=0.8)

# Panel 5: Hours
axes[4].plot(h, H_plot)
axes[4].set_title("Hours")
axes[4].axhline(0, linewidth=0.8)

# Panel 6: Interest rate
axes[5].plot(h, r_plot)
axes[5].set_title("Interest rate")
axes[5].axhline(0, linewidth=0.8)

# Panel 7: Inflation
axes[6].plot(h, pi_plot)
axes[6].set_title("Inflation")
axes[6].axhline(0, linewidth=0.8)

# Panel 8: empty (Figure 7 has 7 variables)
axes[7].axis("off")

# Styling to resemble paper figure
for ax in axes[:7]:
    ax.set_xlim(0, T_horiz)
    ax.tick_params(axis="both", labelsize=9)

axes[4].set_xlabel("quarters")
axes[5].set_xlabel("quarters")
axes[6].set_xlabel("quarters")

fig.tight_layout()
plt.show()

