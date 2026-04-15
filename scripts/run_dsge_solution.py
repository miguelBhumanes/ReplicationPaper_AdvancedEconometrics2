'''
Solving the DSGE in the paper.

Produces: data/processed/dsge_solution.npz  (F, G matrices)

Requires: FRED_API_KEY environment variable
          (see .env.example)
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred

from replication.dsge.solver import solve_news_model
from replication.io.paths import PROCESSED_DATA_DIR

# === FRED DATA (for steady-state calibration) =======

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise EnvironmentError(
        "FRED_API_KEY environment variable is not set. "
        "Copy .env.example to .env, fill in your key, then run: "
        "  export FRED_API_KEY=$(grep FRED_API_KEY .env | cut -d= -f2)"
    )

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

# Steady-state ratios from US data
CY = (df["C"] / df["Y"]).mean()
IY = 1 - CY - psi
RkK_over_PY = alpha * 1/(1+mup)

# === DEFINITIONS =====

'''
The log linearization and dynamics of the model is provided in the paper, with:
- 13 log linear approximated equations
- 7 exogenous shock processes
- 1 monetary policy rule

System:  X0 xt = X1 xt-1 + X2 Ext+1 + X3 xit

x_t vector (index → variable):
  0  lambdahatt   1  chatt      2  deltaat    3  rt       4  rkt
  5  pit          6  phihatt    7  dt         8  zetahatt 9  ut
  10 mt           11 whatt      12 khatt      13 nt       14 kbart
  15 yhatt        16 gt         17 mpt        18 mwt      19 qt
  20 Tt           21 epswt      22 epst       23 epst_1   24 epst_2
  25 epst_3       26 epspt

u_t vector (index → shock):
  0 epsqt   1 epsgt   2 epswt   3 vt   4 epst   5 epspt   6 epsdt
'''

nx = 27  # dimension of x_t
nu = 7   # dimension of u_t

# Structural (LRE) matrices
X0 = np.zeros((nx, nx))
X1 = np.zeros((nx, nx))
X2 = np.zeros((nx, nx))
X3 = np.zeros((nx, nu))

# EQUATION 1 (Taylor rule):
# rt - (1-rhor)*gammapi*pit - (1-rhor)*gammay*yhatt - qt = rhor*rt-1
X0[0, 3]  = 1.0
X0[0, 5]  = -(1.0 - rhor) * gammapi
X0[0, 15] = -(1.0 - rhor) * gammay
X0[0, 19] = -1.0
X1[0, 3]  = rhor

# EQUATION 2 (Marginal utility / habit):
den = (Gamma - hgreek*beta) * (Gamma - hgreek)
coef_ct      = (Gamma**2 + hgreek**2 * beta) / den
coef_ct_lag  = (hgreek * Gamma) / den
coef_ct_lead = (hgreek * beta * Gamma) / den

X0[1, 0] = 1.0
X0[1, 1] = -coef_ct
X0[1, 2] = coef_ct_lag
X1[1, 1] = coef_ct_lag
X2[1, 1] = coef_ct_lead
X2[1, 2] = coef_ct_lead

# EQUATION 3 (Bond Euler):
# lambdahatt - rt = Ext+1[lambdahatt] - Ext+1[deltaat] - Ext+1[pit]
X0[2, 0] = 1.0
X0[2, 3] = -1.0
X2[2, 0] = 1.0
X2[2, 2] = -1.0
X2[2, 5] = -1.0

# EQUATION 4 (Tobin's q):
coef_phi_lead   = ((1.0 - delta) * beta) / Gamma
coef_other_lead = 1.0 - coef_phi_lead

X0[3, 6] = 1.0
X2[3, 6] = coef_phi_lead
X2[3, 0] = coef_other_lead
X2[3, 2] = -1.0
X2[3, 4] = coef_other_lead

# EQUATION 5 (Investment FOC):
coef_adj = chi * (Gamma**2)

X0[4, 0] = 1.0
X0[4, 6] = -1.0
X0[4, 7] = -1.0
X0[4, 8] = coef_adj * (1.0 + beta)
X0[4, 2] = coef_adj
X1[4, 8] = coef_adj
X2[4, 8] = beta * coef_adj
X2[4, 2] = beta * coef_adj

# EQUATION 6 (Utilization):
X0[5, 4] = 1.0
X0[5, 9] = -zeta

# EQUATION 7 (Marginal cost):
X0[6, 10] = 1.0
X0[6, 4]  = -alpha
X0[6, 11] = -(1.0 - alpha)

# EQUATION 8 (Rental rate condition):
X0[7, 4]  = 1.0
X0[7, 11] = -1.0
X0[7, 12] = 1.0
X0[7, 13] = -1.0

# EQUATION 9 (Capital services identity):
X0[8, 12] = 1.0
X0[8, 9]  = -1.0
X0[8, 2]  = 1.0
X1[8, 14] = 1.0

# EQUATION 10 (Capital accumulation):
coef_kbar_lag = (1.0 - delta) / Gamma
coef_d = 1.0 - coef_kbar_lag

X0[9, 14] = 1.0
X0[9, 2]  = coef_kbar_lag
X0[9, 7]  = -coef_d
X0[9, 8]  = -coef_d
X1[9, 14] = coef_kbar_lag

# EQUATION 11 (Production):
X0[10, 15] = 1.0
X0[10, 12] = -alpha
X0[10, 13] = -(1.0 - alpha)

# EQUATION 12 (Goods-market clearing):
X0[11, 15] = (1.0 - psi)
X0[11, 1]  = -CY
X0[11, 8]  = -IY
X0[11, 9]  = -RkK_over_PY
X0[11, 16] = -1.0

# EQUATION 13 (Price Phillips curve):
kappa = ((1.0 - thetap * beta) * (1.0 - thetap)) / thetap

X0[12, 5]  = 1.0
X0[12, 10] = -kappa
X0[12, 17] = -kappa
X2[12, 5]  = beta

# EQUATION 14 (Wage setting / wage Phillips):
kappaw = ((1.0 - thetaw * beta) * (1.0 - thetaw)) / (thetaw * (1.0 + beta) * (1.0 + zeta * (1.0 + 1.0/muw)))
a0 = 1.0 / (1.0 + beta)
a1 = beta / (1.0 + beta)

X0[13, 11] = 1.0 + kappaw
X0[13, 13] = -(kappaw * zeta)
X0[13, 0]  = kappaw
X0[13, 18] = (kappaw**2)
X0[13, 5]  = a0
X0[13, 2]  = a0
X1[13, 11] = a0
X2[13, 11] = a1
X2[13, 5]  = a1
X2[13, 2]  = a1

# EQUATION 15 (News technology shock):
X0[14, 2]  = 1
X0[14, 20] = -1
X1[14, 25] = 1
X1[14, 20] = -1

# EQUATION 16 (Temporary technology shock):
X0[15, 20] = 1
X1[15, 20] = rho
X3[15, 3]  = np.sqrt(Sigmav)

# EQUATION 17 (Investment specific technology shock):
X0[16, 7] = 1
X1[16, 7] = rhod
X3[16, 6] = np.sqrt(Sigmad)

# EQUATION 18 (Price Markup Shock):
X0[17, 17] = 1
X0[17, 26] = -1
X1[17, 17] = rhop
X1[17, 26] = -psip

# EQUATION 19 (Wage Markup Shock):
X0[18, 18] = 1
X0[18, 21] = -1
X1[18, 18] = rhow
X1[18, 21] = -psiw

# EQUATION 20 (Government Spending Shock):
X0[19, 16] = 1
X1[19, 16] = rhog
X3[19, 1]  = np.sqrt(Sigmag)

# EQUATION 21 (Monetary Policy Shock):
X0[20, 19] = 1
X1[20, 19] = rhoq
X3[20, 0]  = np.sqrt(Sigmaq)

# EQUATIONS 22 AND 23 (Buffers for the MA Processes):
X0[21, 21] = 1
X0[22, 26] = 1
X3[21, 2]  = np.sqrt(Sigmaw)
X3[22, 5]  = np.sqrt(Sigmap)

# EQUATIONS 24,25,26 AND 27 (Buffers for the 4th Lag of the Tech Innovation Shock):
X0[23, 22] = 1
X0[24, 23] = 1
X0[25, 24] = 1
X0[26, 25] = 1
X1[24, 22] = 1
X1[25, 23] = 1
X1[26, 24] = 1
X3[23, 4]  = np.sqrt(Sigmae)

# === SOLVE =====

F, G, eigvals = solve_news_model(X0, X1, X2, X3)

np.savez(
    PROCESSED_DATA_DIR / "dsge_solution.npz",
    F=F,
    G=G
)
print(f"Saved: {PROCESSED_DATA_DIR / 'dsge_solution.npz'}")

# === VERIFICATION PLOT =====

T_horiz = 30
shock_idx = 4   # news shock (epst)
shock_size = 1.0

n = F.shape[0]
k = G.shape[1]

x_path = np.zeros((T_horiz + 1, n))
x = np.zeros(n)

u0 = np.zeros(k)
u0[shock_idx] = shock_size

x = F @ x + G @ u0
x_path[0, :] = x

for t in range(1, T_horiz + 1):
    x = F @ x
    x_path[t, :] = x

da   = x_path[:, 2]
chat = x_path[:, 1]
r    = x_path[:, 3]
pi   = x_path[:, 5]
ihat = x_path[:, 8]
nhrs = x_path[:, 13]
yhat = x_path[:, 15]

a = np.cumsum(da)
logC_irf = chat + a
logI_irf = ihat + a
logY_irf = yhat + a
logH_irf = nhrs

TFP_plot = 100.0 * a
C_plot   = 100.0 * logC_irf
I_plot   = 100.0 * logI_irf
Y_plot   = 100.0 * logY_irf
H_plot   = 100.0 * logH_irf
r_plot   = 100.0 * r
pi_plot  = 100.0 * pi

h = np.arange(T_horiz + 1)

fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
axes = axes.ravel()

axes[0].plot(h, TFP_plot);  axes[0].set_title("TFP");          axes[0].axhline(0, linewidth=0.8)
axes[1].plot(h, Y_plot);    axes[1].set_title("GDP");          axes[1].axhline(0, linewidth=0.8)
axes[2].plot(h, C_plot);    axes[2].set_title("Consumption");  axes[2].axhline(0, linewidth=0.8)
axes[3].plot(h, I_plot);    axes[3].set_title("Investment");   axes[3].axhline(0, linewidth=0.8)
axes[4].plot(h, H_plot);    axes[4].set_title("Hours");        axes[4].axhline(0, linewidth=0.8)
axes[5].plot(h, r_plot);    axes[5].set_title("Interest rate");axes[5].axhline(0, linewidth=0.8)
axes[6].plot(h, pi_plot);   axes[6].set_title("Inflation");    axes[6].axhline(0, linewidth=0.8)
axes[7].axis("off")

for ax in axes[:7]:
    ax.set_xlim(0, T_horiz)
    ax.tick_params(axis="both", labelsize=9)

axes[4].set_xlabel("quarters")
axes[5].set_xlabel("quarters")
axes[6].set_xlabel("quarters")

fig.tight_layout()
plt.show()
