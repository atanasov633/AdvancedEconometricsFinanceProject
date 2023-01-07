#packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import datetime as dt
import matplotlib.dates as mdates
import math
from scipy.stats import norm, t

#importing data
PATH = "/Users/davidmathas/Documents/ECONOMETRICS/Msc VAKKEN JAAR 1/Advanced Econometrics /Assignment 3/data_ass3.csv"
df = pd.read_csv(PATH)
                   #header=None, error_bad_lines=False)
df.describe()

# obtain dataframe for each ticker we are interested in and reset the index so that all start from index 1
df_JNJ = df[df["TICKER"] == "JNJ"]
df_JNJ.reset_index(drop=True, inplace=True)
df_KO = df[df["TICKER"] == "KO"]
df_KO.reset_index(drop=True, inplace=True)
df_PFE = df[df["TICKER"] == "PFE"]
df_PFE.reset_index(drop=True, inplace=True)
df_MRK = df[df["TICKER"] == "MRK"]
df_MRK.reset_index(drop=True, inplace=True)

list_dfs = [df_JNJ, df_KO, df_PFE, df_MRK]

#demeaning and perc the data
for i in list_dfs:
    y = np.mean(i['RET'])
    i['RET_PERCENT'] = i['RET'].apply(lambda x: (x - y) * 100)

#creating a new df for convenience
df_new = pd.DataFrame([df_JNJ['date'], df_JNJ['RET_PERCENT'], df_KO['RET_PERCENT'], df_MRK['RET_PERCENT'], df_PFE['RET_PERCENT']]).T
df_new.columns = ['Date', 'Ret JNJ', 'Ret KO', 'Ret MRK', 'Ret PFE']


################# QUESTION 1 #################

#FUNCTION WITH SIGMA_t-1 FIXED AT 1 (as suggested in question) for news impact curves

def upd_volFIXED(DATA_RET, alpha, beta, delta, lamb, omega):
    sigmaTminus = 1
    sigma = []
    for i in range(len(DATA_RET)):
        if DATA_RET[i] < 0:
            NUMERATOR = alpha*(DATA_RET[i]**2) + delta*(DATA_RET[i]**2)
        else:
            NUMERATOR = alpha*(DATA_RET[i]**2)

        DENOMINATOR = 1 + (DATA_RET[i]**2) / (lamb*sigmaTminus)

        sigma.append(omega + (NUMERATOR/DENOMINATOR) + beta*sigmaTminus)

    return sigma

#plotting the news impact curves
deltas = [0, 1, 0.2, 0.4]

for i in range(1,5):
    plt.subplot(2,2,i)
    plt.plot(np.arange(-2, 2.02, 0.02), upd_volFIXED(np.arange(-2, 2.02, 0.02), 0.05, 0.9, deltas[i-1], 2, 0), linewidth=0.4, label=f'delta={deltas[i-1]}, lambda=2')
    plt.plot(np.arange(-2, 2.02, 0.02), upd_volFIXED(np.arange(-2, 2.02, 0.02), 0.05, 0.9, deltas[i-1], 5, 0), linewidth=0.4, label=f'delta={deltas[i-1]}, lambda=5')
    plt.plot(np.arange(-2, 2.02, 0.02), upd_volFIXED(np.arange(-2, 2.02, 0.02), 0.05, 0.9, deltas[i-1], 10, 0), linewidth=0.4, label=f'delta={deltas[i-1]}, lambda=10')
    plt.plot(np.arange(-2, 2.02, 0.02), upd_volFIXED(np.arange(-2, 2.02, 0.02), 0.05, 0.9, deltas[i-1], 50, 0), linewidth=0.4, label=f'delta={deltas[i-1]}, lambda=50')
    plt.plot(np.arange(-2, 2.02, 0.02), upd_volFIXED(np.arange(-2, 2.02, 0.02), 0.05, 0.9, deltas[i-1], 5000, 0), linewidth=0.4, label=f'delta={deltas[i-1]}, lambda=Inf')
    plt.grid()
    plt.xlabel('Xt', fontsize=6)
    plt.xticks(np.arange(-2, 2.4, 0.4), fontsize=4)
    plt.yticks(fontsize=4)
    plt.ylabel('Response of updated volatility', fontsize=6)
    plt.title(f'News-impact curve fixed delta, different lambda values', fontsize=6)
    plt.legend(loc=1, prop={'size': 4})


plt.subplots_adjust(left=0.123,
                    bottom=0.055,
                    right=0.9,
                    top=0.962,
                    wspace=0.34,
                    hspace=0.290)

fig_q1_news = plt.gcf()

fig_q1_news.savefig("/Users/davidmathas/Documents/ECONOMETRICS/Msc VAKKEN JAAR 1/Advanced Econometrics /Assignment 3/TESThighres_plot_Q1_NEWS.png", dpi=300)

#plt.show()


################# QUESTION 2 #################

#create a df with only the returns for the descr table
df_only_returns = df_new.iloc[:, 1:]

#loop for table with statistics per stock
def stats_loop(df):
    col_names = ['JNJ', 'KO', 'MRK', 'PFE']
    descriptives_JNJ = []
    descriptives_KO = []
    descriptives_MRK = []
    descriptives_PFE = []
    list_desc = [descriptives_JNJ, descriptives_KO, descriptives_MRK, descriptives_PFE]

    for l in range(len(list_desc)):
            #list_desc[l].append(col_names[l])
            list_desc[l].append(round(df.iloc[:, l].count(), 3))
            list_desc[l].append(round(df.iloc[:, l].mean(), 3))
            list_desc[l].append(round(df.iloc[:, l].median(), 3))
            list_desc[l].append(round(df.iloc[:, l].std(), 3))
            list_desc[l].append(round(df.iloc[:, l].skew(), 3))
            list_desc[l].append(round(df.iloc[:, l].kurtosis(), 3))
            list_desc[l].append(round(df.iloc[:, l].min(), 3))
            list_desc[l].append(round(df.iloc[:, l].max(), 3))
    table1 = pd.DataFrame(list_desc)
    table1.columns = ['Count', 'Mean', 'Median', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max']
    table2 = table1.T
    table2.columns = col_names

    return pd.DataFrame(table2)

#output the table in latex format (code)
print(stats_loop(df_only_returns).to_latex())

#plot the returns
stocks = ['JNJ', 'KO', 'MRK', 'PFE']

for i in range(1,5):
    plt.subplot(2,2,i)
    plt.ylabel('Returns', fontsize=6)
    dates = df_new.iloc[:,0]
    x = [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in dates]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=850))
    plt.gcf().autofmt_xdate()
    plt.plot(x, df_new.iloc[:,i], color='k', linestyle='dashed', label=f'Returns {stocks[i-1]}', linewidth=0.1)
    plt.legend(loc=4, prop={'size': 4})
    plt.xlabel('Time', fontsize=6)
    plt.ylabel('Returns', fontsize=6)
    plt.title(f'Plot of returns for {stocks[i-1]}', fontsize=6)
    plt.grid(linestyle='--')
    plt.rcParams.update({'font.size': 5})
#plt.rcParams.update({'font.size': 5})

plt.subplots_adjust(left=0.123,
                    bottom=0.155,
                    right=0.9,
                    top=0.92,
                    wspace=0.24,
                    hspace=0.290)

#plt.show() - either plt.show() or plt.gcf() to save produces the plot in Latex

fig1_Q2_returns = plt.gcf()

fig1_Q2_returns.savefig("/Users/davidmathas/Documents/ECONOMETRICS/Msc VAKKEN JAAR 1/Advanced Econometrics /Assignment 3/fig_q2_returns.png", dpi=300)


################# QUESTION 3 #################

#building the with leverage likelihood function
def p_max_likelihood_with_leverage(value, df):
    omega = value[0]
    alpha = value[1]
    delta = value[2]
    lam = value[3]
    beta = value[4]
    initial_sig = value[5]
    #     T = len(df)

    #     c = 0.0001
    p_list = []  # list of log likelihoods
    initial_log = math.lgamma((lam + 1) / 2) - np.log(np.sqrt(lam * np.pi)) - math.lgamma(lam / 2) - (
                (lam + 1) / 2) * np.log(1 + (((df.iloc[0, 4] ** 2) / initial_sig) / lam))
    log_p = (initial_log - 0.5 * np.log(initial_sig))
    p_list.append(log_p)

    sigma_list = []  # open list for all sigma values
    sigma_list.append(initial_sig)  # initialise sigma list with the value given = 1
    for i in range(1, len(df)):  # loop through the data of returns
        if df.iloc[i - 1, 4] < 0:  # check condtion for the event A
            # a is the second sigma that we get from the equation
            a = ((alpha * (df.iloc[i - 1, 4]) ** 2) + delta * ((df.iloc[i - 1, 4]) ** 2) * 1) / (
                        1 + ((df.iloc[i - 1, 4]) ** 2 / (lam * sigma_list[i - 1]))) + omega + (beta * sigma_list[i - 1])
            sigma_list.append(a)  # we attach the sigma
            #             P - LOG LIKELIHOOD
            p = math.lgamma((lam + 1) / 2) - np.log(np.sqrt(lam * np.pi)) - math.lgamma(lam / 2) - (
                        (lam + 1) / 2) * np.log(1 + (((df.iloc[i, 4] ** 2) / a) / lam))
            #             LIST_LIKELIHOOD. APPEND ( p )
            log_p = (p - (0.5 * np.log(a)))
            p_list.append(log_p)
        else:
            # b is the second sigma that we get from the equation when the case is not met
            b = ((alpha * (df.iloc[i - 1, 4]) ** 2) + delta * ((df.iloc[i - 1, 4]) ** 2) * 0) / (
                        1 + ((df.iloc[i - 1, 4]) ** 2 / (lam * sigma_list[i - 1]))) + omega + (beta * sigma_list[i - 1])
            sigma_list.append(b)  # we attach the sigma
            #             P - LOG LIKELIHOOD
            p = math.lgamma((lam + 1) / 2) - np.log(np.sqrt(lam * np.pi)) - math.lgamma(lam / 2) - (
                        (lam + 1) / 2) * np.log(1 + (((df.iloc[i, 4] ** 2) / b) / lam))
            #             LIST_LIKELIHOOD. APPEND ( p )
            log_p = (p - (0.5 * np.log(b)))
            p_list.append(log_p)

    return ((-1) * sum(p_list)) / len(df)


##Creating our list foroptimization TO CHECK WITH HINT FILE LEVERAGE

df_KO_new = df_KO[:2500]

initial_sig_Q2 = np.var(df_new['Ret KO'][:50]) #defining sigma initialization for KO returns


values_with_leverage = [0.007442577, 0.02670845,0.11266453,6.056328, 0.9217446, initial_sig_Q2]
values_with_leverage

##Optimization for Leverage Effect, also CHECKING WITH HINT FILE

options ={'eps':1e-09,  # argument convergence criteria
         'disp': False,  # display iterations
         'maxiter': 6000} # maximum number of iterations

log_likelihood_with_leverage = scipy.optimize.minimize(p_max_likelihood_with_leverage, values_with_leverage,
                                                       args=df_KO_new,
                                                       options=options,
                                                       method='SLSQP')

print('parameter estimates:')
print(log_likelihood_with_leverage.x)

print('log likelihood value:')
print(log_likelihood_with_leverage.fun)

print('Successful minimization?')
print(log_likelihood_with_leverage.success)


#building the NO leverage likelihood function
def p_max_likelihood_without_leverage(value, df):
    omega = value[0]
    alpha = value[1]
    lam = value[2]
    beta = value[3]
    initial_sig = value[4]

    p_list = []  # list of log likelihoods
    initial_log = math.lgamma((lam + 1) / 2) - np.log(np.sqrt(lam * np.pi)) - math.lgamma(lam / 2) - (
                (lam + 1) / 2) * np.log(1 + (((df.iloc[0, 4] ** 2) / initial_sig) / lam))
    log_p = (initial_log - 0.5 * np.log(initial_sig))
    p_list.append(log_p)

    sigma_list = []  # open list for all sigma values
    sigma_list.append(initial_sig)  # initialise sigma list with the value given = 1
    for i in range(1, len(df)):  # loop through the data of returns
        # a is the second sigma that we get from the equation
        a = (alpha * (df.iloc[i - 1, 4] ** 2)) / (
                    1 + ((df.iloc[i - 1, 4]) ** 2 / (lam * sigma_list[i - 1]))) + omega + (beta * sigma_list[i - 1])
        sigma_list.append(a)  # we attach the sigma
        #             P - LOG LIKELIHOOD
        p = math.lgamma((lam + 1) / 2) - np.log(np.sqrt(lam * np.pi)) - math.lgamma(lam / 2) - ((lam + 1) / 2) * np.log(
            1 + (((df.iloc[i, 4] ** 2) / a) / lam))
        #             LIST_LIKELIHOOD. APPEND ( p )
        log_p = (p - (0.5 * np.log(a)))
        p_list.append(log_p)

    #     RETURN LIST_LIKELIHOOD --- TOTAL
    return ((-1) * sum(p_list)) / len(df)
#     return sigma_list

# CHECKING WITH HINT FILE
values_without_leverage = [0.005665683,0.09922153,5.5415955,0.91055318 ,initial_sig_Q2]
values_without_leverage

log_likelihood_without_leverage = scipy.optimize.minimize(p_max_likelihood_without_leverage, values_without_leverage,
                                                          args=df_KO_new,
                                                          options=options,
                                                          method='SLSQP')

print('parameter estimates:')
print(log_likelihood_without_leverage.x)

print('log likelihood value:')
print(log_likelihood_without_leverage.fun)

print('Successful minimization?')
print(log_likelihood_without_leverage.success)

# Building the Hessian matrix and step-wise function
values_without_leverage = [0.005665683,0.09922153,5.5415955,0.91055318 ,initial_sig_Q2]
np_values = np.array(values_without_leverage)

#number of rows
iT = 2500

def _gh_stepsize(vP):
    vh = 1e-8*(np.fabs(vP)+1e-8)   # Find stepsize
    vh= np.maximum(vh, 5e-6)       # Don't go too small

    return vh

def hessian_2sided(fun, vP, *args):
    """
    Purpose:A
      Compute numerical hessian, using a 2-sided numerical difference

    Author:
      Kevin Sheppard, adapted by Charles Bos

    Source:
      https://www.kevinsheppard.com/Python_for_Econometrics

    Inputs:
      fun     function, as used for minimize()
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mH      iP x iP matrix with symmetric hessian
    """
    iP = np.size(vP,0)
    vP= vP.reshape(iP)    # Ensure vP is 1D-array

    f = fun(vP, *args)
    vh= _gh_stepsize(vP)
    vPh = vP + vh
    vh = vPh - vP

    mh = np.diag(vh)            # Build a diagonal matrix out of vh

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    fpp = np.zeros((iP,iP))
    fmm = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            fpp[i,j] = fun(vP + mh[i] + mh[j], *args)    # obtains the errors from the diagonal
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(vP - mh[i] - mh[j], *args)
            fmm[j,i] = fmm[i,j]

    vh = vh.reshape((iP,1))
    mhh = vh @ vh.T             # mhh= h h', outer product of h-vector

    mH = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i,j])/mhh[i,j]/2
            mH[j,i] = mH[i,j]

    return mH

#After calculating the total likelihood function, we need to change it to return the average likelihood so that we
#can use it in the Hassian

def compute_std_errors(par, iT, avg_lik, *args):
    ## compute standard errors

    H = hessian_2sided(avg_lik, par, *args)  # we re-write the parameters according to our hassian function

    Hinv = np.linalg.inv(H)  ###inverse Hessian based

    return [np.sqrt(np.diag(Hinv) / iT)]


######### optimization Pfizer with leverage and without leverage
df_PFE
df_PFE_new = df_PFE[:2500]
initial_sig_PFE = np.var(df_PFE_new.iloc[:50, 4])
omega_PFE = np.var(df_PFE_new.iloc[:, 4])/50

##List for Pfizer with leverage
params_PFE_lev = [omega_PFE, 0.02, 0, 5, 0.96, initial_sig_PFE]
## LIST for Pfizer without Leverage
params_PFE_WITHOUT_lev = [omega_PFE, 0.02, 5, 0.96, initial_sig_PFE]

#with leverage PFI
log_likelihood_with_leverage_PFE = scipy.optimize.minimize(p_max_likelihood_with_leverage, params_PFE_lev,
                                                           args=df_PFE_new,
                                                           options=options,
                                                           method='SLSQP')

print('parameter estimates for Pfizer with leverage:')
print(log_likelihood_with_leverage_PFE.x)

print('avg log likelihood value for Pfizer with leverage:')
print(log_likelihood_with_leverage_PFE.fun)

print('Successful minimization?')
print(log_likelihood_with_leverage_PFE.success)


## Creating an array of parameters with the ones obtained from the optimization
PFE_with_lev = np.array([0.01385358, 0.02823194, 0.11146147, 6.01015225, 0.91970488, 3.92465354])

#Pfizer with leverage hessian
compute_std_errors(PFE_with_lev, iT, p_max_likelihood_with_leverage, df_PFE_new)


#Pfizer optimization without leverage
log_likelihood_without_leverage_PFE = scipy.optimize.minimize(p_max_likelihood_without_leverage, params_PFE_WITHOUT_lev,
                                                              args=df_PFE_new,
                                                              options=options,
                                                              method='SLSQP')

print('parameter estimates for Pfizer without leverage:')
print(log_likelihood_without_leverage_PFE.x)

print('avg log likelihood value for Pfizer without leverage:')
print(log_likelihood_without_leverage_PFE.fun)

print('Successful minimization?')
print(log_likelihood_without_leverage_PFE.success)

## Creating an array of parameters with the ones obtained from the optimization
PFE_without_lev = np.array([0.02905731, 0.11871423, 6.19213681, 0.88166099, 3.99853266])

#hessian matrix PFI no leverage
compute_std_errors(PFE_without_lev, iT, p_max_likelihood_without_leverage, df_PFE_new)


##### optimization for Johhnson and Johnson with leverage and no leverage

##List for J&J with leverage
params_JNJ_lev = [omega_JNJ, 0.02, 0, 5, 0.96, initial_sig_JNJ]
## LIST for J&J without Leverage
params_JNJ_WITHOUT_lev = [omega_JNJ, 0.02, 5, 0.96, initial_sig_JNJ]

df_JNJ_new = df_JNJ[:2500]
initial_sig_JNJ = np.var(df_JNJ_new.iloc[:50, 4])
omega_JNJ = np.var(df_JNJ_new.iloc[:, 4])/50


#JNJ optimization with leverage
log_likelihood_with_leverage_JNJ = scipy.optimize.minimize(p_max_likelihood_with_leverage, params_JNJ_lev,
                                                           args=df_JNJ_new,
                                                           options=options,
                                                           method='SLSQP')

print('parameter estimates for J&J with leverage:')
print(log_likelihood_with_leverage_JNJ.x)

print('avg log likelihood value for J&J with leverage:')
print(log_likelihood_with_leverage_JNJ.fun)

print('Successful minimization?')
print(log_likelihood_with_leverage_JNJ.success)

## Creating an array of parameters with the ones obtained from the optimization
JNJ_with_lev = np.array([0.01339828, 0.04123331, 0.16762957, 6.40251796, 0.8772484,  2.68102568])

#hessian JNJ with leverage
compute_std_errors(JNJ_with_lev, iT, p_max_likelihood_with_leverage, df_JNJ_new)

#JNJ optimization without leverage
log_likelihood_without_leverage_JNJ = scipy.optimize.minimize(p_max_likelihood_without_leverage, params_JNJ_WITHOUT_lev,
                                                              args=df_JNJ_new,
                                                              options=options,
                                                              method='SLSQP')

print('parameter estimates for J&J without leverage:')
print(log_likelihood_without_leverage_JNJ.x)

print('avg log likelihood value for J&J without leverage:')
print(log_likelihood_without_leverage_JNJ.fun)

print('Successful minimization?')
print(log_likelihood_without_leverage_JNJ.success)

## Creating an array of parameters with the ones obtained from the optimization
JNJ_without_lev = np.array([0.01372416, 0.15548189, 6.01059734, 0.85226852, 2.76944672])

#JNJ Hessian without leverage
compute_std_errors(JNJ_without_lev, iT, p_max_likelihood_without_leverage, df_JNJ_new)

#### optimization Merck with and without leverage
df_MRK_new = df_MRK[:2500]

initial_sig_MRK = np.var(df_MRK_new.iloc[:50, 4])
omega_MRK = np.var(df_MRK_new.iloc[:, 4])/50

##List for merck with leverage
params_MRK_lev = [omega_MRK, 0.02, 0, 5, 0.96, initial_sig_MRK]
## LIST for merck without Leverage
params_MRK_WITHOUT_lev = [omega_MRK, 0.02, 5, 0.96, initial_sig_MRK]


#MRK optimization with leverage
log_likelihood_with_leverage_MRK = scipy.optimize.minimize(p_max_likelihood_with_leverage, params_MRK_lev,
                                                           args=df_MRK_new,
                                                           options=options,
                                                           method='SLSQP')

print('parameter estimates for MRK with leverage:')
print(log_likelihood_with_leverage_MRK.x)

print('avg log likelihood value for MRK with leverage:')
print(log_likelihood_with_leverage_MRK.fun)

print('Successful minimization?')
print(log_likelihood_with_leverage_MRK.success)

## Creating an array of parameters with the ones obtained from the optimization
MRK_with_lev = np.array([0.03530676, 0.04176289, 0.15542425, 4.38347144, 0.88301761, 3.26788176])

#Merck hessian with leverage
compute_std_errors(MRK_with_lev, iT, p_max_likelihood_with_leverage, df_MRK_new)


#merck optimization without leverage
log_likelihood_without_leverage_MRK = scipy.optimize.minimize(p_max_likelihood_without_leverage, params_MRK_WITHOUT_lev,
                                                              args=df_MRK_new,
                                                              options=options,
                                                              method='SLSQP')

print('parameter estimates for MRK without leverage:')
print(log_likelihood_without_leverage_MRK.x)

print('avg log likelihood value for MRK without leverage:')
print(log_likelihood_without_leverage_MRK.fun)

print('Successful minimization?')
print(log_likelihood_without_leverage_MRK.success)

## Creating an array of parameters with the ones obtained from the optimization
MRK_without_lev = np.array([0.04963202, 0.16335967, 4.37566884, 0.83875601, 3.31127253])

#hessian merck without leverage
compute_std_errors(MRK_without_lev, iT, p_max_likelihood_without_leverage, df_MRK_new)

##### AIC and BIC
# Order: PFI, JNJ, MRK
#averge log likelihood lists
list_likelihood_with_lev = [1.812701893826533, 1.4196527089892694, 1.8683790092238712]
list_likelihood_without_lev = [1.8169156811616947, 1.4269208877813107, 1.874006698335025]

"""Obtain AIC and BIC values for each company WITH leverage effect"""
AIC_list_with = []
BIC_list_with = []
n = len(df_JNJ)
for i in list_likelihood_with_lev:
    AIC_list_with.append(2 * 5 - 2 * i)
    BIC_list_with.append(np.log(n) * 5 - 2 **i)

# Order: PFI, JNJ, MRK
AIC_list_with

# Order: PFI, JNJ, MRK
BIC_list_with

"""Obtain AIC and BIC values for each company WITHOUT leverage effect"""
AIC_list_without = []
BIC_list_without = []
n = len(df_JNJ)
for i in list_likelihood_without_lev:
    AIC_list_without.append(2 * 4 - 2 *i)
    BIC_list_without.append(np.log(n) * 4 - 2 *i)

# Order: PFI, JNJ, MRK
AIC_list_without

# Order: PFI, JNJ, MRK
BIC_list_without

################# QUESTION 4 #################

#all the parameter estimates in one spot:
list_with_leverage_PFI = [0.01385358, 0.02823194, 0.11146147, 6.01015225, 0.91970488]
list_with_leverage_JNJ = [0.01339828, 0.04123331, 0.16762957, 6.40251796, 0.8772484]
list_with_leverage_MRK = [0.03530676, 0.04176289, 0.15542425, 4.38347144, 0.88301761]

list_no_leverage_PFI = [0.02905731, 0.11871423, 6.19213681, 0.88166099]
list_no_leverage_JNJ = [0.01372416, 0.15548189, 6.01059734, 0.85226852]
list_no_leverage_MRK = [0.04963202, 0.16335967 ,4.37566884, 0.83875601]

parameter_list = [list_with_leverage_PFI, list_no_leverage_PFI, list_with_leverage_JNJ, list_no_leverage_JNJ, list_with_leverage_MRK, list_no_leverage_MRK]
names_q4 = ['PFE with lev', 'PFE without lev', 'JNJ with lev', 'JNJ without lev', 'MRK with lev', 'MRK without lev']


# rewrite vol function for delta is not zero models

def upd_vol_DELTA(DATA_RET, vector_delta):
    omega = vector_delta[0]
    alpha = vector_delta[1]
    delta = vector_delta[2]
    lamb = vector_delta[3]
    beta = vector_delta[4]

    sigmaTminus = 1
    sigma = []
    for i in range(len(DATA_RET)):
        if DATA_RET[i] < 0:
            NUMERATOR = alpha*(DATA_RET[i]**2) + delta*(DATA_RET[i]**2)
        else:
            NUMERATOR = alpha*(DATA_RET[i]**2)

        DENOMINATOR = 1 + (DATA_RET[i]**2) / (lamb*sigmaTminus)

        sigma.append(omega + (NUMERATOR/DENOMINATOR) + beta*sigmaTminus)

    return sigma


# rewrite vol function for delta is not zero models FOR FILT_VOL

def FILT_upd_vol_DELTA(DATA_RET, vector_delta):
    omega = vector_delta[0]
    alpha = vector_delta[1]
    delta = vector_delta[2]
    lamb = vector_delta[3]
    beta = vector_delta[4]

    SIGMA_ini = np.var(DATA_RET[:50])
    sigma = [SIGMA_ini]

    for i in range(len(DATA_RET)):
        if DATA_RET[i] < 0:
            NUMERATOR = alpha*(DATA_RET[i]**2) + delta*(DATA_RET[i]**2)
        else:
            NUMERATOR = alpha*(DATA_RET[i]**2)

        DENOMINATOR = 1 + (DATA_RET[i]**2) / (lamb*sigma[i])

        sigma.append(omega + (NUMERATOR/DENOMINATOR) + beta*sigma[i])

    return sigma[:-1]


# rewrite vol function for delta is zero

def upd_vol_NO_DELTA(DATA_RET, vector_ndelta):
    omega = vector_ndelta[0]
    alpha = vector_ndelta[1]
    lamb = vector_ndelta[2]
    beta = vector_ndelta[3]

    sigmaTminus = 1
    sigma = []
    for i in range(len(DATA_RET)):
        NUMERATOR = alpha*(DATA_RET[i]**2)

        DENOMINATOR = 1 + (DATA_RET[i]**2) / (lamb*sigmaTminus)

        sigma.append(omega + (NUMERATOR/DENOMINATOR) + beta*sigmaTminus)

    return sigma

# rewrite vol function for delta is zero FOR FILT VOL

def FILT_upd_vol_NO_DELTA(DATA_RET, vector_ndelta):
    omega = vector_ndelta[0]
    alpha = vector_ndelta[1]
    lamb = vector_ndelta[2]
    beta = vector_ndelta[3]

    SIGMA_ini = np.var(DATA_RET[:50])
    sigma = [SIGMA_ini]
    for i in range(len(DATA_RET)):
        NUMERATOR = alpha*(DATA_RET[i]**2)

        DENOMINATOR = 1 + (DATA_RET[i]**2) / (lamb*sigma[i])

        sigma.append(omega + (NUMERATOR/DENOMINATOR) + beta*sigma[i])

    return sigma[:-1]


# plotting the news impact curves

for i in range(1,7):
    plt.subplot(3,2,i)
    if i%2 == 0: #check if the plot should use no lev ==> here its odd
        plt.plot(np.arange(-2, 2.02, 0.02), upd_vol_NO_DELTA(np.arange(-2, 2.02, 0.02), parameter_list[i-1]), linewidth=0.8, label=f'{names_q4[i-1]}', color='g')
    else:
        plt.plot(np.arange(-2, 2.02, 0.02), upd_vol_DELTA(np.arange(-2, 2.02, 0.02), parameter_list[i-1]), linewidth=0.8, label=f'{names_q4[i-1]}', color='r')

    plt.grid()
    if i == 5 or i == 6:
        plt.xlabel('Xt', fontsize=6)
    plt.xticks(np.arange(-2, 2.4, 0.4), fontsize=4)
    plt.yticks(np.arange(0.8, 1.4, 0.1), fontsize=4)
    plt.ylabel('Response of updated volatility', fontsize=4)
    plt.title(f'{names_q4[i-1]} parameters', fontsize=6)
    plt.legend(loc=1, prop={'size': 4})


plt.subplots_adjust(left=0.123,
                    bottom=0.095,
                    right=0.9,
                    top=0.962,
                    wspace=0.34,
                    hspace=0.290)


#plt.show() - one of two methods produces the latex plot

fig1_Q4_NEWS = plt.gcf()

fig1_Q4_NEWS.savefig("/Users/davidmathas/Documents/ECONOMETRICS/Msc VAKKEN JAAR 1/Advanced Econometrics /Assignment 3/fig_q4_NEWS.png", dpi=300)




# plotting the filtered volatilites

#returns df to get the for loop working
RETURNS_q4 = pd.DataFrame([df_only_returns.iloc[:,3], df_only_returns.iloc[:,3], df_only_returns.iloc[:,0], df_only_returns.iloc[:,0], df_only_returns.iloc[:,2], df_only_returns.iloc[:,2]]).T


#Used this function:

for i in range(1,7):
    plt.subplot(3,2,i)

    dates = df_new.iloc[:,0]
    x = [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in dates]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=650))
    plt.gcf().autofmt_xdate()

    if i%2 == 0: #check if the plot should use no lev ==> here its odd
        plt.plot(x, FILT_upd_vol_NO_DELTA(RETURNS_q4.iloc[:,i-1], parameter_list[i-1]), linewidth=0.25, label=f'Filtered vol {names_q4[i-1]}', color='k')
    else:
        plt.plot(x, FILT_upd_vol_DELTA(RETURNS_q4.iloc[:,i-1], parameter_list[i-1]), linewidth=0.25, label=f'Filtered vol {names_q4[i-1]}', color='r')

    #vertical line at end of in sample observations
    plt.axvline(dt.datetime(2010, 12, 9), color='b', label='End of in-sample observations', linewidth=0.7, linestyle='--')



    plt.grid()
    plt.xlabel('Xt', fontsize=6)
    plt.xticks(fontsize=4)
    plt.yticks(np.arange(0,25, 5), fontsize=4)
    plt.ylabel('Filtered volatility size', fontsize=6)
    plt.title(f'{names_q4[i-1]} filtered volatility', fontsize=6)
    plt.legend(loc=1, prop={'size': 4})


plt.subplots_adjust(left=0.123,
                    bottom=0.070,
                    right=0.9,
                    top=0.962,
                    wspace=0.34,
                    hspace=0.290)

plt.subplots_adjust(left=0.123,
                    bottom=0.070,
                    right=0.9,
                    top=0.962,
                    wspace=0.34,
                    hspace=0.290)

plt.show()

fig1_Q4_FILT_VOL = plt.gcf()

fig1_Q4_FILT_VOL.savefig("/Users/davidmathas/Documents/ECONOMETRICS/Msc VAKKEN JAAR 1/Advanced Econometrics /Assignment 3/fig_q4_FILT_VOL.png", dpi=300)



################# QUESTION 5 #################

#creating the new dfs for the new time period
#RUN PART OF THE SCRIPT (line by line) for every stock and change 'value' and 'value without lev'

print(df_JNJ.loc[df_JNJ['date'] == '1/4/2020'])
df_JNJ_April = df_JNJ[:4842]
df_MRK_April = df_MRK[:4842]
df_PFE_April = df_PFE[:4842]

#values
values = [0.01339939, 0.04123968, 0.16755291, 6.40646062, 0.87725846, initial_sig_JNJ] #estimates JNJ
values = [0.03531633, 0.04177188, 0.15546195, 4.38115885, 0.88299542, initial_sig_MRK] #estimate MRK
values = [0.01386286, 0.02828159, 0.11148409, 6.01089225, 0.91964496, initial_sig_PFE] #estimate PFE


def sigma_with_leverage_Q5(value, df):
    omega = value[0]
    alpha = value[1]
    delta = value[2]
    lam = value[3]
    beta = value[4]
    initial_sig = value[5]
    #     T = len(df)

    #     c = 0.0001
    p_list = []  # list of log likelihoods
    initial_log = math.lgamma((lam + 1) / 2) - np.log(np.sqrt(lam * np.pi)) - math.lgamma(lam / 2) - (
            (lam + 1) / 2) * np.log(1 + (((df.iloc[0, 4] ** 2) / initial_sig) / lam))
    log_p = (initial_log - 0.5 * np.log(initial_sig))
    p_list.append(log_p)

    sigma_list = [initial_sig]  # open list for all sigma values
    for i in range(1, len(df)):  # loop through the data of returns
        if df.iloc[i - 1, 4] < 0:  # check condition for the event A
            # a is the second sigma that we get from the equation
            a = ((alpha * (df.iloc[i - 1, 4]) ** 2) + delta * ((df.iloc[i - 1, 4]) ** 2) * 1) / (
                    1 + ((df.iloc[i - 1, 4]) ** 2 / (lam * sigma_list[i - 1]))) + omega + (beta * sigma_list[i - 1])
            sigma_list.append(a)  # we attach the sigma
            #             P - LOG LIKELIHOOD
            p = math.lgamma((lam + 1) / 2) - np.log(np.sqrt(lam * np.pi)) - math.lgamma(lam / 2) - (
                    (lam + 1) / 2) * np.log(1 + (((df.iloc[i, 4] ** 2) / a) / lam))
            #             LIST_LIKELIHOOD. APPEND ( p )
            log_p = (p - (0.5 * np.log(a)))
            p_list.append(log_p)
        else:
            # b is the second sigma that we get from the equation when the case is not met
            b = ((alpha * (df.iloc[i - 1, 4]) ** 2) + delta * ((df.iloc[i - 1, 4]) ** 2) * 0) / (
                    1 + ((df.iloc[i - 1, 4]) ** 2 / (lam * sigma_list[i - 1]))) + omega + (beta * sigma_list[i - 1])
            sigma_list.append(b)  # we attach the sigma
            #             P - LOG LIKELIHOOD
            p = math.lgamma((lam + 1) / 2) - np.log(np.sqrt(lam * np.pi)) - math.lgamma(lam / 2) - (
                    (lam + 1) / 2) * np.log(1 + (((df.iloc[i, 4] ** 2) / b) / lam))
            #             LIST_LIKELIHOOD. APPEND ( p )
            log_p = (p - (0.5 * np.log(b)))
            p_list.append(log_p)

    return sigma_list[-1]

#new sigma values for Q5
values_with_lev = [0.01339939, 0.04123968, 0.16755291, 6.40646062, 0.87725846, 15.408171002829349, -1.808732] #JNJ
values_with_lev = [0.03531633, 0.04177188, 0.15546195, 4.38115885, 0.88299542, 8.74446087717981, -4.10718] #MRK
values_with_lev = [0.01386286, 0.02828159, 0.11148409, 6.01089225, 0.91964496, 10.42101586752793, -2.757989] #PFE


# Forecast X and Volatility WITH LEVERAGE

def matrix_sigma_Q5(value, d, iterr):
    """
    This function returns the return values WITHOUT compounding

    """

    omega = value[0]
    alpha = value[1]
    delta = value[2]
    lam = value[3]
    beta = value[4]
    initial_sig = value[5]
    init_x = value[6]

    df = pd.DataFrame()  # make a DF that stores all the c

    for i in range(0, iterr):  # loops through all iterations / rows /

        x = [init_x]  # appends the initial value to the list
        sig = [initial_sig]  # append a list that keeps the compounded values

        for t in range(1, d):  # loop through the data of returns for the observed horizon

            if x[t - 1] < 0:  # check condition for the event A
                sig_val = ((alpha * x[t - 1] ** 2) + delta * (x[t - 1] ** 2) * 1) / (
                        1 + (x[t - 1] ** 2 / (lam * sig[t - 1]))) + omega + (beta * sig[t - 1])
                sig.append(sig_val)
                x_val = np.sqrt(sig_val) * (np.random.standard_t(lam))  # generate x(t) recursively
                x.append(x_val)

            else:
                sig_val = ((alpha * x[t - 1] ** 2) + delta * (x[t - 1] ** 2) * 0) / (
                        1 + (x[t - 1] ** 2 / (lam * sig[t - 1]))) + omega + (beta * sig[t - 1])
                sig.append(sig_val)
                x_val = np.sqrt(sig_val) * (np.random.standard_t(lam))  # generate x(t) recursively
                x.append(x_val)

        df['Iteration ' + str(i)] = x  # get the new column for Iterration i

    df = df.T
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    return df



a = matrix_sigma_Q5(values_with_lev, 21, 10000)
# print(a)

comp_ret1 = ((1 + a.iloc[:, 0] / 100) - 1) * 100

comp_ret5 = ((1 + a.iloc[:, 0] / 100) * (1 + a.iloc[:, 1] / 100) * (1 + a.iloc[:, 2] / 100) *
             (1 + a.iloc[:, 3] / 100) * (1 + a.iloc[:, 4] / 100) * (1 + a.iloc[:, 5] / 100) - 1) * 100

comp_ret20 = ((1 + a.iloc[:, 0] / 100) * (1 + a.iloc[:, 1] / 100) * (1 + a.iloc[:, 2] / 100) *
              (1 + a.iloc[:, 3] / 100) * (1 + a.iloc[:, 4] / 100) * (1 + a.iloc[:, 5] / 100) *
              (1 + a.iloc[:, 6] / 100) * (1 + a.iloc[:, 7] / 100) * (1 + a.iloc[:, 8] / 100) *
              (1 + a.iloc[:, 9] / 100) * (1 + a.iloc[:, 10] / 100) * (1 + a.iloc[:, 11] / 100) *
              (1 + a.iloc[:, 12] / 100) * (1 + a.iloc[:, 13] / 100) * (1 + a.iloc[:, 14] / 100) *
              (1 + a.iloc[:, 15] / 100) * (1 + a.iloc[:, 16] / 100) * (1 + a.iloc[:, 17] / 100) *
              (1 + a.iloc[:, 18] / 100) * (1 + a.iloc[:, 19] / 100) - 1) * 100

print('Results with leverage')
# with 0.05 VaR

print(np.quantile(comp_ret1, 0.05))  # h=1

print(np.quantile(comp_ret5, 0.05))  # h=5

print(np.quantile(comp_ret20, 0.05))  # h=20

# with 0.01 VaR

print(np.quantile(comp_ret1, 0.01))  # h=1

print(np.quantile(comp_ret5, 0.01))  # h=5

print(np.quantile(comp_ret20, 0.01))  # h=20

# with 0.1 VaR

print(np.quantile(comp_ret1, 0.1))  # h=1

print(np.quantile(comp_ret5, 0.1))  # h=5

print(np.quantile(comp_ret20, 0.1))  # h=20

## FORECAST WITHOUT LEVERAGE

values_without_lev = [0.01355837, 0.1552567, 5.99413356, 0.85269778, 15.408171002829349, -1.808732] #JNJ
values_without_lev = [0.04962759, 0.1632808,  4.37738981, 0.83881665, 8.74446087717981, -4.10718] #MRK
values_without_lev = [0.02905228, 0.11871743, 6.1941698, 0.88166249, 10.42101586752793, -2.757989] #PFE


def matrix_sigma_no_lev(value, d, iterr):
    """
    This function returns the return values WITHOUT compounding

    """

    omega = value[0]
    alpha = value[1]
    lam = value[2]
    beta = value[3]
    initial_sig = value[4]
    init_x = value[5]

    df = pd.DataFrame()  # make a DF that stores all the c

    for i in range(0, iterr):  # loops through all iterations / rows /

        x = [init_x]  # appends the initial value to the list
        sig = [initial_sig]  # append a list that keeps the compounded values

        for t in range(1, d):  # loop through the data of returns for the observed horizon

            sig_val = (alpha * x[t - 1] ** 2) / (1 + (x[t - 1] ** 2 / (lam * sig[t - 1]))) + omega + (
                    beta * sig[t - 1])
            sig.append(sig_val)
            x_val = np.sqrt(sig_val) * (np.random.standard_t(lam))  # generate x(t) recursively
            x.append(x_val)

        df['Iteration ' + str(i)] = x  # get the new column for Iteration i

    df = df.T
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    return df


b = matrix_sigma_no_lev(values_without_lev, 21, 10000)
# days in input should be T + 1 because our for loop starts from 1

comp_ret_without_lev1 = ((1 + b.iloc[:, 0] / 100) - 1) * 100
comp_ret_without_lev5 = ((1 + b.iloc[:, 0] / 100) * (1 + b.iloc[:, 1] / 100) * (1 + b.iloc[:, 2] / 100) *
                         (1 + b.iloc[:, 3] / 100) * (1 + b.iloc[:, 4] / 100) - 1) * 100
comp_ret_without_lev_20 = ((1 + b.iloc[:, 0] / 100) * (1 + b.iloc[:, 1] / 100) * (1 + b.iloc[:, 2] / 100) *
                           (1 + b.iloc[:, 3] / 100) * (1 + b.iloc[:, 4] / 100) * (1 + b.iloc[:, 5] / 100) *
                           (1 + b.iloc[:, 6] / 100) * (1 + b.iloc[:, 7] / 100) * (1 + b.iloc[:, 8] / 100) *
                           (1 + b.iloc[:, 9] / 100) * (1 + b.iloc[:, 10] / 100) * (1 + b.iloc[:, 11] / 100) *
                           (1 + b.iloc[:, 12] / 100) * (1 + b.iloc[:, 13] / 100) * (1 + b.iloc[:, 14] / 100) *
                           (1 + b.iloc[:, 15] / 100) * (1 + b.iloc[:, 16] / 100) * (1 + b.iloc[:, 17] / 100) *
                           (1 + b.iloc[:, 18] / 100) * (1 + b.iloc[:, 19] / 100) - 1) * 100

print('Results without leverage')
# with 0.05 VaR

print(np.quantile(comp_ret_without_lev1, 0.05))  # h=1

print(np.quantile(comp_ret_without_lev5, 0.05))  # h=5

print(np.quantile(comp_ret_without_lev_20, 0.05))  # h=20

# with 0.01 VaR

print(np.quantile(comp_ret_without_lev1, 0.01))  # h=1

print(np.quantile(comp_ret_without_lev5, 0.01))  # h=5

print(np.quantile(comp_ret_without_lev_20, 0.01))  # h=20

# with 0.1 VaR

print(np.quantile(comp_ret_without_lev1, 0.1))  # h=1

print(np.quantile(comp_ret_without_lev5, 0.1))  # h=5

print(np.quantile(comp_ret_without_lev_20, 0.1))  # h=20


################ QUESION 6 ##################

mean_PFE = df_PFE['RET'].mean()
transformed_mean_PFE = 100 * (df_PFE['RET'] - mean_PFE)
df_PFE['transformed_mean'] = transformed_mean_PFE

mean_JNJ = df_JNJ['RET'].mean()
transformed_mean_JNJ = 100 * (df_JNJ['RET'] - mean_JNJ)
df_JNJ['transformed_mean'] = transformed_mean_JNJ

mean_MRK = df_MRK['RET'].mean()
transformed_mean_MRK = 100 * (df_MRK['RET'] - mean_MRK)
df_MRK['transformed_mean'] = transformed_mean_MRK

parameters_PFE_with_lev = [0.01386286, 0.02828159, 0.11148409, 6.01089225, 0.91964496, 3.92457348]
parameters_MRK_with_lev = [0.03531633, 0.04177188, 0.15546195, 4.38115885, 0.88299542, 3.2570950378440005]
parameter_JNJ_with_lev = [0.01339939, 0.04123968, 0.16755291, 6.40646062, 0.87725846, 2.628335991824]

PFE_without_lev = [0.02905731, 0.11871423, 6.19213681, 0.88166099, 3.99853266]
JNJ_without_lev = [0.01372416, 0.15548189, 6.01059734, 0.85226852, 2.76944672]
MRK_without_lev = [0.04963202, 0.16335967, 4.37566884, 0.83875601, 3.31127253]
#funtion to be sure
def sigma_leverage_Q6_with_lev(value, df):
    omega = value[0]
    alpha = value[1]
    delta = value[2]
    lam = value[3]
    beta = value[4]
    initial_sig = value[5]

    sigma_list = [initial_sig]  # open list for all sigma values

    for i in range(1, len(df)):  # loop through the data of returns
        if df.iloc[i - 1, 4] < 0:  # check condtion for the event A
            # a is the second sigma that we get from the equation
            a = ((alpha * (df.iloc[i - 1, 4]) ** 2) + delta * ((df.iloc[i - 1, 4]) ** 2) * 1) / (
                    1 + ((df.iloc[i - 1, 4]) ** 2 / (lam * sigma_list[i - 1]))) + omega + (beta * sigma_list[i - 1])
            sigma_list.append(a)  # we attach the sigma

        else:
            # b is the second sigma that we get from the equation when the case is not met
            b = ((alpha * (df.iloc[i - 1, 4]) ** 2) + delta * ((df.iloc[i - 1, 4]) ** 2) * 0) / (
                    1 + ((df.iloc[i - 1, 4]) ** 2 / (lam * sigma_list[i - 1]))) + omega + (beta * sigma_list[i - 1])
            sigma_list.append(b)  # we attach the sigma

    return sigma_list


def VaR_table_with_lev(parameters_df, df):
    # extract the sigma list
    sig_list = sigma_leverage_Q6_with_lev(parameters_df, df)

    # create a new DataFrame where all the VaRs and returns will be stored
    df_sig = pd.DataFrame()
    df_sig["Actual Volatility sig ^ 2"] = sig_list  # create volatility columnd
    df_sig["Actual Standard errors"] = np.sqrt(df_sig["Actual Volatility sig ^ 2"])  # create standard error column
    df_sig["Returns"] = df['transformed_mean']  # create actual returns column

    df_sig = df_sig[2501:]
    deg_freedom = len(df_sig)
    sample_size = len(df_sig)

    df_sig["VaR at 1 % "] = df_sig["Actual Standard errors"] * t.ppf(0.01, deg_freedom)
    df_sig["VaR at 5 % "] = df_sig["Actual Standard errors"] * t.ppf(0.05, deg_freedom)
    df_sig["VaR at 10 % "] = df_sig["Actual Standard errors"] * t.ppf(0.1, deg_freedom)

    """ a corresponds to 1% Var , b list corresponds to 5% VaR, and c list corresponds to 10% VaR"""
    a = []
    b = []
    c = []

    for i in range(len(df_sig)):
        if df_sig.iloc[i, 3] > df_sig.iloc[i, 2]:
            a.append(1)
        else:
            a.append(0)

    for i in range(len(df_sig)):
        if df_sig.iloc[i, 4] > df_sig.iloc[i, 2]:
            b.append(1)
        else:
            b.append(0)

    for i in range(len(df_sig)):
        if df_sig.iloc[i, 5] > df_sig.iloc[i, 2]:
            c.append(1)
        else:
            c.append(0)

    df_sig["Hit function at 1 %"] = a
    df_sig["Hit function at 5 %"] = b
    df_sig["Hit function at 10 %"] = c

    hit_a = sum(a) / len(a)
    hit_b = sum(b) / len(b)
    hit_c = sum(c) / len(c)

    SE_a = np.sqrt(np.var(a)) / np.sqrt(sample_size)
    SE_b = np.sqrt(np.var(b)) / np.sqrt(sample_size)
    SE_c = np.sqrt(np.var(c)) / np.sqrt(sample_size)

    return "Hit rate for 1% VaR = ", hit_a, "Hit rate for 5% VaR = ", hit_b, "Hit rate for 10% VaR = ", hit_c, "Standard error for 1 % VaR = ", SE_a, "Standard error for 5 % VaR = ", SE_b, "Standard error for 10 % VaR = ", SE_c


def sigma_without_leverage_Q6(value, df):
    omega = value[0]
    alpha = value[1]
    lam = value[2]
    beta = value[3]
    initial_sig = value[4]

    sigma_list = [initial_sig]  # open list for all sigma values

    for i in range(1, len(df)):  # loop through the data of returns
        if df.iloc[i - 1, 4] < 0:  # check condition for the event A
            # a is the second sigma that we get from the equation
            a = (alpha * (df.iloc[i - 1, 4]) ** 2) / (
                    1 + ((df.iloc[i - 1, 4]) ** 2 / (lam * sigma_list[i - 1]))) + omega + (beta * sigma_list[i - 1])
            sigma_list.append(a)  # we attach the sigma

        else:
            # b is the second sigma that we get from the equation when the case is not met
            b = (alpha * (df.iloc[i - 1, 4]) ** 2) / (
                    1 + ((df.iloc[i - 1, 4]) ** 2 / (lam * sigma_list[i - 1]))) + omega + (beta * sigma_list[i - 1])
            sigma_list.append(b)  # we attach the sigma

    return sigma_list


def VaR_table_without_lev(parameters_df, df):
    # extract the sigma list
    sig_list = sigma_without_leverage_Q6(parameters_df, df)

    # create a new DataFrame where all the VaRs and returns will be stored
    df_sig = pd.DataFrame()
    df_sig["Actual Volatility sig ^ 2"] = sig_list  # create volatility columnd
    df_sig["Actual Standard errors"] = np.sqrt(df_sig["Actual Volatility sig ^ 2"])  # create standard error column
    df_sig["Returns"] = df['transformed_mean']  # create actual returns column

    df_sig = df_sig[2501:]
    deg_freedom = len(df_sig)
    sample_size = len(df_sig)

    df_sig["VaR at 1 % "] = df_sig["Actual Standard errors"] * t.ppf(0.01, deg_freedom)
    df_sig["VaR at 5 % "] = df_sig["Actual Standard errors"] * t.ppf(0.05, deg_freedom)
    df_sig["VaR at 10 % "] = df_sig["Actual Standard errors"] * t.ppf(0.1, deg_freedom)

    """ a corresponds to 1% Var , b list corresponds to 5% VaR, and c list corresponds to 10% VaR"""
    a = []
    b = []
    c = []

    for i in range(len(df_sig)):
        if df_sig.iloc[i, 3] > df_sig.iloc[i, 2]:
            a.append(1)
        else:
            a.append(0)

    for i in range(len(df_sig)):
        if df_sig.iloc[i, 4] > df_sig.iloc[i, 2]:
            b.append(1)
        else:
            b.append(0)

    for i in range(len(df_sig)):
        if df_sig.iloc[i, 5] > df_sig.iloc[i, 2]:
            c.append(1)
        else:
            c.append(0)

    df_sig["Hit function at 1 %"] = a
    df_sig["Hit function at 5 %"] = b
    df_sig["Hit function at 10 %"] = c

    hit_a = sum(a) / len(a)
    hit_b = sum(b) / len(b)
    hit_c = sum(c) / len(c)

    SE_a = np.sqrt(np.var(a)) / np.sqrt(sample_size)
    SE_b = np.sqrt(np.var(b)) / np.sqrt(sample_size)
    SE_c = np.sqrt(np.var(c)) / np.sqrt(sample_size)

    return "Hit rate for 1% VaR = ", hit_a, "Hit rate for 5% VaR = ", hit_b, "Hit rate for 10% VaR = ", hit_c, "Standard error for 1 % VaR = ", SE_a, "Standard error for 5 % VaR = ", SE_b, "Standard error for 10 % VaR = ", SE_c


print(VaR_table_with_lev(parameters_MRK_with_lev, df_MRK))
print(VaR_table_with_lev(parameters_PFE_with_lev, df_PFE))
print(VaR_table_with_lev(parameter_JNJ_with_lev, df_JNJ))

print(VaR_table_without_lev(MRK_without_lev, df_MRK))
print(VaR_table_without_lev(PFE_without_lev, df_PFE))
print(VaR_table_without_lev(JNJ_without_lev, df_JNJ))

##### SCRIPT END #####
