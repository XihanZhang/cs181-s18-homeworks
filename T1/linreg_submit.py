# import packages
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

##################################################
##################################################
######               Problem 2              ######
##################################################
##################################################

# read in data:
# (1) store data name
csv_filename = 'year-sunspots-republicans.csv'
# (2) creat column names
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# select data before 1985
sunspot = sunspot_counts[years<last_year]
republican = republican_counts[years<last_year]

#################
# Plot the data #
#################
fig, ax = plt.subplots(3, 1,figsize=(15, 18))
plt.suptitle("Plot of Data", fontsize=25, y = 0.92)
ax[0].scatter(years, republican_counts, s=70)
ax[0].set_xlabel("Year")
ax[0].set_ylabel("Number of Republicans in Congress")

ax[1].scatter(years, sunspot_counts, s=70)
ax[1].set_xlabel("Year")
ax[1].set_ylabel("Number of Sunspots")

ax[2].scatter(sunspot, republican, s=70)
ax[2].set_xlabel("Number of Sunspots")
ax[2].set_ylabel("Number of Republicans in Congress")

plt.savefig('data.png')

###################################################
#          years vs number of Republicans         #
###################################################
#(1) the simplest basis:
# matrix for input
X = np.vstack((np.ones(years.shape), years)).T
##  1st column: offset(1.0)
##  2nd column: time

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# compute mean square error
Yhat  = np.dot(X, w)
mse = mean_squared_error(republican_counts, Yhat)

# Compute the regression line on a grid of inputs.
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(grid_X.T, w)

# TODO: plot and report sum of squared error for each basis
# Plot the data and the regression line.
plt.figure(figsize=(15,6))
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figtext(0.5, 0, "MSE = %s" % mse, ha="center")
plt.savefig('simplest.png')

#(2) basis functions:
def make_basis_a(i):
    X = np.ones(i.shape).T
    for j in range(1, 6):
        X = np.vstack((X, i**j))
    return X.T

def make_basis_b(i):
    X = np.ones(i.shape).T
    for j in range(1960, 2015, 5):
        X = np.vstack((X, np.exp(np.divide(np.subtract(0,(i-j)**2),25))))
    return X.T

def make_basis_c(i):
    X = np.ones(i.shape).T
    for j in range(1, 6):
        X = np.vstack((X, np.cos(i/j)))
    return X.T

def make_basis_d(i):
    X = np.ones(i.shape).T
    for j in range(1, 26):
        X = np.vstack((X, np.cos(i/j)))
    return X.T

# matrix for inputs
X_a = make_basis_a(years)
X_b = make_basis_b(years)
X_c = make_basis_c(years)
X_d = make_basis_d(years)

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
w_a = np.linalg.solve(np.dot(X_a.T, X_a) , np.dot(X_a.T, Y))
w_b = np.linalg.solve(np.dot(X_b.T, X_b) , np.dot(X_b.T, Y))
w_c = np.linalg.solve(np.dot(X_c.T, X_c) , np.dot(X_c.T, Y))
w_d = np.linalg.solve(np.dot(X_d.T, X_d) , np.dot(X_d.T, Y))

# compute mean square error
Yhat_a  = np.dot(X_a, w_a)
mse_a = mean_squared_error(republican_counts, Yhat_a)

Yhat_b  = np.dot(X_b, w_b)
mse_b = mean_squared_error(republican_counts, Yhat_b)

Yhat_c  = np.dot(X_c, w_c)
mse_c = mean_squared_error(republican_counts, Yhat_c)

Yhat_d  = np.dot(X_d, w_d)
mse_d = mean_squared_error(republican_counts, Yhat_d)

# Compute the regression line on a grid of inputs.
grid_years = np.linspace(1960, 2005, 200)
grid_X_a = make_basis_a(grid_years)
grid_Yhat_a  = np.dot(grid_X_a, w_a)

grid_years = np.linspace(1960, 2005, 200)
grid_X_b = make_basis_b(grid_years)
grid_Yhat_b  = np.dot(grid_X_b, w_b)

grid_years = np.linspace(1960, 2005, 200)
grid_X_c = make_basis_c(grid_years)
grid_Yhat_c  = np.dot(grid_X_c, w_c)

grid_years = np.linspace(1960, 2005, 200)
grid_X_d = make_basis_d(grid_years)
grid_Yhat_d  = np.dot(grid_X_d, w_d)

# TODO: plot and report sum of squared error for each basis
# Plot the data and the regression line.
plt.figure(figsize=(15,6))
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_a, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figtext(0.5, 0, "MSE = %s" % mse_a, ha="center")
plt.savefig('a.png')

plt.figure(figsize=(15,6))
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_b, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figtext(0.5, 0, "MSE = %s" % mse_b, ha="center")
plt.savefig('b.png')

plt.figure(figsize=(15,6))
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_c, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figtext(0.5, 0, "MSE = %s" % mse_a, ha="center")
plt.savefig('c.png')

plt.figure(figsize=(15,6))
plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat_d, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figtext(0.5, 0, "MSE = %s" % mse_d, ha="center")
plt.savefig('d.png')

###################################################
#       sunspots vs number of republicans         #
###################################################
#(1) the simplest basis:
# matrix for input
X_0_2 = np.vstack((np.ones(sunspot.shape), sunspot)).T
##  1st column: time
##  2nd column: offset(1.0)

# Nothing fancy for outputs.
Y_2 = republican

# Find the regression weights using the Moore-Penrose pseudoinverse.
w_2 = np.linalg.solve(np.dot(X_0_2.T, X_0_2) , np.dot(X_0_2.T, Y_2))

# compute mean square error
Yhat_2  = np.dot(X_0_2, w_2)
mse_2 = mean_squared_error(republican, Yhat_2)

# Compute the regression line on a grid of inputs.
grid_sunspot = np.linspace(10.2, 159, 200)
grid_X_0_2 = np.vstack((np.ones(grid_sunspot.shape), grid_sunspot))
grid_Yhat_2  = np.dot(grid_X_0_2.T, w_2)

# TODO: plot and report sum of squared error for each basis
# Plot the data and the regression line.
plt.figure(figsize=(15,6))
plt.plot(sunspot, republican, 'o', grid_sunspot, grid_Yhat_2, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.figtext(0.5, 0, "MSE = %s" % mse, ha="center")
plt.savefig('simplest_2.png')

# (2) matrix for inputs:
X_a_2 = make_basis_a(sunspot)
X_c_2 = make_basis_c(sunspot)
X_d_2 = make_basis_d(sunspot)

# Nothing fancy for outputs.
Y_2 = republican

# Find the regression weights using the Moore-Penrose pseudoinverse.
w_a_2 = np.linalg.solve(np.dot(X_a_2.T, X_a_2) , np.dot(X_a_2.T, Y_2))
w_c_2 = np.linalg.solve(np.dot(X_c_2.T, X_c_2) , np.dot(X_c_2.T, Y_2))
w_d_2 = np.linalg.solve(np.dot(X_d_2.T, X_d_2) , np.dot(X_d_2.T, Y_2))

# compute mean square error
Yhat_a_2  = np.dot(X_a_2, w_a_2)
mse_a_2 = mean_squared_error(Y_2, Yhat_a_2)

Yhat_c_2  = np.dot(X_c_2, w_c_2)
mse_c_2 = mean_squared_error(Y_2, Yhat_c_2)

Yhat_d_2  = np.dot(X_d_2, w_d_2)
mse_d_2 = mean_squared_error(Y_2, Yhat_d_2)

# Compute the regression line on a grid of inputs.
grid_sunspot = np.linspace(10.2, 159, 200)
grid_X_a_2 = make_basis_a(grid_sunspot)
grid_Yhat_a_2  = np.dot(grid_X_a_2, w_a_2)

grid_sunspot = np.linspace(10.2, 159, 200)
grid_X_c_2 = make_basis_c(grid_sunspot)
grid_Yhat_c_2  = np.dot(grid_X_c_2, w_c_2)

grid_sunspot = np.linspace(10.2, 159, 200)
grid_X_d_2 = make_basis_d(grid_sunspot)
grid_Yhat_d_2  = np.dot(grid_X_d_2, w_d_2)

# TODO: plot and report sum of squared error for each basis
# Plot the data and the regression line.
plt.figure(figsize=(15,6))
plt.plot(sunspot, republican, 'o', grid_sunspot, grid_Yhat_a_2, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.figtext(0.5, 0, "MSE = %s" % mse_a_2, ha="center")
plt.savefig('a_2.png')

plt.figure(figsize=(15,6))
plt.plot(sunspot, republican, 'o', grid_sunspot, grid_Yhat_c_2, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.figtext(0.5, 0, "MSE = %s" % mse_c_2, ha="center")
plt.savefig('c_2.png')

plt.figure(figsize=(15,6))
plt.plot(sunspot, republican, 'o', grid_sunspot, grid_Yhat_d_2, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.figtext(0.5, 0, "MSE = %s" % mse_d_2, ha="center")
plt.savefig('d_2.png')



##################################################
##################################################
######               Problem 3              ######
##################################################
##################################################

##################################################
#       (a) function to generate sample          #
##################################################

# function to generate N samples with K_true polynomials
def poly_sample(N,K_true):

    # generate matrix with xi: 1.0, xi
    x = np.random.uniform(-5,5,N)            # 1 row of xi under Unif(-5, 5)
    x_offset = np.vstack((np.ones(x.shape), x)).T   # 1st column: 1.0  2nd column: xi

    # make polynomial features (xi --> 1*a0, xi*a1, xi^2*a2, xi^3*a3...xi^K_true*a_true)
    x_poly = []
    for xi in x_offset:
        # the 1st column: 1.0 -->a0
        x_row = []
        x_row.append(np.random.uniform(-1,1))
        # turn the 2nd column to more columns: xi --> xi*a1, xi^2*a2, xi^3*a3...xi^K_true*a_true
        for k in range(K_true):
            x_row.append(np.random.uniform(-1,1)*(xi[1]**(k+1)))
        x_poly.append(x_row)

    # calculate f(x)
    f = []
    for xi in x_poly:
        f.append(sum(xi))
    f_x = [[i] for i in f]  # convert row to column

    # calculate sigma
    sigma = ((max(f)-min(f))/10)**(1/2)

    # calculate y
    y = []
    for fi in f:
        e = np.random.normal(loc=0.0, scale=sigma)
        y.append(fi+e)
    y_x = [[i] for i in y]  # convert row to column

    # x: 1 row of 20 xi    x_poly: 20 rows of [1*a0, xi*a1, xi^2*a2, xi^3*a3...xi^K_true*a_true]
    # f: 1 row of 20 fi    f_x: 1 column of 20 fi
    # y: 1 row of 20 yi    y_x: 1 column of 20 yi

    return x, y, sigma

############################################################################
#   (b) function that minimizes chi-square for a polynomial of degree K    #
############################################################################

# (1) function to calculate Chi-square:
def Chi_square(N, x, y, K, sigma):
    # calculate coefficient a={a0, a1, a2, ..., aK}
    a = np.polyfit(x,y,K,full=True)[0]

    # calculate sum of square error
    #if N >= 17:
    if len(np.polyfit(x,y,K,full=True)[1])==0:
        SSE = 0
    else:
        [SSE] = np.polyfit(x,y,K,full=True)[1]

    # calculate Chi-square error
    Chi = SSE/((sigma)**2)

    return Chi

# (2) plot Chi-square(K) to make sure the function is decreasing when N = 20

# generate N=20 samples with K_true=10 polynomials
N = 20
K_true = 10
x,y,sigma = poly_sample(N=20,K_true=10)
# get x: K_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  y: Chi-square
K_test = []
Chi_test = []
for t in range(K_true+5):
    K_test.append(t+1)
    Chi_test.append(Chi_square(N, x, y, K_test[t], sigma))

# plot K vs Chi-square
plt.figure(figsize=(15,6))
plt.plot(K_test, Chi_test, 'o')
plt.xlabel("K")
plt.ylabel("Chi-square error")
plt.savefig('3_b.png')

##################################################
#       (c) mean and variance of optimal K       #
##################################################
# (1) function to select optimal K for 1 trial:
def optimal_K(x, y, sigma):

    K_set = []       # K_set = [1,2,3,...,K_true,K_true+1,...,K_true+5]
    BIC_set = []     # BIC_set = [BIC_1,BIC_2,...,BIC_K_True+5]
    for t in range(K_true):
        # generate set of K
        K_set.append(t+1)                         # K_set[t]: K

        # calculate BIC of each K
        Chi_sq = Chi_square(N, x, y, K_set[t], sigma)   # Chi_sq: arg(min)Chi-square
        BIC = Chi_sq + (K_set[t]+1)*np.log(N)

        # generate set of BIC
        BIC_set.append(BIC)

    # get the indext of the minumum BIC
    ID = BIC_set.index(min(BIC_set))
    # get the optimal K
    K_optimal = K_set[ID]

    return K_optimal #,BIC_set, K_set

# (2) function to run T tials with N sample and calculate mean & variance of optimal K
def simulate_T(T, N, K_true):

    K_optimals = []     # store K_optimal for each trial
    trial = []          # store trial ID

    # loop 200 times
    for t in range(T):
        trial.append(t)                       # trail = [0, 1, 2, 3, ..., 199]
        x, y, sigma = poly_sample(N,K_true)   # make N samples
        K_optimal = optimal_K(x, y, sigma)    # select K_optimal
        K_optimals.append(K_optimal)          # store K_optimal in K_optimals

    # calculate mean of K_optimals:
    K_optimal_mean = np.mean(K_optimals)

    # calculate variance of K_optimals:
    K_optimal_var = np.var(K_optimals)

    return K_optimal_mean, K_optimal_var#,  K_optimals

# (3) run 500 trials with N=20, K_true=10
K_optimal_mean, K_optimal_var = simulate_T(T=500, N=20, K_true=10)
print("The mean of optimal K over 500 trials is %s" % K_optimal_mean)
print("The varriance of optimal K over 500 trials is %s" % K_optimal_var)

##################################################
#           (d) plot optimal_K vs N              #
##################################################
# (1) function to run T tials with N sample and calculate mean & variance of optimal K
def simulate_T_N(T, e_N_max, K_true):

    K_means = []                                          # y:   store K_mean for each trial
    K_vars = []                                           # bar: store K_var for each trial
    Ns = np.rint(3*np.logspace(0,e_N_max,40)).astype(int) # x:   store N in Ns

    for N in Ns:
        K_mean, K_var = simulate_T(T, N, K_true)  # mean & varriance of optimal K of 500 trials
        K_means.append(K_mean)                    # store K_mean in K_means
        K_vars.append(K_var)                      # store K_var in K_vars
    return K_means, K_vars, Ns

# (2) do the final simulation
K_means, K_vars, Ns = simulate_T_N(T=500, e_N_max=4, K_true=10)

# (3) plot mean of K with error bar vs. sample size N
plt.figure(figsize=(15,6))
plt.errorbar(Ns, K_means, yerr=K_vars, fmt='o')
plt.xlabel("Sample Size (N)")
plt.ylabel("Mean Optimal K of 500 Trials")
plt.xscale('log')
plt.savefig('3_d.png')
