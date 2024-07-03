import numpy as np
import math
from scipy.stats import norm
from scipy import integrate
from scipy.integrate import solve_ivp
import cmath

class OptionPricing:
    def __init__(self):
        pass

    # Computes the stock price matrix in the CRR model
    def CRR_stock(self, S_0, r, sigma, T, M):
        delta_t = T / M  # Time step
        beta = 0.5 * (math.exp(-r * delta_t) + math.exp((r + sigma**2) * delta_t))  # Beta value for the CRR model 
        u = beta + math.sqrt(math.pow(beta, 2) - 1)  # Up factor
        d = 1 / u  # Down factor

        # Generating an empty matrix for the stock prices
        S = np.empty((M + 1, M + 1))

        # Computing and inputting stock prices into the matrix 'S'
        for i in range(M + 1):  # i represents the time step
            for j in range(i + 1):  # j represents the number of up movements
                S[j, i] = S_0 * (u ** j) * (d ** (i - j))  # Stock price at time i and j up movements

        return S
    
    # Computes the option price using the Cox-Ross-Rubinstein (CRR) model
    # for European and American call/put options
    def CRR_Option(self, S_0, r, sigma, T, M, K, option_type='call', option_style='European'):
        try:
            # Check types of input parameters
            if not all(isinstance(arg, (int, float)) for arg in [S_0, r, sigma, T, M, K]):
                raise TypeError("All input parameters must be integers or floats.")
            
            delta_t = T / M
            beta = 0.5 * (math.exp(-r * delta_t) + math.exp((r + sigma**2) * delta_t))
            u = beta + math.sqrt(math.pow(beta, 2) - 1)
            d = beta - math.sqrt(math.pow(beta, 2) - 1)
            q = (math.exp(r * delta_t) - d) / (u - d)  # Risk Neutral Probability 

            S = self.CRR_stock(S_0, r, sigma, T, M)  # Stock price matrix

            # Computing option prices at maturity
            if option_type == 'call':
                V = np.maximum(0, S - K)
            elif option_type == 'put':
                V = np.maximum(0, K - S)
            else:
                raise ValueError("Option type must be either 'call' or 'put'.")

            # Backward induction to compute option prices
            if option_style == 'European':
                for i in range(M-1, -1, -1):
                    for j in range(i+1):
                        V[j, i] = math.exp(-r * delta_t) * (q * V[j, i+1] + (1 - q) * V[j+1, i+1])
            elif option_style == 'American':
                for i in range(M-1, -1, -1):
                    for j in range(i+1):
                        V[j, i] = max(V[j, i], math.exp(-r * delta_t) * (q * V[j, i+1] + (1 - q) * V[j+1, i+1]))

            return V[0, 0]

        except TypeError as e:
            return e
        except ValueError as e:
            return e


    # Computes the option price using the Black-Scholes model
    # for European call/put options
    def BlackScholes_Option(self, S_0, r, sigma, T, K, option_type='call'):
        try:
            # Check types of input parameters
            if not all(isinstance(arg, (int, float)) for arg in [S_0, r, sigma, T, K]):
                raise TypeError("All input parameters must be integers or floats.")

            d1 = (math.log(S_0 / K) + (r + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)

            if option_type == 'call':
                V_0 = S_0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            elif option_type == 'put':
                V_0 = K * math.exp(-r * T) * norm.cdf(-d2) - S_0 * norm.cdf(-d1)
            else:
                raise ValueError("Option type must be either 'call' or 'put'.")

            return V_0
        
        except TypeError as e:
            return e
        except ValueError as e:
            return e
        

    # Computes the option price using Monte Carlo simulation
    # for European call/put options
    def MonteCarlo_Option(self, S0, r, sigma, T, K, M, option_type='call'):
        try:
            # Check types of input parameters
            if not all(isinstance(arg, (int, float)) for arg in [S0, r, sigma, T, K, M]):
                raise TypeError("All input parameters must be integers or floats.")

            # Generate M samples from a standard normal distribution
            X = np.random.normal(0, 1, M)

            # Calculate terminal stock prices
            ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * X)

            # Define payoff functions for call and put options
            def call_payoff(ST, K):
                return np.maximum(ST - K, 0)

            def put_payoff(ST, K):
                return np.maximum(K - ST, 0)

            # Determine payoff function based on option type
            if option_type == 'call':
                payoff_fun = call_payoff(ST, K)
            elif option_type == 'put':
                payoff_fun = put_payoff(ST, K)
            else:
                raise ValueError("Option type must be 'call' or 'put'.")

            # Discounted expected payoff
            V0 = np.exp(-r * T) * np.mean(payoff_fun)

            # Standard error and confidence intervals
            se = np.std(payoff_fun) / np.sqrt(M)
            z = 1.96  # 95% confidence interval
            c1 = V0 - z * se
            c2 = V0 + z * se

            return V0, c1, c2

        except TypeError as e:
            return e
        except ValueError as e:
            return e


    # Computes the option price using numerical integration
    # for European call/put options
    def BS_Price_Int(self, S0, r, sigma, T, K, option_type='call'):
        try:
            # Check types of input parameters
            if not all(isinstance(arg, (int, float)) for arg in [S0, r, sigma, T, K]):
                raise TypeError("All input parameters must be integers or floats.")

            # Define the payoff functions for call and put options
            def call_payoff(ST, K):
                return np.maximum(ST - K, 0)

            def put_payoff(ST, K):
                return np.maximum(K - ST, 0)

            def integrand(x):
                norm_const = 1 / math.sqrt(2 * math.pi)
                exponent = (r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * x
                stock_price_term = S0 * math.exp(exponent)

                if option_type == 'call':
                    payoff = call_payoff(stock_price_term, K)
                elif option_type == 'put':
                    payoff = put_payoff(stock_price_term, K)
                else:
                    raise ValueError("Option type must be 'call' or 'put'.")

                discount_factor = math.exp(-r * T)
                normal_exp_term = math.exp(-0.5 * math.pow(x, 2))
                V = norm_const * payoff * discount_factor * normal_exp_term
                return V

            integral = integrate.quad(integrand, -np.inf, np.inf)
            return integral[0]

        except TypeError as e:
            return e
        except ValueError as e:
            return e
        
    # Computes the option price using the Laplace transform method in the Black-Scholes model
    # for European call/put options    
    def laplace_BS(self, S0, r, sigma, T, K, R, option_type='call'):
        try:
            # Check types of input parameters
            if not all(isinstance(arg, (int, float)) for arg in [S0, r, sigma, T, K, R]):
                raise TypeError("All input parameters must be integers or floats.")

            # Define the payoff functions for call and put options
            def f_tilde(z):
                if option_type == 'call':
                    return cmath.exp((1 - z) * math.log(K)) / (z * (z - 1))
                elif option_type == 'put':
                    return cmath.exp((1 - z) * math.log(K)) / (z * (z - 1))
    
            # Define the characteristic function for the Black-Scholes model
            def chi(u):
                return cmath.exp(complex(0, 1) * u * (math.log(S0) + r * T) - (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * math.pow(sigma, 2) / 2 * T)
            
            # Define the integrand for the Laplace transform method
            def integrand(u):
                return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

            V0 = integrate.quad(integrand, 0, 50)
            return V0

        except TypeError as e:
            return e
        except ValueError as e:
            return e

    # Computes the option price using the finite difference scheme (explicit method)
    def AmPerpPut_ODE(self, S_max, N, r, sigma, K):
        try:
            g = lambda S: np.maximum(K - S, 0)

            S_grid = np.linspace(0, S_max, N + 1)
            v_grid = np.zeros_like(S_grid)

            def fun(x, v):
                return np.array([v[1], 2 * r / (sigma ** 2 * x ** 2) * (v[0] - x * v[1])])

            x_star = 2 * K * r / (2 * r + sigma ** 2)

            v_grid[S_grid <= x_star] = g(S_grid[S_grid <= x_star])

            result = solve_ivp(fun=fun, t_span=(x_star, S_max), y0=[g(x_star), -1], t_eval=S_grid[S_grid > x_star])
            v_grid[S_grid > x_star] = result.y[0]

            return S_grid, v_grid

        except TypeError as e:
            return e
        except ValueError as e:
            return e

    # Compute the option price using the Laplace transform method in the Heston model
    # for European call/put options
    def laplace_heston(self, S0, r, nu0, kappa, lmbda, sigma_tilde, T, K, R, p):
        try:
            def f_tilde(z):
                return p * cmath.exp((1 - z / p) * math.log(K)) / (z * (z - p))

            def chi(u):
                d = cmath.sqrt(math.pow(lmbda, 2) + math.pow(sigma_tilde, 2) * (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))))
                n = cmath.cosh(d * T / 2) + lmbda * cmath.sinh(d * T / 2) / d
                z1 = math.exp(lmbda * T / 2)
                z2 = (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * cmath.sinh(d * T / 2) / d
                v = cmath.exp(complex(0, 1) * u * (math.log(S0) + r * T)) * cmath.exp(2 * kappa / math.pow(sigma_tilde, 2) * cmath.log(z1 / n)) * cmath.exp(-nu0 * z2 / n)
                return v

            def integrand(u):
                return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

            V0 = integrate.quad(integrand, 0, 50)
            return V0

        except TypeError as e:
            return e
        except ValueError as e:
            return e
    

    # Computes the option price using the finite difference scheme (explicit method)
    # for European call/put options
    def BS_EuOption_FiDi_Explicit(self, S0, r, sigma, T, K, m, nu_max, option_type='call'):
        try:
            q = 2 * r / sigma ** 2 
            delta_x = (math.log(2 * K / S0)) / m  
            delta_t = sigma ** 2 * T / (2 * nu_max) 
            fidi_lambda = delta_t / delta_x ** 2  
            lambda_tilde = (1 - 2 * fidi_lambda) 

            x = np.arange(-m, m + 1) * delta_x 
            w = np.zeros((2 * m + 1, nu_max + 1))

            if option_type == 'call':
                w[:, 0] = np.maximum(S0 * np.exp(x) - K, 0)
            elif option_type == 'put':
                w[:, 0] = np.maximum(K - S0 * np.exp(x), 0)
            else:
                raise ValueError("Option type must be 'call' or 'put'.")

            for i in range(1, nu_max + 1):
                for j in range(1, 2 * m):
                    w[j, i] = fidi_lambda * w[j - 1, i - 1] + lambda_tilde * w[j, i - 1] + fidi_lambda * w[j + 1, i - 1]

            # Finding the index of S0 in the grid
            index_S0 = np.argmin(np.abs(S0 - S0 * np.exp(x)))
            V0 = np.exp(-r * T) * w[index_S0, nu_max]
            return V0

        except TypeError as e:
            return e
        except ValueError as e:
            return e

option_pricing = OptionPricing()

# Test parameters
S_0 = 100  # Initial stock price
r = 0.05   # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying asset
T = 1      # Time to expiration (in years)
K = 105    # Strike price
M = 100    # Number of time steps for CRR model

# Test commands for different methods and scenarios

# 1. Test Cox-Ross-Rubinstein (CRR) model for a European call option
option_type = 'call'
option_style = 'European'
option_price_crr_eu_call = option_pricing.CRR_Option(S_0, r, sigma, T, M, K, option_type, option_style)
print(f"European Call Option Price (CRR model): {option_price_crr_eu_call}")

# Test Cox-Ross-Rubinstein (CRR) model for an American put option
option_type = 'put'
option_style = 'American'
option_price_crr_am_put = option_pricing.CRR_Option(S_0, r, sigma, T, M, K, option_type, option_style)
print(f"American Put Option Price (CRR model): {option_price_crr_am_put}")

# 2. Test Black-Scholes model for a European put option
option_type = 'put'
option_price_bs_eu_put = option_pricing.BlackScholes_Option(S_0, r, sigma, T, K, option_type)
print(f"European Put Option Price (Black-Scholes model): {option_price_bs_eu_put}")

# 3. Test Monte Carlo simulation for a European call/put option
M = 10000  # Increase number of simulations for Monte Carlo
option_type = 'call'
option_price_mc_eu_call, _, _ = option_pricing.MonteCarlo_Option(S_0, r, sigma, T, K, M, option_type)
print(f"European Call Option Price (Monte Carlo simulation): {option_price_mc_eu_call}")

option_type = 'put'
option_price_mc_eu_put, _, _ = option_pricing.MonteCarlo_Option(S_0, r, sigma, T, K, M, option_type)
print(f"European Put Option Price (Monte Carlo simulation): {option_price_mc_eu_put}")

# 4. Test numerical integration for an European call/put option
option_type = 'call'
option_price_int_eu_call = option_pricing.BS_Price_Int(S_0, r, sigma, T, K, option_type)
print(f"European Call Option Price (Numerical Integration): {option_price_int_eu_call}")

option_type = 'put'
option_price_int_eu_put = option_pricing.BS_Price_Int(S_0, r, sigma, T, K, option_type)
print(f"European Put Option Price (Numerical Integration): {option_price_int_eu_put}")

# 5. Test Laplace transform method in Black-Scholes model for a European call option
R = 1.1
option_type = 'call'
option_price_laplace_call = option_pricing.laplace_BS(S_0, r, sigma, T, K, R, option_type)
print(f"European Call Option Price (Laplace transform method): {option_price_laplace_call}")

# 6. Test American perpetual put option (no need to specify type/style)
S_max = 200  # Maximum stock price
N = 1000     # Number of grid points
S_grid, v_grid = option_pricing.AmPerpPut_ODE(S_max, N, r, sigma, K)
print(f"American Perpetual Put Option Price: {v_grid[0]}")

# 7. Test Heston model using Laplace transform for a European put option
nu0 = math.pow(0.3, 2)     # Heston model parameters
kappa = math.pow(0.3, 2)
lmbda = 2.5
sigma_tilde = 0.2
p = 1
option_type = 'put'
option_price_heston_eu_put = option_pricing.laplace_heston(S_0, r, nu0, kappa, lmbda, sigma_tilde, T, K, R=1.5, p=p)
print(f"Heston European Put Option Price (Laplace transform method): {option_price_heston_eu_put[0]}")

# 8. Test for European call option using Finite Difference Scheme
m = 100  # Number of space steps
nu_max = 1000  # Number of time steps

# Test for European call option using Finite Difference Scheme
option_type = 'call'
option_price_fd_eu_call = option_pricing.BS_EuOption_FiDi_Explicit(S_0, r, sigma, T, K, m, nu_max, option_type)
print(f"European Call Option Price (Finite Difference Scheme): {option_price_fd_eu_call}")

# Test for European put option using Finite Difference Scheme
option_type = 'put'
option_price_fd_eu_put = option_pricing.BS_EuOption_FiDi_Explicit(S_0, r, sigma, T, K, m, nu_max, option_type)
print(f"European Put Option Price (Finite Difference Scheme): {option_price_fd_eu_put}")