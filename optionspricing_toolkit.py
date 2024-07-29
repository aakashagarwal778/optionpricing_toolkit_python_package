import numpy as np
import math
from scipy.stats import norm
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import cmath

class OptionPricing:
    def __init__(self, S_0, r, sigma, T, K, M):
        self.S_0 = S_0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.M = M

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

    # Computes the option price using the Laplace transform method in the Black-Scholes model
    # for European call/put options
    def laplace_heston(self, S0, r, gam0, kappa, lamb, sig_tild, T, K, R, p):
        try:
            def f_tilde(z):
                return p * cmath.exp((1 - z / p) * math.log(K)) / (z * (z - p))

            # Characteristic function
            def chi(u):
                return self.heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T)

            def integrand(u):
                return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

            V0 = integrate.quad(integrand, 0, 50)
            return V0

        except TypeError as e:
            return e
        except ValueError as e:
            return e

    def heston_char(self, u, S0, r, gam0, kappa, lamb, sig_tild, T):
        d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
        phi = np.cosh(0.5 * d * T)
        psi = np.sinh(0.5 * d * T) / d
        first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi)) ** (2 * kappa / sig_tild ** 2)
        second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
        return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor

    def Heston_FFT(self, S0, r, gam0, kappa, lamb, sig_tild, T, K, R, N):
        try:
            K = np.atleast_1d(K)
            f_tilde_0 = lambda u: 1 / (u * (u - 1))
            chi_0 = lambda u: self.heston_char(u, S0=S0, r=r, gam0=gam0, kappa=kappa, lamb=lamb, sig_tild=sig_tild, T=T)
            g = lambda u: f_tilde_0(R + 1j * u) * chi_0(u - 1j * R)

            kappa_1 = np.log(K[0])
            M = np.minimum(2 * np.pi * (N - 1) / (np.log(K[-1]) - kappa_1), 500)
            Delta = M / N
            n = np.arange(1, N + 1)
            kappa_m = np.linspace(kappa_1, kappa_1 + 2 * np.pi * (N - 1) / M, N)

            x = g((n - 0.5) * Delta) * Delta * np.exp(-1j * (n - 1) * Delta * kappa_1)
            x_hat = np.fft.fft(x)

            V_kappa_m = np.exp(-r * T + (1 - R) * kappa_m) / np.pi * np.real(x_hat * np.exp(-0.5 * Delta * kappa_m * 1j))
            return interp1d(kappa_m, V_kappa_m)(np.log(K))
        
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