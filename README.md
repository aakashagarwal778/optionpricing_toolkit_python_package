# **OptionPricing Toolkit - Python Package**

## **Introduction**

The OptionPricing toolkit is a Python package designed to empower financial analysts, investors, and decision-makers with robust tools for option pricing and analysis. Leveraging a range of financial models and numerical methods, it provides accurate computations of option parameters and prices, facilitating informed decision-making in the financial domain.

## **Package Contents**

### **OptionPricing Class**

The core functionality is encapsulated in the OptionPricing class, leveraging essential libraries like NumPy, SciPy, and utilities for robust mathematical computations and data manipulation. Here's an overview of included methods:

#### **Cox-Ross-Rubinstein (CRR) Model**

- **CRR_stock:** Computes stock price matrix using the CRR model.
- **CRR_Option:** Computes European and American call/put option prices using the CRR model.

#### **Black-Scholes Model**

- **BlackScholes_Option:** Computes European call/put option prices using the Black-Scholes model.

#### **Monte Carlo Simulation**

- **MonteCarlo_Option:** Computes European call/put option prices using Monte Carlo simulation.

#### **Numerical Integration**

- **BS_Price_Int:** Computes European call/put option prices using numerical integration.

#### **Laplace Transform Method (Black-Scholes Model)**

- **laplace_BS:** Computes European call/put option prices using the Laplace transform method in the Black-Scholes model.

#### **Finite Difference Scheme**

- **AmPerpPut_ODE:** Computes American perpetual put option prices using the finite difference scheme (explicit method).
- **BS_EuOption_FiDi_Explicit:** Computes European call/put option prices using the finite difference scheme (explicit method).

#### **Laplace Transform Method (Heston Model)**

- **laplace_heston:** Computes European call/put option prices using the Laplace transform method in the Heston model.

## **Error Handling**

The toolkit is equipped with robust error-handling mechanisms:

- Validate input parameters to prevent type errors and value errors.
- Handle boundary conditions and mathematical errors gracefully.
- Provide clear and informative error messages for troubleshooting.

## **Real-World Data Integration**

Utilize real datasets from sources like Yahoo Finance for:

- Option pricing calculations.
- Implied volatility computations.
- Arbitrage analysis and detection.

## **Documentation and Usage**

Comprehensive documentation within the package offers:

- Detailed descriptions of each function, including parameters and return values.
- Practical examples demonstrating usage in various financial scenarios.
