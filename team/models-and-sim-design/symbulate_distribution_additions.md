# Symbulate Distribution Additions

A prioritized list of common univariate and multivariate distributions not currently implemented in Symbulate (as of the `dev` branch), spanning probability, statistics, actuarial science, engineering, and finance coursework at both the undergraduate and graduate level.

**Total candidate distributions/wrappers:** 55

## Contents
- [Tier 1 — Highest Priority](#tier-1-highest-priority) (12 items)
- [Tier 2 — High Priority](#tier-2-high-priority) (13 items)
- [Tier 3 — Medium Priority](#tier-3-medium-priority) (12 items)
- [Tier 4 — Lower Priority](#tier-4-lower-priority) (18 items)

## Tier 1 — Highest Priority

| Distribution | Type | Field(s) | Typical Course(s) | Difficulty | Notes / Rationale |
|---|---|---|---|---|---|
| **Weibull** | Univariate Continuous | Engineering, Actuarial, Stats | Reliability Engineering, Survival Analysis, Intro Stats | Easy | Standard time-to-failure distribution; huge, near-universal gap |
| **Logistic** | Univariate Continuous | Statistics, ML/Data Science | Regression, Intro Stats, ML | Easy | Underlies logistic regression; simple closed-form CDF |
| **Laplace (Double Exponential)** | Univariate Continuous | Statistics, Signal Processing, Bayesian Stats | Mathematical Statistics, Bayesian Modeling | Easy | Robust error model; L1-style Bayesian priors |
| **Gompertz** | Univariate Continuous | Actuarial Science | Life Contingencies, Survival Models | Easy-Medium | Foundational mortality/force-of-mortality law |
| **Makeham** | Univariate Continuous | Actuarial Science | Life Contingencies | Easy-Medium | Gompertz plus constant accident term; taught alongside Gompertz |
| **De Moivre** | Univariate Continuous | Actuarial Science | Intro Life Contingencies | Easy | Simplest survival law (uniform remaining lifetime); reparameterized Uniform |
| **Dirichlet** | Multivariate | Statistics, Bayesian Stats, ML | Bayesian Statistics, Multivariate Stats, NLP/Topic Modeling | Medium | Natural multivariate extension of Beta; pairs with existing Multinomial |
| **Generalized Extreme Value (GEV)** | Univariate Continuous | Engineering, Actuarial, Finance/Risk | Extreme Value Theory, Hydrology, Risk Management | Medium | Unifies Gumbel/Frechet/reverse-Weibull via shape parameter; one class covers three named laws |
| **Noncentral t** | Univariate Continuous | Statistics | Mathematical Statistics, Power Analysis | Medium | Needed for power/effect-size analysis; central t already implemented |
| **Noncentral Chi-square** | Univariate Continuous | Statistics | Mathematical Statistics, Power Analysis | Medium | Needed for power analysis and non-null test distributions |
| **Noncentral F** | Univariate Continuous | Statistics | ANOVA, Power Analysis | Medium | Needed for ANOVA power analysis; central F already implemented |
| **Beta-Binomial** | Univariate Discrete | Statistics, Actuarial, Bayesian Stats | Bayesian Statistics, Loss Models | Easy-Medium | Standard conjugate model for overdispersed binomial/count data |

## Tier 2 — High Priority

| Distribution | Type | Field(s) | Typical Course(s) | Difficulty | Notes / Rationale |
|---|---|---|---|---|---|
| **Generalized Pareto Distribution (GPD)** | Univariate Continuous | Actuarial, Finance/Risk, Engineering | Extreme Value Theory, Insurance Risk | Medium | Peaks-over-threshold counterpart to GEV; pairs with GEV in EVT curricula |
| **Tweedie** | Univariate Continuous | Actuarial Science | P&C Ratemaking, GLMs | Medium-Hard | Compound Poisson-Gamma; core to actuarial claim modeling via GLMs |
| **Burr (Burr XII)** | Univariate Continuous | Actuarial Science, Economics | Loss Models (SOA curriculum) | Medium | Heavy-tailed severity distribution, standard in Klugman's Loss Models |
| **Lomax (Pareto Type II)** | Univariate Continuous | Actuarial Science, Economics | Loss Models, Income Distributions | Easy-Medium | Common claim-severity distribution; related to Pareto/Burr |
| **Inverse Gamma** | Univariate Continuous | Statistics, Bayesian Stats | Bayesian Statistics | Easy | Standard conjugate prior for variance parameters; pairs with existing Gamma |
| **Half-Normal** | Univariate Continuous | Statistics, Bayesian Stats | Bayesian Statistics | Easy | Common weakly-informative prior (e.g., Stan/PyMC defaults) |
| **Half-Cauchy** | Univariate Continuous | Statistics, Bayesian Stats | Bayesian Statistics | Easy | Common weakly-informative prior for scale parameters |
| **Multivariate t (Student)** | Multivariate | Statistics, Finance | Multivariate Statistics, Robust Regression, Finance | Medium-Hard | Robust alternative to Multivariate Normal; common in finance |
| **Wishart** | Multivariate | Statistics, Bayesian Stats | Multivariate Bayesian Statistics | Medium-Hard | Distribution over covariance matrices; pairs with existing MultivariateNormal |
| **Inverse Wishart** | Multivariate | Statistics, Bayesian Stats | Multivariate Bayesian Statistics | Medium-Hard | Standard conjugate prior for covariance matrices |
| **Gumbel (Extreme Value Type I)** | Univariate Continuous | Engineering, Finance, ML | Hydrology, Risk Management, ML (Gumbel-softmax) | Easy | Still taught by name even where GEV is available; unbounded-tail extreme value case |
| **Zipf** | Univariate Discrete | Statistics, Linguistics, CS | Intro Stats/CS Crossover, Power-Law Phenomena | Easy | Standard example of heavy-tailed/power-law discrete behavior |
| **Generic Truncation/Zero-Modification Wrapper** | Modifier (any discrete dist.) | Actuarial Science | Loss Models (SOA curriculum) | Medium | Not a single distribution — a wrapper enabling zero-truncated/zero-modified Poisson, Binomial, NegBinomial, Geometric; multiplies value of existing distributions |

## Tier 3 — Medium Priority

| Distribution | Type | Field(s) | Typical Course(s) | Difficulty | Notes / Rationale |
|---|---|---|---|---|---|
| **Negative Hypergeometric** | Univariate Discrete | Statistics, Probability | Probability Theory | Easy-Medium | Natural counterpart to existing Hypergeometric |
| **Multivariate Hypergeometric** | Multivariate | Statistics, Probability | Probability Theory | Medium | Generalizes existing Hypergeometric the way Multinomial generalizes Binomial |
| **Triangular** | Univariate Continuous | Simulation, Risk Modeling, Project Management | Simulation, Risk Analysis, PERT Estimation | Easy | Common first hand-built distribution; widely used in simulation/risk modeling |
| **Skew-Normal** | Univariate Continuous | Statistics, Finance, Engineering | Applied Statistics | Medium | Flexible skewed generalization of Normal |
| **Inverse Gaussian (Wald)** | Univariate Continuous | Statistics, Engineering | Applied Statistics, Reliability | Medium | First-passage-time distribution; used in reliability and applied stats |
| **Birnbaum-Saunders (Fatigue Life)** | Univariate Continuous | Engineering | Reliability Engineering, Fatigue Analysis | Medium | Standard fatigue-life model in engineering reliability |
| **Generalized Gamma** | Univariate Continuous | Statistics, Engineering | Reliability, Survival Analysis | Medium-Hard | Umbrella family covering Gamma, Weibull, Exponential as special cases |
| **Erlang** | Univariate Continuous | Engineering, Operations Research | Queueing Theory | Easy | Gamma with integer shape; taught by name constantly in queueing theory |
| **Rice / Rician** | Univariate Continuous | Engineering | Signal Processing, Communications | Medium | Standard model for signal fading in communications engineering |
| **Skellam** | Univariate Discrete | Statistics, Sports Analytics | Applied Statistics | Medium | Difference of two Poissons; appears in count-differencing applications |
| **Fréchet (Extreme Value Type II)** | Univariate Continuous | Finance, Engineering | Extreme Value Theory | Easy | Special case of GEV; lower priority once GEV implemented |
| **Log-Logistic (Fisk)** | Univariate Continuous | Actuarial Science, Economics | Survival Analysis, Loss Models | Medium | Survival-analysis alternative to Weibull; income distribution use also |

## Tier 4 — Lower Priority

| Distribution | Type | Field(s) | Typical Course(s) | Difficulty | Notes / Rationale |
|---|---|---|---|---|---|
| **Von Mises** | Univariate Continuous | Statistics, Engineering | Directional/Circular Statistics | Medium | Needed only for circular/directional data (wind direction, time-of-day) |
| **Nakagami** | Univariate Continuous | Engineering | Wireless Signal Modeling | Medium | Domain-specific to wireless/physics applications |
| **Lévy** | Univariate Continuous | Physics, Finance | Stochastic Processes | Medium | Special case of stable distributions; domain-specific |
| **Conway-Maxwell-Poisson** | Univariate Discrete | Statistics | Advanced Count-Data Modeling | Hard | Flexible over/under-dispersion count model; advanced elective topic |
| **Yule-Simon** | Univariate Discrete | Statistics, Network Science | Advanced Count-Data Modeling | Medium | Power-law discrete model; specialized |
| **Discrete Weibull** | Univariate Discrete | Statistics, Engineering | Advanced Reliability | Medium | Discrete analogue of Weibull; specialized |
| **Logarithmic (Log-Series)** | Univariate Discrete | Statistics, Ecology | Advanced Count-Data Modeling | Easy-Medium | Specialized count model, occasionally in ecology/biostatistics |
| **Variance-Gamma** | Univariate Continuous | Finance | Quantitative Finance (grad) | Hard | Alternative to Normal for log-returns in option pricing; no simple closed form |
| **Normal-Inverse Gaussian (NIG)** | Univariate Continuous | Finance | Quantitative Finance (grad) | Hard | Heavy-tailed returns model; grad-level quant finance |
| **Stable (Lévy-stable)** | Univariate Continuous | Finance, Statistics | Quantitative Finance, Probability Theory (grad) | Hard | Heavy-tailed returns/risk theory; no closed-form density in general |
| **Generalized Hyperbolic** | Univariate Continuous | Finance | Quantitative Finance (grad) | Hard | Umbrella family covering Variance-Gamma, NIG; lower priority vs. implementing those directly |
| **Johnson SU / SB** | Univariate Continuous | Finance, Engineering | Applied Statistics, Distribution Fitting | Medium-Hard | Flexible skew/kurtosis family for fitting non-normal data |
| **Gaussian Copula** | Multivariate / Dependence | Finance, Statistics | Risk Management, Quantitative Finance (grad) | Hard | Dependence modeling for correlated defaults/portfolio risk |
| **t-Copula** | Multivariate / Dependence | Finance | Quantitative Finance (grad) | Hard | Dependence modeling with heavier tail dependence than Gaussian copula |
| **Matrix Normal** | Multivariate | Statistics | Advanced Multivariate Statistics (grad) | Hard | Advanced matrix-variate model, mostly grad-level |
| **Phase-Type Distributions** | Univariate Continuous (class) | Actuarial Science, Operations Research | Ruin Theory, Queueing Theory (grad) | Hard | Advanced lifetime/ruin-theory modeling; grad-level actuarial/OR |
| **Singh-Maddala** | Univariate Continuous | Economics, Actuarial Science | Income Distribution Modeling | Medium | Reparameterization of Burr XII for income distributions |
| **Champernowne** | Univariate Continuous | Economics | Income Distribution Modeling (elective) | Medium | Specialized income-distribution model |

---

*Generated from `symbulate_distribution_additions.xlsx`. Difficulty and tier assignments reflect a combination of real-world usage frequency across fields and estimated implementation effort, not availability in any particular library (e.g., scipy).*