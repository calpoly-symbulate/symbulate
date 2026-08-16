# Symbulate Distribution Additions

A prioritized list of common univariate and multivariate distributions not currently implemented in Symbulate (as of the `dev` branch), spanning probability, statistics, actuarial science, engineering, and finance coursework at both the undergraduate and graduate level. Cross-referenced against Leemis's Univariate Distribution Relationship Chart, several distribution-calculator/simulation reference sites, and Wikipedia's List of Probability Distributions, in addition to the original scipy-based pass.

**Total candidate distributions/wrappers:** 120

**A note on the two difficulty columns:** "scipy.stats Availability" and "Symbulate Integration Effort" are deliberately separate. Whether scipy already has a ready-made version answers *can this be wrapped quickly*; integration effort answers *how much Symbulate-side work is still needed* even when scipy has it — e.g., translating an actuarial parameterization into scipy's convention (Gompertz), handling vector- or matrix-valued output (Dirichlet, Wishart, Matrix Normal, LKJ), or working around a numerically delicate scipy implementation (Stable). A few names are also naming traps worth flagging directly: `scipy.stats.dweibull` is the continuous "double Weibull," not the discrete Weibull, and `scipy.stats.zipf` is actually the Zeta distribution, not the (finite-support) Zipf distribution — that's `zipfian`.

**A few items below are architectural features rather than single named distributions** (the discrete truncation/zero-modification wrapper, its continuous counterpart, a general mixture-distribution builder, and a check on whether order statistics are already achievable via existing RV transformations). These are included because they'd unlock or simplify several other entries in this table rather than because they're single classes to implement.

## Contents
- [Tier 1 — Highest Priority](#tier-1-highest-priority) (13 items)
- [Tier 2 — High Priority](#tier-2-high-priority) (31 items)
- [Tier 3 — Medium Priority](#tier-3-medium-priority) (40 items)
- [Tier 4 — Lower Priority](#tier-4-lower-priority) (36 items)

## Tier 1 — Highest Priority

| Distribution | Type | Field(s) | Typical Course(s) | scipy.stats Availability | Symbulate Integration Effort | Notes / Rationale |
|---|---|---|---|---|---|---|
| **Weibull** | Univariate Continuous | Engineering, Actuarial, Stats | Reliability Engineering, Survival Analysis, Intro Stats | Yes (weibull_min) | Low | Simple 2-param wrap; standard time-to-failure distribution, huge near-universal gap |
| **Logistic** | Univariate Continuous | Statistics, ML/Data Science | Regression, Intro Stats, ML | Yes (logistic) | Low | Simple wrap; underlies logistic regression |
| **Laplace (Double Exponential)** | Univariate Continuous | Statistics, Signal Processing, Bayesian Stats | Mathematical Statistics, Bayesian Modeling | Yes (laplace) | Low | Simple wrap; robust error model, L1-style Bayesian priors |
| **Gompertz** | Univariate Continuous | Actuarial Science | Life Contingencies, Survival Models | Yes (gompertz) | Medium | scipy has it, but its (c, scale) parameterization must be translated from the actuarial (B, c) mortality convention |
| **Makeham** | Univariate Continuous | Actuarial Science | Life Contingencies | No | Medium-High | Not in scipy; needs a custom survival-function/hazard-based build (Gompertz + constant term), though the math is well-known and closed-form |
| **De Moivre** | Univariate Continuous | Actuarial Science | Intro Life Contingencies | No | Low | Not in scipy, but it's literally a reparameterized Uniform, so trivial to build directly |
| **Dirichlet** | Multivariate | Statistics, Bayesian Stats, ML | Bayesian Statistics, Multivariate Stats, NLP/Topic Modeling | Yes (dirichlet) | Medium | scipy has it, but vector-valued output needs the same joint-sampling/marginal handling work MultivariateNormal already required |
| **Generalized Extreme Value (GEV)** | Univariate Continuous | Engineering, Actuarial, Finance/Risk | Extreme Value Theory, Hydrology, Risk Management | Yes (genextreme) | Low | One wrap covers Gumbel/Frechet/reverse-Weibull via the shape parameter |
| **Noncentral t** | Univariate Continuous | Statistics | Mathematical Statistics, Power Analysis | Yes (nct) | Low | Direct wrap; needed for power/effect-size analysis alongside existing central t |
| **Noncentral Chi-square** | Univariate Continuous | Statistics | Mathematical Statistics, Power Analysis | Yes (ncx2) | Low | Direct wrap; needed for power analysis and non-null test distributions |
| **Noncentral F** | Univariate Continuous | Statistics | ANOVA, Power Analysis | Yes (ncf) | Low | Direct wrap; needed for ANOVA power analysis |
| **Beta-Binomial** | Univariate Discrete | Statistics, Actuarial, Bayesian Stats | Bayesian Statistics, Loss Models | Yes (betabinom) | Low | Direct wrap; standard conjugate model for overdispersed binomial/count data |
| **LKJ (correlation matrix prior)** | Multivariate (matrix-valued) | Statistics, Bayesian Stats | Bayesian Statistics, Multivariate Bayesian Modeling | No | Medium-High | The default correlation-matrix prior in modern Bayesian workflows (Stan/PyMC); arguably as important as Wishart for this audience, and neither scipy nor Symbulate has it |

## Tier 2 — High Priority

| Distribution | Type | Field(s) | Typical Course(s) | scipy.stats Availability | Symbulate Integration Effort | Notes / Rationale |
|---|---|---|---|---|---|---|
| **Generalized Pareto Distribution (GPD)** | Univariate Continuous | Actuarial, Finance/Risk, Engineering | Extreme Value Theory, Insurance Risk | Yes (genpareto) | Low | Direct wrap; peaks-over-threshold counterpart to GEV |
| **Tweedie** | Univariate Continuous | Actuarial Science | P&C Ratemaking, GLMs | No | High | Not in scipy.stats; compound Poisson-Gamma has no simple closed-form density in general, requires custom simulation and density approximation |
| **Burr (Burr XII)** | Univariate Continuous | Actuarial Science, Economics | Loss Models (SOA curriculum) | Yes (burr12) | Low | Direct wrap; standard heavy-tailed severity distribution in Klugman's Loss Models |
| **Lomax (Pareto Type II)** | Univariate Continuous | Actuarial Science, Economics | Loss Models, Income Distributions | Yes (lomax) | Low | Direct wrap; common claim-severity distribution |
| **Inverse Gamma** | Univariate Continuous | Statistics, Bayesian Stats | Bayesian Statistics | Yes (invgamma) | Low | Direct wrap; standard conjugate prior for variance parameters |
| **Half-Normal** | Univariate Continuous | Statistics, Bayesian Stats | Bayesian Statistics | Yes (halfnorm) | Low | Direct wrap; common weakly-informative prior (Stan/PyMC defaults) |
| **Half-Cauchy** | Univariate Continuous | Statistics, Bayesian Stats | Bayesian Statistics | Yes (halfcauchy) | Low | Direct wrap; common weakly-informative prior for scale parameters |
| **Multivariate t (Student)** | Multivariate | Statistics, Finance | Multivariate Statistics, Robust Regression, Finance | Yes (multivariate_t) | Medium | scipy has it, but vector-valued output needs the same handling MultivariateNormal already required |
| **Wishart** | Multivariate | Statistics, Bayesian Stats | Multivariate Bayesian Statistics | Yes (wishart) | Medium-High | scipy has it, but draws are entire matrices, not vectors — Symbulate has no existing precedent for plotting/summarizing matrix-valued draws |
| **Inverse Wishart** | Multivariate | Statistics, Bayesian Stats | Multivariate Bayesian Statistics | Yes (invwishart) | Medium-High | Same matrix-output handling challenge as Wishart |
| **Gumbel (Extreme Value Type I)** | Univariate Continuous | Engineering, Finance, ML | Hydrology, Risk Management, ML (Gumbel-softmax) | Yes (gumbel_r / gumbel_l) | Low | Direct wrap; still taught by name even where GEV is available |
| **Zipf** | Univariate Discrete | Statistics, Linguistics, CS | Intro Stats/CS Crossover, Power-Law Phenomena | Yes (zipfian / zipf) | Low | Direct wrap; standard example of heavy-tailed/power-law discrete behavior |
| **Generic Truncation/Zero-Modification Wrapper** | Modifier (any discrete dist.) | Actuarial Science | Loss Models (SOA curriculum) | No (architectural feature, not a distribution) | Medium-High | Needs to operate generically over any existing discrete distribution (Poisson, Binomial, NegBinomial, Geometric) rather than being one class; a design decision, not a wrap |
| **Beta-Pascal / Beta-Negative Binomial** | Univariate Discrete | Actuarial Science, Statistics | Bayesian Statistics, Loss Models | Yes (betanbinom) | Low | Overdispersed negative-binomial counterpart to Beta-Binomial (already in this table); direct wrap |
| **Kumaraswamy** | Univariate Continuous | Statistics, Engineering | Applied Statistics, Simulation | No | Low | Beta look-alike with simple closed-form CDF/quantile; genuinely useful and easy to implement even without scipy backing |
| **PERT** | Univariate Continuous | Project Management, Risk Analysis, Statistics | Simulation, Risk Analysis, PERT Estimation | No | Low-Medium | Special case of 4-parameter Beta; pairs directly with Triangular already in this table |
| **Bates** | Univariate Continuous | Statistics, Simulation | Simulation, Intro Statistics (CLT teaching) | No | Low | Mean of n uniforms; companion to Irwin-Hall (sum vs. mean), useful CLT teaching pair |
| **Irwin–Hall** | Univariate Continuous | Statistics, Simulation | Simulation, Intro Statistics (CLT teaching) | Yes (irwinhall) | Low | Sum of n iid uniforms; should have been in the original table — genuine omission, not a low-priority call |
| **Hyperexponential** | Univariate Continuous | Engineering, Operations Research | Queueing Theory, Reliability Engineering | No | Medium | Staple of queueing theory; models 'fast or slow' service-time heterogeneity via a mixture of exponentials |
| **Hypoexponential** | Univariate Continuous | Engineering, Operations Research | Queueing Theory, Reliability Engineering | No | Medium | Companion to Hyperexponential; models multi-stage service/failure processes as a sum of exponentials with distinct rates |
| **Log-Gamma** | Univariate Continuous | Actuarial Science, Statistics | Loss Models, Extreme Value Theory | Yes (loggamma) | Low | Mentioned in an earlier discussion of actuarial additions but never actually added — genuine omission being corrected here |
| **Truncated Normal** | Univariate Continuous | Statistics, Engineering, Bayesian Stats | Applied Statistics, Bayesian Statistics | Yes (truncnorm) | Medium | Extremely common in practice; also shows the existing discrete-only truncation wrapper needs a continuous counterpart |
| **Scaled/Inverse Chi-Squared** | Univariate Continuous | Statistics, Bayesian Stats | Bayesian Statistics | Yes (reparameterization of invgamma) | Low | Extremely common naming convention in Bayesian textbooks (e.g., Gelman et al.); worth a named alias even though mathematically = Inverse Gamma |
| **Gamma/Gompertz** | Univariate Continuous | Actuarial Science | Life Contingencies, Survival Models | No | Medium | Gompertz mortality with Gamma frailty (population heterogeneity); natural extension once Gompertz/Makeham exist |
| **Exponentially Modified Gaussian** | Univariate Continuous | Statistics, Psychology, Chemistry | Applied Statistics, Psychometrics | Yes (exponnorm) | Low | Real applied use in chromatography and reaction-time modeling in psychology — broader audience than most lower-tier items |
| **Skew t** | Univariate Continuous | Finance, Statistics | Robust Statistics, Quantitative Finance | No | Medium-High | Natural extension of Skew-Normal already in this table; notable in finance/robust modeling |
| **Reciprocal / Log-Uniform** | Univariate Continuous | Statistics, ML, Engineering | Applied Statistics, ML Hyperparameter Search | Yes (loguniform / reciprocal) | Low | Mentioned in the very first response of this conversation and never actually added to the table — genuine omission being corrected here; also widely used for ML hyperparameter search ranges |
| **Hotelling's T-squared** | Multivariate | Statistics | Multivariate Statistics | No | Medium | Multivariate generalization of t²; pairs directly with Multivariate t already in this table, core topic in multivariate stats courses |
| **Categorical (formal Distribution-API version)** | Univariate Discrete (non-numeric outcomes) | Statistics, ML | Intro Probability, ML | No (not a scipy concept) | Medium | Directly closes the gap identified in the BoxModel vs. Empirical discussion — would give BoxModel-style non-numeric outcomes the same pmf/mode/plot access Empirical gives integer outcomes |
| **General Mixture-Distribution Builder** | Modifier (any distribution) | Statistics, Bayesian Stats | Applied Statistics, Bayesian Statistics | Partial (component distributions exist; no generic container) | Medium-High | Generic weighted-combination wrapper over any existing distributions; would also directly enable Hyperexponential as a special case |
| **Continuous Truncation Wrapper** | Modifier (any continuous distribution) | Statistics, Engineering | Applied Statistics, Bayesian Statistics | Partial (truncnorm exists as one instance; no generic wrapper) | Medium | Continuous counterpart to the existing discrete truncation/zero-modification wrapper already in this table |

## Tier 3 — Medium Priority

| Distribution | Type | Field(s) | Typical Course(s) | scipy.stats Availability | Symbulate Integration Effort | Notes / Rationale |
|---|---|---|---|---|---|---|
| **Negative Hypergeometric** | Univariate Discrete | Statistics, Probability | Probability Theory | Yes (nhypergeom) | Low | Direct wrap; natural counterpart to existing Hypergeometric |
| **Multivariate Hypergeometric** | Multivariate | Statistics, Probability | Probability Theory | Yes (multivariate_hypergeom) | Medium | scipy has it, but needs the same vector-output handling as other multivariate additions |
| **Triangular** | Univariate Continuous | Simulation, Risk Modeling, Project Management | Simulation, Risk Analysis, PERT Estimation | Yes (triang) | Low | Direct wrap; common first hand-built distribution in simulation/risk modeling |
| **Skew-Normal** | Univariate Continuous | Statistics, Finance, Engineering | Applied Statistics | Yes (skewnorm) | Low | Direct wrap; flexible skewed generalization of Normal |
| **Inverse Gaussian (Wald)** | Univariate Continuous | Statistics, Engineering | Applied Statistics, Reliability | Yes (invgauss / wald) | Low | Direct wrap; first-passage-time distribution |
| **Birnbaum-Saunders (Fatigue Life)** | Univariate Continuous | Engineering | Reliability Engineering, Fatigue Analysis | Yes (fatiguelife) | Low | Direct wrap; standard fatigue-life model in engineering reliability |
| **Generalized Gamma** | Univariate Continuous | Statistics, Engineering | Reliability, Survival Analysis | Yes (gengamma) | Low-Medium | scipy has it, but mapping parameters across its Gamma/Weibull/Exponential special cases needs care to present clearly to students |
| **Erlang** | Univariate Continuous | Engineering, Operations Research | Queueing Theory | Yes (erlang) | Low | Direct wrap; Gamma with integer shape, taught by name constantly in queueing theory |
| **Rice / Rician** | Univariate Continuous | Engineering | Signal Processing, Communications | Yes (rice) | Low | Direct wrap; standard model for signal fading in communications engineering |
| **Skellam** | Univariate Discrete | Statistics, Sports Analytics | Applied Statistics | Yes (skellam) | Low | Direct wrap; difference of two Poissons |
| **Fréchet (Extreme Value Type II)** | Univariate Continuous | Finance, Engineering | Extreme Value Theory | Yes (invweibull) | Low | Direct wrap, though scipy's naming is a trap: it's called invweibull, not frechet (the old frechet_r/l names were deprecated) |
| **Log-Logistic (Fisk)** | Univariate Continuous | Actuarial Science, Economics | Survival Analysis, Loss Models | Yes (fisk) | Low | Direct wrap; survival-analysis alternative to Weibull |
| **Benford's Law** | Univariate Discrete | Statistics, Forensic Accounting, Data Science | Intro Stats, Forensic Accounting, Data Science Electives | No | Low | Classic fraud-detection/applied-stats teaching example; simple closed-form pmf P(d)=log10(1+1/d) |
| **Poisson Binomial** | Univariate Discrete | Statistics, Data Science | Applied Statistics, Survey Sampling | No | Medium | Sum of independent non-identical Bernoullis; pmf needs convolution/recursive algorithm, not simply closed-form |
| **Zeta** | Univariate Discrete | Statistics, Number Theory | Applied Statistics, Power-Law Phenomena | Yes (zipf) | Low | Infinite-support limiting case of Zipf; scipy's naming is a trap (zipf=zeta, zipfian=finite Zipf) worth flagging directly |
| **Noncentral Hypergeometric (Fisher's / Wallenius')** | Univariate Discrete | Statistics, Genetics, Ecology | Applied Statistics, Biostatistics | Yes (nchypergeom_fisher, nchypergeom_wallenius) | Low-Medium | Models biased sampling without replacement; two related variants, some explanation needed to distinguish them |
| **Negative Multinomial** | Multivariate | Statistics | Multivariate Statistics | No | Medium | Multivariate generalization of Negative Binomial, the way Multinomial generalizes Binomial |
| **Degenerate / Point-Mass** | Univariate Discrete | Probability Theory | Intro Probability | No | Low | Trivial constant-outcome distribution; useful building block for mixtures and edge-case teaching |
| **Birthday Distribution** | Univariate Discrete (combinatorial) | Probability Theory | Intro Probability, Combinatorics | No | Medium | Classic 'birthday problem' teaching distribution; likely more valuable as a worked example than a formal class |
| **Coupon Collector Distribution** | Univariate Discrete (combinatorial) | Probability Theory | Intro Probability, Combinatorics | No | Medium | Classic combinatorial waiting-time problem; pairs pedagogically with Birthday Distribution |
| **Arcsine** | Univariate Continuous | Statistics, Probability Theory | Probability Theory, Random Walks | Yes (arcsine) | Low | Beta(1/2,1/2) special case; appears in random walk / order statistics theory |
| **Chi** | Univariate Continuous | Statistics, Engineering | Mathematical Statistics | Yes (chi) | Low | Square root of Chi-square; parent distribution of Rayleigh and Maxwell-Boltzmann |
| **Noncentral Beta** | Univariate Continuous | Statistics | Power Analysis (R²/partial correlation tests) | No | Medium-High | Confirmed absent from scipy; used in power analysis for R² and partial-correlation tests |
| **Exponential Power / Generalized Normal** | Univariate Continuous | Statistics, Signal Processing | Applied Statistics | Yes (exponpow, gennorm) | Low | Unifies Normal/Laplace/Uniform-like tail behavior via one shape parameter |
| **Beta Prime / Inverted Beta** | Univariate Continuous | Statistics | Mathematical Statistics, Bayesian Statistics | Yes (betaprime) | Low | Related to F-distribution; used in Bayesian variance modeling |
| **Power / Power-Function** | Univariate Continuous | Statistics, Reliability | Applied Statistics | Yes (powerlaw) | Low | Beta(a,1) special case; simple but distinct named distribution worth surfacing directly |
| **Two-Sided Power (TSP)** | Univariate Continuous | Statistics, Simulation | Simulation, Risk Analysis | No | Medium | Generalizes Triangular the way GEV generalizes Gumbel; natural extension once Triangular exists |
| **Generalized Beta of the Second Kind (GB2)** | Univariate Continuous | Actuarial Science, Economics | Loss Models, Income Distribution Modeling (grad) | No | Medium-High | 4-parameter umbrella family over Burr/Singh-Maddala/Lomax already in this table |
| **Dagum** | Univariate Continuous | Actuarial Science, Economics | Income Distribution Modeling, Loss Models | No | Medium | Income-distribution family sitting alongside Burr/Singh-Maddala already in this table |
| **Folded-Normal** | Univariate Continuous | Statistics, Engineering | Applied Statistics | Yes (foldnorm) | Low | \|X\| where X~Normal(μ,σ) with μ≠0; distinct from Half-Normal already in this table, which assumes μ=0 |
| **Maxwell-Boltzmann** | Univariate Continuous | Physics, Engineering | Statistical Mechanics, Physics-Stats Crossover | Yes (maxwell) | Low | Classic physics speed distribution; special case of Chi with 3 degrees of freedom |
| **Wigner Semicircle** | Univariate Continuous | Statistics, Physics | Random Matrix Theory (grad) | Yes (semicircular) | Low | Foundational random matrix theory result; notable for grad-level probability |
| **Trapezoidal** | Univariate Continuous | Statistics, Simulation | Simulation, Risk Analysis | Yes (trapezoid) | Low | Generalizes Triangular/Uniform; common in simulation modeling |
| **Generalized Logistic** | Univariate Continuous | Statistics | Applied Statistics, Extreme Value Theory | Yes (genlogistic) | Low | Skewed generalization of Logistic already in this table |
| **Marshall–Olkin Exponential** | Univariate Continuous (bivariate construction) | Engineering, Reliability | Reliability Engineering (grad) | No | Medium-High | Models dependent component failure times via a shock-model construction; notable in reliability engineering |
| **Continuous Bernoulli** | Univariate Continuous | Machine Learning, Statistics | ML/Data Science | No | Low-Medium | Modern ML use (VAE probabilistic pixel outputs); relevant for ML-adjacent stats courses |
| **Logit-Normal** | Univariate Continuous | Statistics, Bayesian Stats, ML | Bayesian Statistics, ML | No | Low-Medium | Common for modeling probabilities in Bayesian/ML contexts |
| **von Mises–Fisher** | Multivariate (directional) | Statistics, Engineering | Directional/Circular Statistics (grad) | Yes (vonmises_fisher) | Medium | Spherical generalization of Von Mises already in this table, to higher dimensions |
| **Order Statistics / Finite Order Statistic** | Modifier/Feature (derived from any distribution) | Statistics | Mathematical Statistics | No (not a distribution per se) | Low-Medium | Distribution of the k-th order statistic of n iid draws; likely already achievable via existing RV transformations in Symbulate — worth confirming before building anything new |
| **Test-Statistic Sampling Distributions (KS, Kolmogorov, Anderson-Darling)** | Univariate Continuous (special-purpose) | Statistics | Mathematical Statistics, Goodness-of-Fit Testing | Yes (ksone, kstwobign, kstwo, anderson) | Low | A different category from modeling distributions — these are sampling distributions of goodness-of-fit test statistics; direct scipy wraps |

## Tier 4 — Lower Priority

| Distribution | Type | Field(s) | Typical Course(s) | scipy.stats Availability | Symbulate Integration Effort | Notes / Rationale |
|---|---|---|---|---|---|---|
| **Von Mises** | Univariate Continuous | Statistics, Engineering | Directional/Circular Statistics | Yes (vonmises / vonmises_line) | Low-Medium | scipy has it, but circular support (wraparound at 2π) needs care in plotting/spinner display, which assume linear support |
| **Nakagami** | Univariate Continuous | Engineering | Wireless Signal Modeling | Yes (nakagami) | Low | Direct wrap; domain-specific to wireless/physics applications |
| **Lévy** | Univariate Continuous | Physics, Finance | Stochastic Processes | Yes (levy / levy_l) | Low | Direct wrap; special case of stable distributions |
| **Conway-Maxwell-Poisson** | Univariate Discrete | Statistics | Advanced Count-Data Modeling | No | High | Not in scipy; normalizing constant has no closed form and requires numerical summation, complicating both pmf evaluation and sampling |
| **Yule-Simon** | Univariate Discrete | Statistics, Network Science | Advanced Count-Data Modeling | Yes (yulesimon) | Low | Direct wrap; power-law discrete model |
| **Discrete Weibull** | Univariate Discrete | Statistics, Engineering | Advanced Reliability | No | Medium-High | NOT in scipy despite appearances: scipy.stats.dweibull is the continuous 'double Weibull,' not the discrete Weibull — a naming trap worth flagging for whoever implements this; needs a fully custom pmf/cdf |
| **Logarithmic (Log-Series)** | Univariate Discrete | Statistics, Ecology | Advanced Count-Data Modeling | Yes (logser) | Low | Direct wrap; specialized count model, occasionally in ecology/biostatistics |
| **Variance-Gamma** | Univariate Continuous | Finance | Quantitative Finance (grad) | No | High | Not in scipy; density involves a modified Bessel function with no simple closed form, and no ready sampling routine |
| **Normal-Inverse Gaussian (NIG)** | Univariate Continuous | Finance | Quantitative Finance (grad) | Yes (norminvgauss) | Low-Medium | scipy has it, but it's a 4-parameter family that takes care to present intuitively to students |
| **Stable (Lévy-stable)** | Univariate Continuous | Finance, Statistics | Quantitative Finance, Probability Theory (grad) | Yes (levy_stable) | Medium-High | scipy has it, but its own docs note the general (non-special-case) implementation is numerically delicate and slow |
| **Generalized Hyperbolic** | Univariate Continuous | Finance | Quantitative Finance (grad) | Yes (genhyperbolic) | Medium | scipy has it, but many parameters and subtle special-case relationships to Variance-Gamma/NIG need careful explanation for students |
| **Johnson SU / SB** | Univariate Continuous | Finance, Engineering | Applied Statistics, Distribution Fitting | Yes (johnsonsu, johnsonsb) | Low | Direct wrap; flexible skew/kurtosis family for fitting non-normal data |
| **Gaussian Copula** | Multivariate / Dependence | Finance, Statistics | Risk Management, Quantitative Finance (grad) | No | High | No copula support anywhere in scipy.stats; requires building joint-CDF construction via a correlation matrix plus marginal transforms from scratch |
| **t-Copula** | Multivariate / Dependence | Finance | Quantitative Finance (grad) | No | High | Same gap as Gaussian Copula, plus needs multivariate-t machinery underneath |
| **Matrix Normal** | Multivariate | Statistics | Advanced Multivariate Statistics (grad) | Yes (matrix_normal) | Medium-High | scipy has it, but matrix-valued output has the same display/summary challenge as Wishart |
| **Phase-Type Distributions** | Univariate Continuous (class) | Actuarial Science, Operations Research | Ruin Theory, Queueing Theory (grad) | No | High | Not in scipy; requires building general sub-generator-matrix machinery from scratch, a substantial standalone project |
| **Singh-Maddala** | Univariate Continuous | Economics, Actuarial Science | Income Distribution Modeling | Yes (via burr12 reparameterization) | Low-Medium | Not a distinct scipy distribution, but a direct reparameterization of Burr XII, so implementation is mostly a relabeling exercise |
| **Champernowne** | Univariate Continuous | Economics | Income Distribution Modeling (elective) | No | Medium | Not in scipy, but has a known closed-form CDF, so custom implementation is straightforward even though not a wrap |
| **Rademacher** | Univariate Discrete | Probability Theory, Statistics | Intro Probability | No | Low | Trivial variant of Bernoulli (±1, p=.5); useful in random-sign/Monte Carlo contexts |
| **Zipf-Mandelbrot** | Univariate Discrete | Statistics, Linguistics, Information Theory | Applied Statistics, NLP | No | Medium | Generalizes Zipf with an added parameter; needs custom normalizing-constant computation |
| **Discrete ArcSine** | Univariate Discrete | Probability Theory | Random Walks, Stochastic Processes | No | Medium | Discrete counterpart to continuous Arcsine; pmf derived from random-walk sign-change theory |
| **Boltzmann** | Univariate Discrete | Physics, Statistics | Statistical Mechanics, Physics-Stats Crossover | No | Low | Statistical-physics distribution over discrete energy levels |
| **Borel** | Univariate Discrete | Probability Theory, Operations Research | Branching Processes, Queueing Theory (grad) | No | Medium | Arises from branching processes and busy-period queueing analysis |
| **Matching Distribution** | Univariate Discrete (combinatorial) | Probability Theory | Intro Probability, Combinatorics | No | Medium | Classic derangement/matching problem distribution |
| **Noncentral Chi** | Univariate Continuous | Statistics, Engineering | Mathematical Statistics, Signal Processing | No | Medium-High | Less common than noncentral chi-square, but a real remaining gap once that one is added |
| **Doubly Noncentral t** | Univariate Continuous | Statistics | Advanced Mathematical Statistics (grad) | No | High | Both numerator and denominator noncentral; no standard package support, advanced power-analysis edge case |
| **Doubly Noncentral F** | Univariate Continuous | Statistics | Advanced Mathematical Statistics (grad) | No | High | Same rationale as Doubly Noncentral t |
| **Hyperbolic-Secant** | Univariate Continuous | Statistics | Applied Statistics | Yes (hypsecant) | Low | Logistic-like shape; occasional alternative to Normal |
| **Marchenko–Pastur** | Univariate Continuous | Statistics, Physics | Random Matrix Theory (grad) | No | Medium-High | Pairs with Wigner Semicircle in random matrix theory; density is piecewise depending on aspect-ratio parameter |
| **Tracy–Widom** | Univariate Continuous | Statistics, Physics | Random Matrix Theory (grad) | No | High | Same cluster as Semicircle/Marchenko-Pastur; no closed-form density, defined via Painlevé differential equations |
| **U-Quadratic** | Univariate Continuous | Statistics | Applied Statistics | No | Low-Medium | Bathtub-shaped density on a bounded interval; niche but simple |
| **Fisher's z-distribution** | Univariate Continuous | Statistics | Mathematical Statistics (historical/classical) | No | Low-Medium | Classical transform of F; historically important but rarely used directly today |
| **Hyperbolic (general family)** | Univariate Continuous | Finance | Quantitative Finance (grad) | No | High | Umbrella family for NIG and Variance-Gamma already in this table; the general family is harder to implement than its special cases |
| **Voigt Profile** | Univariate Continuous | Physics, Engineering | Spectroscopy, Signal Processing | Yes (voigt_profile) | Low | Convolution of Normal+Cauchy; used in spectroscopy and engineering signal analysis |
| **Wakeby** | Univariate Continuous | Hydrology, Environmental Statistics | Environmental/Hydrology Statistics (grad) | No | High | Flood-frequency modeling; 5-parameter family with no simple closed-form density, defined via quantile function |
| **Matrix t** | Multivariate (matrix-valued) | Statistics | Advanced Multivariate Statistics (grad) | No | High | Matrix-variate generalization, natural companion to Matrix Normal already in this table; grad-level |

---

*Generated from `symbulate_distribution_additions.xlsx`, cross-checked directly against `scipy.stats` (version 1.17.1) rather than from memory. Priority tiers reflect real-world usage frequency across fields, not scipy availability. Excludes a separately-noted 'skip-tier' of extremely narrow or superseded distributions (e.g., physics phase functions, single-purpose combinatorics curiosities) that were surfaced during the cross-reference but judged not worth adding.*