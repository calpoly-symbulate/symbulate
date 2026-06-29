# Degenerate Distribution Examples

A **degenerate distribution** is one where a boundary parameter value makes the outcome deterministic — every draw returns the same value. These cases now work as expected in symbulate instead of raising errors or returning wrong results.

---

## Poisson

`Poisson(lam=0)` — zero average events means the count is always 0.

```python
X = Poisson(lam=0)
X.sim(5)   # → [0, 0, 0, 0, 0]
```

---

## Bernoulli

`Bernoulli(p=0)` — impossible event, always 0.

```python
X = RV(Bernoulli(p=0))
X.sim(5)   # → [0, 0, 0, 0, 0]
```

`Bernoulli(p=1)` — certain event, always 1.

```python
X = RV(Bernoulli(p=1))
X.sim(5)   # → [1, 1, 1, 1, 1]
```

---

## Binomial

`Binomial(n=10, p=0)` — zero probability of success, always 0.

```python
X = RV(Binomial(n=10, p=0))
X.sim(5)   # → [0, 0, 0, 0, 0]
```

`Binomial(n=10, p=1)` — certain success every trial, always n.

```python
X = RV(Binomial(n=10, p=1))
X.sim(5)   # → [10, 10, 10, 10, 10]
```

`Binomial(n=0, p=0.5)` — zero trials, always 0.

```python
X = RV(Binomial(n=0, p=0.5))
X.sim(5)   # → [0, 0, 0, 0, 0]
```

---

## Geometric

`Geometric(p=1)` — success on the very first trial, always 1.

```python
X = Geometric(p=1)
X.sim(5)   # → [1, 1, 1, 1, 1]
```

---

## NegativeBinomial / Pascal

`NegativeBinomial(r=10, p=1)` — every trial succeeds, so exactly r trials needed, always r.

```python
X = NegativeBinomial(r=10, p=1)
X.sim(5)   # → [10, 10, 10, 10, 10]
```

`Pascal(r=10, p=1.0)` — every trial is a success, so 0 failures before r successes, always 0.

```python
X = Pascal(r=10, p=1.0)
X.sim(5)   # → [0, 0, 0, 0, 0]
```

---

## DiscreteUniform

`DiscreteUniform(a, b)` with `a == b` — only one possible value.

```python
X = DiscreteUniform(3, 3)
X.sim(5)   # → [3, 3, 3, 3, 3]
```

---

## LogNormal

`LogNormal(mu=0, sigma=0)` — zero spread means the value is always `exp(mu)`.

```python
X = LogNormal(mu=0, sigma=0)
X.sim(5)   # → [1.0, 1.0, 1.0, 1.0, 1.0]  (exp(0) = 1)

X = LogNormal(mu=1, sigma=0)
X.sim(5)   # → [e, e, e, e, e]  (exp(1) ≈ 2.718)
```
