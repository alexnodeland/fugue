#!/usr/bin/env python3
"""Standalone (stdlib-only) reference-value generator for FG-14.

No network / scipy available in this sandbox, so this independently
re-derives the special functions needed (regularized incomplete gamma P(a,x)
and regularized incomplete beta I_x(a,b)) via the SAME Numerical-Recipes
continued-fraction algorithm the Rust test file uses, and then cross-checks
every value against a numerical-integration ground truth (adaptive Simpson's
rule directly on the analytic PDF, which does not depend on any special
function at all). Any value printed below has therefore been verified two
independent ways.
"""
import math

def lgamma(x):
    return math.lgamma(x)

def gser(a, x):
    gln = lgamma(a)
    ap = a
    s = 1.0 / a
    d = s
    for _ in range(500):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-15:
            break
    return s * math.exp(-x + a * math.log(x) - gln)

def gcf(a, x):
    gln = lgamma(a)
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h

def gammp(a, x):
    if x < 0 or a <= 0:
        raise ValueError
    if x == 0:
        return 0.0
    if x < a + 1.0:
        return gser(a, x)
    else:
        return 1.0 - gcf(a, x)

def betacf(a, b, x):
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h

def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    else:
        return 1.0 - bt * betacf(b, a, 1 - x) / b

def simpson_cdf(pdf, lo, x, n=2_000_000):
    """Composite Simpson's rule integral of pdf over [lo, x]. n must be even."""
    if x <= lo:
        return 0.0
    if n % 2 == 1:
        n += 1
    h = (x - lo) / n
    total = pdf(lo) + pdf(x)
    for i in range(1, n):
        xi = lo + i * h
        total += (4 if i % 2 == 1 else 2) * pdf(xi)
    return total * h / 3.0

# --- Distribution PDFs (matching src/core/distribution.rs formulas exactly) ---

def gamma_pdf(shape, rate):
    c = shape * math.log(rate) - lgamma(shape)
    def f(x):
        if x <= 0:
            return 0.0
        return math.exp(c + (shape - 1) * math.log(x) - rate * x)
    return f

def beta_pdf(a, b):
    c = lgamma(a + b) - lgamma(a) - lgamma(b)
    def f(x):
        if x <= 0 or x >= 1:
            return 0.0
        return math.exp(c + (a - 1) * math.log(x) + (b - 1) * math.log(1 - x))
    return f

def studentt_pdf(df, loc, scale):
    c = lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * math.log(df * math.pi) - math.log(scale)
    def f(x):
        z = (x - loc) / scale
        return math.exp(c - 0.5 * (df + 1) * math.log1p(z * z / df))
    return f

def chi2_pdf(k):
    return gamma_pdf(k / 2, 0.5)

def invgamma_pdf(shape, rate):
    c = shape * math.log(rate) - lgamma(shape)
    def f(x):
        if x <= 0:
            return 0.0
        return math.exp(c - (shape + 1) * math.log(x) - rate / x)
    return f

def check(name, closed_form, numeric, tol=1e-6):
    diff = abs(closed_form - numeric)
    status = "OK" if diff < tol else "MISMATCH"
    print(f"{status:9s} {name:40s} closed={closed_form!r:24s} numeric={numeric!r:24s} diff={diff:.3e}")

if __name__ == "__main__":
    print("== Gamma(shape=2, rate=1).cdf(1.0) ==")
    check("gamma(2,1).cdf(1.0)", gammp(2.0, 1.0 * 1.0), simpson_cdf(gamma_pdf(2.0, 1.0), 0.0, 1.0))

    print("== Gamma(shape=3, rate=0.5).cdf(4.0) ==")
    check("gamma(3,0.5).cdf(4.0)", gammp(3.0, 0.5 * 4.0), simpson_cdf(gamma_pdf(3.0, 0.5), 0.0, 4.0))

    print("== ChiSquared(k=4).cdf(2.0) (= Gamma(2, 0.5)) ==")
    check("chi2(4).cdf(2.0)", gammp(2.0, 1.0), simpson_cdf(chi2_pdf(4.0), 0.0, 2.0))

    print("== ChiSquared(k=2).cdf(3.0) closed form 1-exp(-1.5) ==")
    check("chi2(2).cdf(3.0)", 1 - math.exp(-1.5), gammp(1.0, 1.5))

    print("== Beta(2,3).cdf(0.5) ==")
    check("beta(2,3).cdf(0.5)", betai(2.0, 3.0, 0.5), simpson_cdf(beta_pdf(2.0, 3.0), 0.0, 0.5))

    print("== Beta(5,2).cdf(0.7) ==")
    check("beta(5,2).cdf(0.7)", betai(5.0, 2.0, 0.7), simpson_cdf(beta_pdf(5.0, 2.0), 0.0, 0.7))

    print("== Beta(1,1).cdf(0.37) should equal 0.37 ==")
    check("beta(1,1).cdf(0.37)", betai(1.0, 1.0, 0.37), 0.37)

    print("== StudentT(df=5,0,1).cdf(1.5) ==")
    # CDF via incomplete beta relation: t>=0 -> 1 - 0.5*I_{df/(df+t^2)}(df/2,1/2)
    df, t = 5.0, 1.5
    xarg = df / (df + t * t)
    cdf_closed = 1 - 0.5 * betai(df / 2, 0.5, xarg)
    check("studentt(5).cdf(1.5)", cdf_closed, simpson_cdf(studentt_pdf(5.0, 0.0, 1.0), -50.0, 1.5))

    print("== StudentT(df=1) should match Cauchy cdf: 0.5 + atan(t)/pi at t=2.0 ==")
    df, t = 1.0, 2.0
    xarg = df / (df + t * t)
    cdf_closed = 1 - 0.5 * betai(df / 2, 0.5, xarg)
    cauchy_closed = 0.5 + math.atan(t) / math.pi
    check("studentt(1).cdf(2.0) vs cauchy", cdf_closed, cauchy_closed)

    print("== InverseGamma(3,2).cdf(1.5) : P(X<=x) = Q(shape, rate/x) = 1-P(shape,rate/x) ==")
    shape, rate, x = 3.0, 2.0, 1.5
    cdf_closed = 1.0 - gammp(shape, rate / x)
    check("invgamma(3,2).cdf(1.5)", cdf_closed, simpson_cdf(invgamma_pdf(shape, rate), 1e-9, x))

    print("== InverseGamma(2.5, 1.0).cdf(0.8) ==")
    shape, rate, x = 2.5, 1.0, 0.8
    cdf_closed = 1.0 - gammp(shape, rate / x)
    check("invgamma(2.5,1).cdf(0.8)", cdf_closed, simpson_cdf(invgamma_pdf(shape, rate), 1e-9, x))
