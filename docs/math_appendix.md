# LiftLab Math Appendix

> Each estimator's derivation is filled in as it is implemented. The goal: every
> formula here is matched by a test against an analytic result or statsmodels.

## 0. Synthetic experiment design & ground truth (Phase 2)

LiftLab is **semi-synthetic**: units come from the real (or synthetic-stand-in)
population, but the treatment effect is **injected**, so the true effect is *known*.

Let $X_i$ be unit $i$'s pre-period spend (`pre_period_value`) and
$z_i = (X_i - \bar X)/\mathrm{sd}(X)$ its standardization (population sd, so
$\operatorname{Var}(z)=1$). Assignment $T_i \sim \mathrm{Bernoulli}(r)$, independent of $z$.

**Continuous outcome** (revenue), with target covariate correlation $\rho$ and scale $s$:
$$Y_i = \mu + \beta z_i + \tau T_i + \varepsilon_i,\quad \beta=\rho s,\ \ \varepsilon_i\sim N\!\big(0,(1-\rho^2)s^2\big).$$
Because $z\perp T$, $z\perp\varepsilon$ and $\operatorname{Var}(z)=1$:
$$\operatorname{Cov}(z,Y)=\beta=\rho s,\qquad \operatorname{Var}(Y)=\beta^2+(1-\rho^2)s^2+\tau^2 r(1-r)=s^2+\tau^2 r(1-r).$$
So $\operatorname{corr}(z,Y)=\rho/\sqrt{1+\tau^2 r(1-r)/s^2}\approx\rho$ (exact when $\tau=0$).
With the default $\tau=2,\,s=40,\,r=0.5$ the dilution factor is $\sqrt{1+1/1600}\approx1.0003$, i.e.
corr $\approx 0.5998$. The **true ATE is exactly $\tau$** ($E[Y\mid T{=}1]-E[Y\mid T{=}0]=\tau$).

Since CUPED's variance reduction is $\rho^2$ (§3), corr $\approx 0.6 \Rightarrow$ reduction
$\approx 36\% \ge 30\%$ **by construction**, verified at 36.0% in the Monte-Carlo eval (35.5% on a single draw) on the zero-inflated covariate.

**Binary outcome** (conversion): $p_i=\mathrm{base}+\tau_p T_i+\delta z_i$, clipped to $(0,1)$,
$Y_i\sim\mathrm{Bernoulli}(p_i)$. As $E[z]=0$, the **true ATE is exactly $\tau_p$** (clipping is
negligible at the configured values: $p$ stays well inside $(0,1)$).

Monte-Carlo replicates (Phase 5) hold the units fixed and re-randomize $T$ and redraw
$\varepsilon$ with $\text{seed}=\text{base}+i$, valid frequentist replication of the experiment's
own randomness, conditional on the realized population.

## 1. Power / MDE (Phase 3)

Normal-approximation, two-sided, at level $\alpha$ and power $1-\beta$. Let
$z_a=z_{1-\alpha/2}$, $z_b=z_{\text{power}}$.

**Two means.** With per-unit sd $\sigma$ and group sizes $n_T,n_C$, the SE is
$\mathrm{SE}=\sigma\sqrt{1/n_T+1/n_C}$ and
$$\text{power}=\Phi\!\Big(\tfrac{|\Delta|}{\mathrm{SE}}-z_a\Big)+\Phi\!\Big(\tfrac{-|\Delta|}{\mathrm{SE}}-z_a\Big),\qquad \mathrm{MDE}=(z_a+z_b)\,\mathrm{SE}.$$
Inverting for the total $N$ with treatment fraction $r$:
$$N=\Big(\tfrac{(z_a+z_b)\sigma}{\mathrm{MDE}}\Big)^2\Big(\tfrac1r+\tfrac1{1-r}\Big).$$
Validated to match `statsmodels` `NormalIndPower` (exact) and `TTestIndPower` (≈, at large $N$).

**Two proportions.** Using the pooled variance under $H_0$ and the unpooled variance under $H_1$
(consistent with the z-test below):
$\mathrm{SE}_0=\sqrt{\bar p(1-\bar p)(1/n_T+1/n_C)}$, $\mathrm{SE}_1=\sqrt{\tfrac{p_T(1-p_T)}{n_T}+\tfrac{p_C(1-p_C)}{n_C}}$,
$$\text{power}=\Phi\!\Big(\tfrac{|\Delta|-z_a\,\mathrm{SE}_0}{\mathrm{SE}_1}\Big)+\Phi\!\Big(\tfrac{-|\Delta|-z_a\,\mathrm{SE}_0}{\mathrm{SE}_1}\Big).$$
The MDE and required $N$ are found by solving $\text{power}=1-\beta$ numerically (Brent's method).
Validated against a Monte-Carlo rejection-rate simulation.

## 2. Two-sample test (Phase 3)

**Welch's t-test (continuous).** $\hat\Delta=\bar Y_T-\bar Y_C$,
$\mathrm{SE}=\sqrt{s_T^2/n_T+s_C^2/n_C}$, Welch–Satterthwaite dof
$$\nu=\frac{(s_T^2/n_T+s_C^2/n_C)^2}{\frac{(s_T^2/n_T)^2}{n_T-1}+\frac{(s_C^2/n_C)^2}{n_C-1}},\quad t=\hat\Delta/\mathrm{SE},\quad \text{CI}=\hat\Delta\pm t_{1-\alpha/2,\nu}\,\mathrm{SE}.$$
The $t$ critical value (vs $z$) makes coverage $\ge$ nominal. Matches `scipy.stats.ttest_ind(equal_var=False)`.

**Two-proportion z-test (binary).** $\hat\Delta=\hat p_T-\hat p_C$. The **CI** uses the unpooled
(Wald) SE $\sqrt{\hat p_T(1-\hat p_T)/n_T+\hat p_C(1-\hat p_C)/n_C}$; the **p-value** uses the pooled SE
$\sqrt{\bar p(1-\bar p)(1/n_T+1/n_C)}$ with $\bar p=(x_T+x_C)/(n_T+n_C)$, the standard test of
$H_0:p_T=p_C$ (correct Type-I error, which the A/A gate relies on). Matches `statsmodels.proportions_ztest`.

## 3. CUPED (Phase 4)

With a pre-period covariate $X$ (unaffected by treatment), define the adjusted metric
$$Y^{\text{cuped}} = Y - \theta\,(X - \bar X),\qquad \theta=\frac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)}.$$
Because $E[X-\bar X]=0$ and $X\perp T$ (pre-treatment), $\theta$ can be estimated on the pooled
sample without biasing the ATE: the adjusted difference in means equals the raw difference minus
$\theta(\bar X_T-\bar X_C)$, and the covariate is balanced in expectation. The variance shrinks:
$$\operatorname{Var}(Y^{\text{cuped}})=\operatorname{Var}(Y)-\frac{\operatorname{Cov}(Y,X)^2}{\operatorname{Var}(X)}=\operatorname{Var}(Y)\,(1-\rho^2),\quad \rho=\operatorname{corr}(Y,X).$$
So the reported reduction $1-\operatorname{Var}(Y^{\text{cuped}})/\operatorname{Var}(Y)=\rho^2$. With the
calibrated $\rho=0.6$, that is $36\%$, verified at $36.0\%$ (eval; $35.5\%$ single draw). The CUPED CI uses
Welch's t-test on $Y^{\text{cuped}}$; at the experiment's $N$ the cost of estimating $\theta$ is
negligible, so coverage stays $\approx 95\%$ (validated by simulation).

## 4. Sample-Ratio Mismatch (Phase 4)

A chi-square goodness-of-fit test of the realized split against the intended ratio $r$. With observed
$(n_C,n_T)$ and expected $(N(1-r),Nr)$, $N=n_C+n_T$:
$$\chi^2=\sum_{g\in\{C,T\}}\frac{(\text{obs}_g-\text{exp}_g)^2}{\text{exp}_g}\ \sim\ \chi^2_1,\qquad p=P(\chi^2_1>\chi^2).$$
Flag an SRM when $p<\alpha_{\text{SRM}}$ (default $0.001$, strict, because a false alarm needlessly
discards a valid experiment). Matches `scipy.stats.chisquare`.

## 5. Causal fallback (Phase 6)

When randomization breaks (treatment depends on a confounder), the naive post-period difference is
biased by $\lambda(\bar z_T-\bar z_C)\neq 0$. Two observational estimators recover the ATE:

**Difference-in-differences (DiD).** With pre/post outcomes $y_{\text{pre}},y_{\text{post}}$,
$$\hat\tau_{\text{DiD}}=(\bar y^{\,T}_{\text{post}}-\bar y^{\,T}_{\text{pre}})-(\bar y^{\,C}_{\text{post}}-\bar y^{\,C}_{\text{pre}}).$$
Identifies $\tau$ under **parallel trends**: absent treatment, both arms would have moved equally
(the confounder's effect is time-invariant, so it cancels in the within-unit differencing).

**Inverse-propensity weighting (IPW).** Fit $\hat e(x)=P(T{=}1\mid x)$ by logistic regression and use
the stabilized (Hajek) estimator
$$\hat\tau_{\text{IPW}}=\frac{\sum_i \tfrac{T_i y_i}{\hat e(x_i)}}{\sum_i \tfrac{T_i}{\hat e(x_i)}}-\frac{\sum_i \tfrac{(1-T_i) y_i}{1-\hat e(x_i)}}{\sum_i \tfrac{1-T_i}{1-\hat e(x_i)}}.$$
Identifies $\tau$ under **unconfoundedness** (no unobserved confounders) and **overlap**
($0<\hat e(x)<1$; we trim to $[0.02,0.98]$). CI by nonparametric bootstrap.

**Sensitivity / refutation.** A placebo treatment (permuted $T$) should yield $\approx 0$; DoWhy's
random-common-cause and data-subset refuters should leave the estimate stable. DoWhy's propensity
estimate is reported as an independent cross-check of the hand-rolled IPW.
