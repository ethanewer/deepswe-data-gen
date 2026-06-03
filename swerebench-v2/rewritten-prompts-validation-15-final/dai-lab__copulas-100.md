# dai-lab__copulas-100

- repo: DAI-Lab/Copulas
- language: python
- difficulty: hard

## Rewritten Prompt

Improve serialization for the library’s univariate distributions so fitted instances of the same class always serialize to the same structure, regardless of any internal constant-value state. The serialized form should not expose `constant_value` directly; instead, constant distributions must encode that information implicitly in their normal serialized parameters.

Update the public behavior of the univariate wrappers and distributions so deserialization can reconstruct both constant and non-constant fitted models from the new serialized dictionaries. This includes the constructor and serialization/deserialization behavior for `ScipyWrapper.__init__`, `GaussianUnivariate.__init__`, `GaussianUnivariate._fit_params`, `GaussianUnivariate.from_dict`, `GaussianKDE.__init__`, `GaussianKDE.fit`, `GaussianKDE._fit_params`, `GaussianKDE.from_dict`, `KDEUnivariate.__init__`, `KDEUnivariate.fit`, `KDEUnivariate._fit_params`, `KDEUnivariate.from_dict`, `TruncNorm.fit`, `TruncNorm.from_dict`, and `TruncNorm._fit_params`.

Preserve compatibility for the existing fitted-state restoration and return shapes, while ensuring constant distributions are represented only through their serialized parameters: Gaussian univariates should round-trip constant values through `mean`/`std`, KDE-based distributions should serialize only the dataset needed to rebuild the model, and truncated normals should encode constants via identical bounds.

## Preserved Requirements

- Serialized output must be identical for all fitted distributions of the same class, independent of internal constant-value state.
- `constant_value` must not be exposed directly in the serialized form.
- Constant distributions must encode their constant status implicitly in the serialized parameters.
- `GaussianUnivariate.__init__` should initialize with undefined `mean` and `std` until fitted.
- `GaussianUnivariate._fit_params` must return constant `mean`/`std=0` for constant columns and normal fitted parameters otherwise.
- `GaussianUnivariate.from_dict` must restore fitted state and reconstruct constant distributions when `std` is 0.
- `GaussianKDE.__init__` must accept optional `sample_size` and store it.
- `GaussianKDE.fit` must detect constant data, optionally mask data with sampled points when `sample_size` is provided, fit the KDE model, set `sample` to the model’s resampling method, and mark the instance fitted.
- `GaussianKDE._fit_params` must serialize only the essential dataset, using repeated constant values when constant, and exclude internal SciPy fields.
- `GaussianKDE.from_dict` must restore fitted state and reconstruct constant or non-constant KDEs from the dataset without requiring `constant_value`.
- `KDEUnivariate.__init__` must accept optional `sample_size` with default 10 and store it.
- `KDEUnivariate.fit` must detect constants, optionally fit from a fresh sample when `sample_size` is provided, otherwise fit directly, and mark the instance fitted.
- `KDEUnivariate._fit_params` must serialize only the essential dataset, using repeated constant values when constant, and exclude internal SciPy fields.
- `KDEUnivariate.from_dict` must restore fitted state and reconstruct constant or non-constant KDEs from the dataset without requiring `constant_value`.
- `TruncNorm.fit` must detect constant columns, set `constant_value`, replace sampling/scoring behavior for constants, otherwise compute bounds and fit normally, and mark the instance fitted.
- `TruncNorm._fit_params` must serialize constants with identical `a` and `b` values and otherwise include the model parameters.
- `TruncNorm.from_dict` must restore fitted state and reconstruct constant distributions when `a == b`.
- `ScipyWrapper.__init__` must forward arbitrary positional and keyword arguments to the base univariate constructor.

## Removed Noise

- Issue/PR/template boilerplate
- External URLs
- PR/test references
- Implementation hints about where to change code
- File path and source location details
- Metadata about confidence, difficulty, or changed files
- Redundant repeated wording about the same serialization goal

## Risk Notes

- The exact serialized key set for non-constant KDE/truncated normal objects must remain compatible with existing callers except for the removal of `constant_value`.
- Round-tripping constant KDE and truncated normal instances depends on how the repository represents fitted state and model reconstruction today.
- The default `sample_size=10` for `KDEUnivariate` is part of the public behavior and should be preserved.
- Gaussian univariate constant handling must remain distinguishable through the serialized `mean`/`std` pair.

## Original Prompt

Improve serialization of univariate distributions
The current implementation of the `to_dict` and ` form_dict` methods exposes the internal attribute `constant_value` and returns different values depending on its value.

We should change the implementation to make that all fitted distributions of the same class have the exact same serialization independently of the value of the `constant_value` attribute, and also have the attribute `constant_value` implicitly codified in the serialization.

## Original Interface

Method: ScipyWrapper.__init__(self, *args, **kwargs)
Location: copulas/univariate/base.py → class ScipyWrapper(Univariate)
Inputs: any positional and keyword arguments forwarded to Univariate.__init__ (typically none).
Outputs: an initialized ScipyWrapper instance; no return value.
Description: Updated constructor now forwards arbitrary *args/**kwargs to the base Univariate class, enabling subclasses to inject additional parameters (e.g., sample_size) without breaking the inheritance chain.

Method: GaussianUnivariate.__init__(self, *args, **kwargs)
Location: copulas/univariate/gaussian.py → class GaussianUnivariate(Univariate)
Inputs: any positional/keyword arguments passed to Univariate.__init__.
Outputs: a GaussianUnivariate instance with attributes `mean=None` and `std=None` (previously 0/1).
Description: Initializes a Gaussian univariate distribution with undefined mean and standard deviation until fitted; aligns serialization expectations by removing default numeric values.

Method: GaussianUnivariate._fit_params(self)
Location: copulas/univariate/gaussian.py → class GaussianUnivariate
Inputs: none (operates on instance state).
Outputs: dict containing either `{ 'mean': constant_value, 'std': 0 }` when a constant column is detected, otherwise `{ 'mean': self.mean, 'std': self.std }`.
Description: Provides fitting parameters for serialization; includes special handling for constant columns to embed the constant in the serialized dict instead of exposing `constant_value`.

Method: GaussianUnivariate.from_dict(cls, copula_dict)
Location: copulas/univariate/gaussian.py → class GaussianUnivariate
Inputs: a dictionary produced by `to_dict` (may lack `constant_value`).
Outputs: an instance with `fitted` flag restored; if `std` is 0 the instance’s `constant_value` is set to the stored `mean`, otherwise `mean` and `std` are restored.
Description: Deserializes a GaussianUnivariate, correctly reconstructing constant‑value distributions without requiring an explicit `constant_value` key.

Method: GaussianKDE.__init__(self, sample_size=None, *args, **kwargs)
Location: copulas/univariate/gaussian_kde.py → class GaussianKDE(ScipyWrapper)
Inputs: optional `sample_size` (int) controlling how many synthetic points are drawn when masking data; any additional args/kwargs forwarded to ScipyWrapper.
Outputs: a GaussianKDE instance with attribute `sample_size` set (default None).
Description: Enables privacy‑preserving fitting by optionally limiting the number of samples that the KDE model sees; the size is stored for later use in `_fit_params`.

Method: GaussianKDE.fit(self, X, *args, **kwargs)
Location: copulas/univariate/gaussian_kde.py → class GaussianKDE
Inputs: raw data array `X`; optional args/kwargs.
Outputs: Sets `self.constant_value` (via `_get_constant_value`), possibly masks the data with `self.sample(self.sample_size)` when a constant is not present and `sample_size` is given, fits the underlying SciPy KDE, assigns `self.sample` to the model’s `resample` method, sets `self.fitted=True`.
Description: Core fitting routine now supports constant‑value detection and optional data masking; after fitting, `instance.sample` points to the model’s sampling method, matching test expectations.

Method: GaussianKDE._fit_params(self)
Location: copulas/univariate/gaussian_kde.py → class GaussianKDE
Inputs: none.
Outputs: If `self.constant_value` is not None, returns `{'dataset': [constant] * self.sample_size}`; otherwise returns `{'dataset': self.model.dataset.tolist()}` (no `d`, `n`, `covariance`, `factor`, `inv_cov`).
Description: Serializes only the essential dataset (or repeated constant) to ensure identical dictionaries for all fitted instances, removing internal SciPy attributes and the now‑implicit `constant_value`.

Method: GaussianKDE.from_dict(cls, copula_dict)
Location: copulas/univariate/gaussian_kde.py → class GaussianKDE
Inputs: dictionary without `constant_value` (contains `fitted` and `dataset`).
Outputs: Instance with `fitted` restored; if the dataset contains a single unique value, sets `constant_value` to that value; otherwise recreates the SciPy KDE model from the dataset.
Description: Deserialization respects the new constant‑value encoding, allowing reconstruction of both constant and non‑constant KDEs without an explicit `constant_value` entry.

Method: KDEUnivariate.__init__(self, sample_size=None, *args, **kwargs)
Location: copulas/univariate/kde.py → class KDEUnivariate(Univariate)
Inputs: optional `sample_size` (int, default 10); additional args/kwargs forwarded to Univariate.
Outputs: KDEUnivariate instance with `self.sample_size` set.
Description: Adds a configurable sample size for privacy‑preserving KDE fitting; default matches test expectations (`instance.sample_size == 10`).

Method: KDEUnivariate.fit(self, X, *args, **kwargs)
Location: copulas/univariate/kde.py → class KDEUnivariate
Inputs: data array `X`.
Outputs: Detects constant columns via `_get_constant_value`; if no constant and `sample_size` is set, draws a fresh sample of size `sample_size` from a temporary KDE before fitting; otherwise fits directly; sets `self.fitted=True`.
Description: Mirrors GaussianKDE’s masking logic, ensuring that when `sample_size` is provided the fitted model does not retain original data, and that constant columns are handled via method replacement.

Method: KDEUnivariate._fit_params(self)
Location: copulas/univariate/kde.py → class KDEUnivariate
Inputs: none.
Outputs: If `self.constant_value` is not None, returns `{'dataset': [constant] * self.sample_size}`; otherwise returns `{'dataset': self.model.dataset.tolist()}` (no SciPy internal fields).
Description: Provides a stable, minimal serialization format that excludes `constant_value` and internal SciPy attributes, satisfying the “identical dict” requirement.

Method: KDEUnivariate.from_dict(cls, copula_dict)
Location: copulas/univariate/kde.py → class KDEUnivariate
Inputs: dict without `constant_value`.
Outputs: Instance with `fitted` flag restored; if the dataset is uniform, sets `constant_value`; otherwise creates a SciPy KDE from the dataset.
Description: Deserialization aligns with the new serialization format, automatically reconstructing constant‑value KDEs.

Method: TruncNorm.fit(self, X)
Location: copulas/univariate/truncnorm.py → class TruncNorm(ScipyWrapper)
Inputs: data array `X`.
Outputs: Detects constant columns; if constant, sets `self.constant_value` and replaces sampling/scoring methods via `_replace_constant_methods`; otherwise computes `min_`/`max_` bounds (with EPSILON) and calls `super().fit`. Sets `self.fitted=True`.
Description: Extends fitting to recognize constant distributions and hide the constant via method replacement; removes the previously exposed `constant_value` from serialized output.

Method: TruncNorm.from_dict(cls, parameters)
Location: copulas/univariate/truncnorm.py → class TruncNorm
Inputs: dict containing `'fitted'`, `'a'`, `'b'` (no `constant_value`).
Outputs: Instance with `fitted` restored; if `a == b` sets `constant_value` to that value; otherwise creates SciPy `truncnorm` model with parameters `a` and `b`.
Description: Deserialization now embeds constant information directly in the `a`/`b` parameters, eliminating the need for a separate `constant_value` field.

Method: TruncNorm._fit_params(self)
Location: copulas/univariate/truncnorm.py → class TruncNorm
Inputs: none.
Outputs: If `self.constant_value` is set, returns `{'a': constant, 'b': constant}`; otherwise returns the model’s `a` and `b` (plus any SciPy kwds).
Description: Serializes a TruncNorm distribution in a way that constant distributions are represented by identical `a` and `b` values, satisfying the test expectations that `constant_value` no longer appears.

Method: TruncNorm.from_dict (class method) updates to handle the new a/b constant representation as described above.
