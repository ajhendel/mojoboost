"""scikit-learn conventions, implemented without depending on sklearn.

The estimators follow the scikit-learn estimator contract closely enough
for `clone`, `Pipeline`, `GridSearchCV`, and `cross_val_score` to work, but
scikit-learn stays an optional dependency: everything here degrades to
plain Python when it is not installed. `check_estimator`'s full suite has
not been run, so mojotrees does not claim to be a compliant estimator.

Two pieces need scikit-learn when it is present:

- `NotFittedError` subclasses scikit-learn's when it can, so `except
  sklearn.exceptions.NotFittedError` catches it. It also subclasses
  `RuntimeError`, which is what mojotrees raised before, so existing
  callers keep working.
- `estimator_tags` builds the `Tags` object scikit-learn 1.6 and newer
  require from `__sklearn_tags__`. It borrows the defaults from
  scikit-learn's own mixins rather than constructing `Tags` field by
  field, so a new or renamed tag field does not break us.
"""

import inspect

try:  # pragma: no cover - depends on the install
    from sklearn.exceptions import NotFittedError as _SklearnNotFittedError
except ImportError:  # pragma: no cover - depends on the install

    class _SklearnNotFittedError(ValueError, AttributeError):
        """Stand-in with scikit-learn's base classes, so `except ValueError`
        and `except AttributeError` behave the same either way."""


class NotFittedError(_SklearnNotFittedError, RuntimeError):
    """Raised when an estimator method needs a model that is not there yet.

    Also a `RuntimeError`, which is what mojotrees raised before it grew a
    named exception.
    """


def estimator_tags(kind):
    """scikit-learn `Tags` for a mojotrees estimator, `kind` being
    "classifier" or "regressor". Requires scikit-learn: only scikit-learn
    itself calls `__sklearn_tags__`."""
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

    mixin = ClassifierMixin if kind == "classifier" else RegressorMixin
    proxy = type("_MojoTreesTagProxy", (mixin, BaseEstimator), {})()
    tags = proxy.__sklearn_tags__()
    # NaN is the missing-value marker (src/mojotrees/binning.mojo); sparse
    # input is rejected; training is deterministic given the same inputs.
    tags.input_tags.allow_nan = True
    tags.input_tags.sparse = False
    tags.non_deterministic = False
    return tags


class ParamsMixin:
    """`get_params`/`set_params` with scikit-learn's semantics.

    Parameter names and defaults are read from the `__init__` of every
    class in the MRO, most derived first, so a hyperparameter added to a
    shared base is picked up with no list maintained by hand. Each
    `__init__` must store its arguments unmodified under their own names,
    which is what makes `clone` reconstruct an equivalent estimator.

    scikit-learn asks that every parameter be an explicit keyword argument
    of the estimator's own `__init__`, and mojotrees's subclasses forward
    the shared ones through `**kwargs` instead. `get_params`, `set_params`,
    `clone`, `Pipeline`, and `GridSearchCV` all work either way, but
    `inspect.signature(MojoTreesRegressor)` does not list the inherited
    parameters, and this is one of the reasons mojotrees does not claim to
    pass `check_estimator`.
    """

    @classmethod
    def _init_params(cls):
        """{name: default} across the MRO, most derived winning."""
        found = {}
        for klass in reversed(cls.__mro__):
            # object.__init__ is an implementation detail with a generic
            # *args/**kwargs signature, not part of the estimator contract.
            if klass is object:
                continue
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for p in inspect.signature(init).parameters.values():
                if p.name == "self" or p.kind == p.VAR_KEYWORD:
                    continue
                if p.kind == p.VAR_POSITIONAL:
                    raise RuntimeError(
                        f"{klass.__name__} uses *args in __init__, which a "
                        "scikit-learn style estimator may not do"
                    )
                found[p.name] = p.default
        return found

    @classmethod
    def _get_param_names(cls):
        return sorted(cls._init_params())

    def get_params(self, deep=True):
        """Constructor parameters as a dict. `deep` recurses into any
        parameter value that is itself an estimator; mojotrees has none
        today, so it makes no difference."""
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(
                value, type
            ):
                for sub, sub_value in value.get_params().items():
                    out[f"{key}__{sub}"] = sub_value
            out[key] = value
        return out

    def set_params(self, **params):
        """Set constructor parameters in place and return self."""
        if not params:
            return self
        valid = self._get_param_names()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"invalid parameter {key!r} for estimator "
                    f"{type(self).__name__}; valid parameters are "
                    + ", ".join(valid)
                )
            setattr(self, key, value)
        return self

    def __repr__(self):
        defaults = self._init_params()
        parts = []
        for name in sorted(defaults):
            value = getattr(self, name, None)
            default = defaults[name]
            try:
                same = bool(value == default)
            except Exception:  # pragma: no cover - exotic parameter values
                same = False
            if not same:
                parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"
