import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import_and_version():
    import ts2eg
    assert hasattr(ts2eg, "__version__")
    # version follows PEP 440
    import re
    assert re.match(r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9\.\-]*)?$", ts2eg.__version__) is not None


def test_documented_payoff_helpers_are_public_top_level_imports():
    import ts2eg
    from ts2eg import (
        growth_payoffs,
        info_gain_payoffs,
        project_out_identities,
        value_gradient_payoffs,
    )

    helpers = {
        "growth_payoffs": growth_payoffs,
        "info_gain_payoffs": info_gain_payoffs,
        "project_out_identities": project_out_identities,
        "value_gradient_payoffs": value_gradient_payoffs,
    }
    for name, helper in helpers.items():
        assert callable(helper)
        assert getattr(ts2eg, name) is helper
        assert name in ts2eg.__all__
