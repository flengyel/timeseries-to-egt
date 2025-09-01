import sys; import os; sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import_and_version():
    import ts2eg
    assert hasattr(ts2eg, "__version__")
    # version follows PEP 440
    import re
    assert re.match(r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9\.\-]*)?$", ts2eg.__version__) is not None
