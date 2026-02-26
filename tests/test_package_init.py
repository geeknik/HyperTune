import pytest


def test_lazy_exports_load_symbols():
    import hypertune

    assert hypertune.HyperTune is not None
    assert hypertune.Database is not None
    assert hypertune.HyperPredictor is not None


def test_unknown_lazy_export_raises():
    import hypertune

    with pytest.raises(AttributeError):
        _ = hypertune.NotARealExport
