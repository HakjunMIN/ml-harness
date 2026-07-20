def test_package_exports_version():
    import power_forecasting
    assert power_forecasting.__version__ == "0.1.0"
