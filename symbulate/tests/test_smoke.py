def test_symbulate_imports():
    """Confirm Symbulate can be imported and core objects are accessible."""
    import symbulate

    assert hasattr(symbulate, "RV")
    assert hasattr(symbulate, "BoxModel")
