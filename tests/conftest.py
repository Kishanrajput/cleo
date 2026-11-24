"""Pytest configuration and shared fixtures."""




import pytest


def pytest_configure(config):
    """Register markers declared in pytest.ini so -m filters work everywhere."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Seed anything that needs determinism across the whole test session."""
    import random

    random.seed(42)
    yield
