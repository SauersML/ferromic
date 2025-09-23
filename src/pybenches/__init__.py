"""Test benchmarks package for Pytest discovery.

Adding this module allows benchmark tests to use package-relative imports
when executed by pytest, which treats the directory as a package once an
``__init__`` module is present.
"""

