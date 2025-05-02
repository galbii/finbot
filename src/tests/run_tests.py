#!/usr/bin/env python3
"""
Test runner for the trading strategy system.

Run this script to execute all unit tests.
"""

import unittest
import os
import sys

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def run_tests():
    """Run all test modules in the tests directory."""
    # Create test suite
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    successful = run_tests()
    sys.exit(0 if successful else 1) 