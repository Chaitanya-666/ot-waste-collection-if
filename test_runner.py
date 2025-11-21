# Author: Chaitanya Shinde (231070066)
#
# This file contains the custom test runner and result classes. These are
# designed to capture detailed metrics for each test case, including execution
# time and status, which are then used to generate a comprehensive test summary
# report in Markdown format.
# test_runner.py

import time
import unittest

class CustomTestResult(unittest.TestResult):
    """A custom test result class that captures detailed information."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_results = []

    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.time() - self.start_time
        self.test_results.append({
            "name": test.id(),
            "description": test.shortDescription() or "",
            "status": "✅ Pass",
            "time": f"{elapsed:.4f}s",
        })

    def addError(self, test, err):
        super().addError(test, err)
        elapsed = time.time() - self.start_time
        self.test_results.append({
            "name": test.id(),
            "description": test.shortDescription() or "",
            "status": f"❌ Error: {err[0].__name__}",
            "time": f"{elapsed:.4f}s",
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = time.time() - self.start_time
        self.test_results.append({
            "name": test.id(),
            "description": test.shortDescription() or "",
            "status": f"FAIL: {err[0].__name__}",
            "time": f"{elapsed:.4f}s",
        })

class CustomTestRunner:
    """A custom test runner that uses the CustomTestResult."""
    def run(self, suite):
        result = CustomTestResult()
        suite.run(result)
        return result
