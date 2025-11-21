# test_report_logger.py

class TestReportLogger:
    """A simple logger to collect detailed test results."""
    def __init__(self):
        self.results = []

    def log(self, test_case_id, parameters, hyperparameters, expected_result, obtained_result, is_optimal, status, time, outputs):
        self.results.append({
            "test_case_id": test_case_id,
            "parameters": parameters,
            "hyperparameters": hyperparameters,
            "expected_result": expected_result,
            "obtained_result": obtained_result,
            "is_optimal": is_optimal,
            "status": status,
            "time": time,
            "outputs": outputs
        })

    def get_results(self):
        return self.results

# Global instance of the logger
test_report_logger = TestReportLogger()
