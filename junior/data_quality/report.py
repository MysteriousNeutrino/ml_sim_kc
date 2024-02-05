"""DQ Report."""

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import pandas as pd

from user_input.metrics import Metric
# from metrics import Metric
# from checklist import CHECKLIST

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def fit(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report."""
        self.report_ = {}
        report = self.report_
        report['title'] = ["table_name", "metric", "limits", "values", "status", "error"]
        report['result'] = {}
        report['passed'] = 0
        report['failed'] = 0
        report['errors'] = 0
        report['total'] = len(self.checklist)
        # Check if engine supported
        if self.engine != "pandas":
            raise NotImplementedError("Only pandas API currently supported!")
        for i, checklist in enumerate(self.checklist):
            # print(checklist[0])
            # print(checklist[1])
            # print(checklist[2])
            try:
                table = tables[checklist[0]]
                # print("table: ", table)
                metric_obj = checklist[1]
                # print("metric_obj: ", metric_obj)
                limits = checklist[2]
                # print("limits: ", limits, list(limits.keys()))
                if list(limits.keys()) == []:
                    # print("limits.keys() is None")
                    value = metric_obj(table)
                    status = '.'
                else:
                    value_equal = metric_obj(table).get(list(limits.keys())[0])
                    value = metric_obj(table)
                    # print("value: ", value_equal, list(limits.values())[0][0], list(limits.values())[0][1])
                    status = '.' if list(limits.values())[0][0] <= value_equal <= list(limits.values())[0][1] else 'F'
                if status == '.':
                    report['passed'] += 1
                elif status == 'F':
                    report['failed'] += 1
                # print("status: ", status)
                error = ''
                # print("error: ", error)
            except Exception as e:
                report['errors'] += 1
                status = 'E'
                error = str(e)

            report_line = [checklist[0], metric_obj, limits, value, status, error]
            # print("report_line: ", report_line)

            report['result'][i] = report_line


        report['passed_pct'] = report['passed'] / report['total']
        report['failed_pct'] = report['failed'] / report['total']
        report['errors_pct'] = report['errors'] / report['total']

        return report

    def to_str(self) -> str:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )


# report = Report(CHECKLIST, "pandas")
# daily_sales_df = pd.read_csv(
#     r"C:\Users\Neesty\PycharmProjects\ml_sim_kc\junior\data_quality\ke_daily_sales.csv"
# )
# report.fit({'sales': daily_sales_df})
# print(report.to_str())