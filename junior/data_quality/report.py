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
        report['title'] = f"DQ Report for tables {sorted(list(set(tables.keys())))}"
        report['result'] = {}
        report['passed'] = 0
        report['failed'] = 0
        report['errors'] = 0
        report['total'] = len(self.checklist)
        # Check if engine supported
        if self.engine != "pandas":
            raise NotImplementedError("Only pandas API currently supported!")
        for i, checklist in enumerate(self.checklist):
            try:
                table = tables[checklist[0]]
                metric_obj = checklist[1]
                limits = checklist[2]
                value_metric_obj = metric_obj(table)
                if list(limits.keys()) == []:
                    value = value_metric_obj
                    status = '.'
                else:
                    value_equal = value_metric_obj.get(list(limits.keys())[0])
                    value = value_metric_obj
                    status = '.' if list(limits.values())[0][0] <= value_equal <= list(limits.values())[0][1] else 'F'
                if status == '.':
                    report['passed'] += 1
                elif status == 'F':
                    report['failed'] += 1
                error = ''
            except Exception as e:
                error = e
                report['errors'] += 1
                status = 'E'

            report_line = [checklist[0], str(metric_obj), str(limits), value, status, error]

            report['result'][i] = report_line
        report['passed_pct'] = report['passed'] / report['total'] * 100
        report['failed_pct'] = report['failed'] / report['total'] * 100
        report['errors_pct'] = report['errors'] / report['total'] * 100

        report['result'] = pd.DataFrame.from_dict(report['result'], orient='index',
                                           columns=['table_name', 'metric', 'limits', 'values', 'status', 'error'])


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


#
# report = Report(CHECKLIST, "pandas")
# daily_sales_df = pd.read_csv(
#     r"C:\Users\Neesty\PycharmProjects\ml_sim_kc\junior\data_quality\ke_daily_sales.csv"
# )
# ke_visits_df = pd.read_csv(
#     r"C:\Users\Neesty\PycharmProjects\ml_sim_kc\junior\data_quality\ke_visits.csv"
# )
# report.fit({'sales': daily_sales_df, 'relevance': ke_visits_df})
# print(report.to_str())
