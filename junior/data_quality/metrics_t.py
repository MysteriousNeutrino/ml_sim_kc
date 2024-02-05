import pandas as pd

from metrics import Metric, CountTotal, CountZeros, CountNull

daily_sales_df = pd.read_csv(
    r"C:\Users\Neesty\PycharmProjects\ml_sim_kc\junior\data_quality\ke_daily_sales.csv"
)

count_total = CountTotal()
count_zeros = CountZeros("qty")
count_null = CountNull(["day", "price"], "any")

print(count_total(daily_sales_df))
print(count_zeros(daily_sales_df))
print(count_null(daily_sales_df))
