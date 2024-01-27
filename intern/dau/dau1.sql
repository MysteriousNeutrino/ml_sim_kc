select
toDate(toDateTime(timestamp)) day,
countDistinct(user_id) dau
from default.churn_submits
group by day