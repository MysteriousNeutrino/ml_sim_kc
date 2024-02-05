SELECT
  DATE_TRUNC('month', date)::DATE as time,
  mode,
  (COUNT(id) FILTER (WHERE status = 'Confirmed'))::float / count(id) * 100   as percents
FROM
  new_payments
 WHERE mode != 'Не определено'
GROUP BY
  time, mode
ORDER BY time, mode