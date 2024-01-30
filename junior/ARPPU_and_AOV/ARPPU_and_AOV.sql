SELECT
  DATE_TRUNC('month', date)::DATE as time,
  avg(amount) as AOV,
  sum(amount) / count(distinct email_id) as ARPPU
FROM
  new_payments
WHERE
  status = 'Confirmed'
GROUP BY
  time
ORDER BY
  time