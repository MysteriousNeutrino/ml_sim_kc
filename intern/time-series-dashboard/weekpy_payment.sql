SELECT
  DATE_TRUNC('week', date) :: DATE AS weeks,
  sum(amount) as sum_receipt
FROM
  new_payments
WHERE
  status = 'Confirmed'
group by
  weeks
order by
  weeks