SELECT
  sku,
  dates,
  avg(price) as price,
  count(dates) as qnt
FROM
  transactions
GROUP BY
  dates,
  sku
 ORDER BY sku, dates