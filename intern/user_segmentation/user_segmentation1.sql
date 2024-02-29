


SELECT
  CASE
    WHEN purchase_range = 1 THEN '0-20000'
    WHEN purchase_range = 2 THEN '20000-40000'
    WHEN purchase_range = 3 THEN '40000-60000'
    WHEN purchase_range = 4 THEN '60000-80000'
    WHEN purchase_range = 5 THEN '80000-100000'
    WHEN purchase_range = 6 THEN CONCAT(
          '100000-',
          (
            select
              max(amount)
            from
              (SELECT
  sum(amount) as amount
FROM
  new_payments
WHERE
  status = 'Confirmed'
  AND mode IN ('MasterCard', 'МИР', 'Visa')
GROUP BY
  email_id
  ORDER BY amount) as q3
          )::int)
    
  END AS purchase_range,
  num_of_users

FROM
  (
    SELECT
      COUNT(*) AS num_of_users,
      CASE
        WHEN amount BETWEEN 0
        AND 20000 THEN 1 -- '0-20000'
        WHEN amount BETWEEN 20001
        AND 40000 THEN 2 -- '20000-40000'
        WHEN amount BETWEEN 40001
        AND 60000 THEN 3 -- '40000-60000'
        WHEN amount BETWEEN 60001
        AND 80000 THEN 4 -- '60000-80000'
        WHEN amount BETWEEN 80001
        AND 100000 THEN 5 -- '80000-100000'
        WHEN amount BETWEEN 100001
        AND 300000 THEN 6 -- CONCAT('100000-', MAX(amount) OVER ())
      END AS purchase_range
    FROM
      (SELECT
  sum(amount) as amount
FROM
  new_payments
WHERE
  status = 'Confirmed'
  AND mode IN ('MasterCard', 'МИР', 'Visa')
GROUP BY
  email_id) as q2
    GROUP BY
      purchase_range
    order by
      purchase_range
  ) AS q1