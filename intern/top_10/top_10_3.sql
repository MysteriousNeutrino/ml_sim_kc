SELECT
vendor,
count( distinct brand) brand
FROM
  sku_dict_another_one
GROUP BY vendor
ORDER BY brand desc
LIMIT 10