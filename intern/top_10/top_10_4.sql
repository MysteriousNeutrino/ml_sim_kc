SELECT
vendor,
count( sku_type) sku
FROM
  sku_dict_another_one
GROUP BY vendor
ORDER BY sku desc
LIMIT 10