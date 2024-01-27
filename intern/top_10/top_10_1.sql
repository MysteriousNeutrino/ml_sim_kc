SELECT
  brand,
  count(sku_type::varchar) count_sku
FROM
  sku_dict_another_one
WHERE brand IS NOT NULL
GROUP BY brand
ORDER BY  count_sku desc
LIMIT 10