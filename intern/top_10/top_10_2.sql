SELECT
sku_type::float,
count(distinct vendor) count_vendor
FROM
  sku_dict_another_one
GROUP BY sku_type
ORDER BY count_vendor desc
LIMIT 10