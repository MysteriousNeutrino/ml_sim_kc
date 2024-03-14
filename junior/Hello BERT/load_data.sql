SELECT
  Id as review_id,
  toDateTime(Time) as dt,
  Score as rating,
  if(Score = 1, 'negative', if(Score = 5, 'positive', 'neutral')) AS sentiment,
  Text as review
FROM
  simulator.flyingfood_reviews

ORDER BY review_id