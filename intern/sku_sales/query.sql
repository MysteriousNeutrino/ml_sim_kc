select
cal_date days,
sum(cnt) sku
    from transactions_another_one
group by cal_date