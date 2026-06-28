-- Payments with numeric value. Real Olist can have multiple payments per order.
select
    order_id,
    try_cast(payment_sequential as integer) as payment_sequential,
    payment_type,
    try_cast(payment_installments as integer) as payment_installments,
    try_cast(payment_value as double) as payment_value
from {{ source('olist', 'olist_order_payments_dataset') }}
where order_id is not null
