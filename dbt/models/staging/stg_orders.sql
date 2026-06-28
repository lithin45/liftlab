-- Orders with typed timestamps. try_cast tolerates the empty delivery dates that
-- appear in the real Olist data for non-delivered orders.
select
    order_id,
    customer_id,
    order_status,
    try_cast(order_purchase_timestamp as timestamp) as order_purchase_timestamp,
    try_cast(order_approved_at as timestamp) as order_approved_at,
    try_cast(order_delivered_carrier_date as timestamp) as order_delivered_carrier_date,
    try_cast(order_delivered_customer_date as timestamp) as order_delivered_customer_date,
    try_cast(order_estimated_delivery_date as timestamp) as order_estimated_delivery_date
from {{ source('olist', 'olist_orders_dataset') }}
where order_id is not null
  and try_cast(order_purchase_timestamp as timestamp) is not null
