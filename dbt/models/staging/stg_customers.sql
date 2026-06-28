-- One row per order-scoped customer_id, mapping to the stable customer_unique_id.
select
    customer_id,
    customer_unique_id,
    customer_zip_code_prefix,
    customer_city,
    customer_state
from {{ source('olist', 'olist_customers_dataset') }}
where customer_id is not null
