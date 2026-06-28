-- Singular test: customer_metrics must satisfy basic invariants.
-- dbt fails the build if this query returns any rows.
select
    customer_unique_id,
    n_orders,
    n_orders_pre,
    n_orders_post,
    total_value
from {{ ref('customer_metrics') }}
where n_orders < 1
   or total_value < 0
   or n_orders_pre < 0
   or n_orders_post < 0
   or (n_orders_pre + n_orders_post) <> n_orders
