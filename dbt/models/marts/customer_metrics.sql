-- ============================================================================
-- customer_metrics, the Phase 1 "experiment fact table" smoke test.
--
-- Grain: one row per customer_unique_id. Splits each customer's history at
-- var('cutoff_date') into a PRE period (the CUPED covariate substrate) and a
-- POST period (the outcome substrate). This is the realistic population layer;
-- the synthetic randomized assignment + injected treatment effect are overlaid
-- on top of it in Phase 2 (liftlab.simulation).
-- ============================================================================

with order_value as (
    -- Collapse line items to one value per order (avoids join fan-out).
    select
        order_id,
        sum(coalesce(price, 0) + coalesce(freight_value, 0)) as order_value
    from {{ ref('stg_order_items') }}
    group by order_id
),

order_review as (
    -- Average review score per order (orders may have 0..n reviews).
    select
        order_id,
        avg(review_score) as review_score
    from {{ ref('stg_order_reviews') }}
    group by order_id
),

orders_enriched as (
    select
        c.customer_unique_id,
        c.customer_state,
        o.order_id,
        o.order_purchase_timestamp,
        coalesce(ov.order_value, 0) as order_value,
        r.review_score,
        case
            when o.order_purchase_timestamp < cast('{{ var("cutoff_date") }}' as timestamp)
            then 1 else 0
        end as is_pre_period
    from {{ ref('stg_orders') }} o
    join {{ ref('stg_customers') }} c using (customer_id)
    left join order_value ov using (order_id)
    left join order_review r using (order_id)
)

select
    customer_unique_id,
    -- Deterministic pick: a customer_unique_id can map to many order-scoped
    -- customer_id rows in DIFFERENT states on real Olist; take the most recent.
    max_by(customer_state, order_purchase_timestamp) as customer_state,
    count(*) as n_orders,
    sum(is_pre_period) as n_orders_pre,
    sum(1 - is_pre_period) as n_orders_post,
    round(sum(order_value), 2) as total_value,
    round(sum(case when is_pre_period = 1 then order_value else 0 end), 2) as pre_period_value,
    round(sum(case when is_pre_period = 0 then order_value else 0 end), 2) as post_period_value,
    round(avg(order_value), 2) as avg_order_value,
    avg(review_score) as avg_review_score,
    min(order_purchase_timestamp) as first_order_at,
    max(order_purchase_timestamp) as last_order_at,
    -- A simple binary outcome substrate: did the customer transact post-cutoff?
    case when sum(1 - is_pre_period) > 0 then 1 else 0 end as post_converted
from orders_enriched
group by customer_unique_id
