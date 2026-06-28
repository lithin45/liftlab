-- Reviews with numeric score. Not every order has a review.
select
    review_id,
    order_id,
    try_cast(review_score as integer) as review_score,
    try_cast(review_creation_date as timestamp) as review_creation_date,
    try_cast(review_answer_timestamp as timestamp) as review_answer_timestamp
from {{ source('olist', 'olist_order_reviews_dataset') }}
where order_id is not null
  and try_cast(review_score as integer) is not null
