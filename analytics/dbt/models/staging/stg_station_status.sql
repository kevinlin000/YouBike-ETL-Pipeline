select
    id,
    station_no,
    cast(bikes_available as signed) as bikes_available,
    cast(spaces_available as signed) as spaces_available,
    cast(record_time as datetime) as record_time,
    bikes_available + spaces_available as observed_capacity,
    case
        when bikes_available <= 2 then 1
        else 0
    end as is_stock_out_risk,
    case
        when spaces_available <= 2 then 1
        else 0
    end as is_full_load_risk
from {{ source('youbike_raw', 'station_status') }}
