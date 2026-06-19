with station_hourly as (
    select
        s.station_no,
        i.district,
        date_format(s.record_time, '%Y-%m-%d %H:00:00') as record_hour,
        hour(s.record_time) as hour_of_day,
        case
            when hour(s.record_time) between 7 and 9
                or hour(s.record_time) between 17 and 19
                then 'peak'
            else 'off_peak'
        end as period_type,
        s.bikes_available,
        s.spaces_available,
        i.total_spaces,
        s.is_stock_out_risk,
        s.is_full_load_risk
    from {{ ref('stg_station_status') }} as s
    inner join {{ ref('stg_station_info') }} as i
        on s.station_no = i.station_no
)

select
    concat(district, '|', record_hour) as district_hour_key,
    district,
    record_hour,
    hour_of_day,
    period_type,
    count(distinct station_no) as stations_observed,
    count(*) as observations,
    avg(bikes_available) as avg_bikes_available,
    avg(spaces_available) as avg_spaces_available,
    avg(bikes_available / nullif(total_spaces, 0)) as avg_filling_rate,
    sum(is_stock_out_risk) as stock_out_risk_observations,
    sum(is_full_load_risk) as full_load_risk_observations
from station_hourly
group by
    district,
    record_hour,
    hour_of_day,
    period_type
