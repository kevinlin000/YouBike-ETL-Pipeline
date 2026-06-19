select
    concat(i.district, '|', date_format(s.record_time, '%Y-%m-%d %H:00:00')) as district_hour_key,
    i.district,
    date_format(s.record_time, '%Y-%m-%d %H:00:00') as record_hour,
    hour(s.record_time) as hour_of_day,
    case
        when hour(s.record_time) between 7 and 9
            or hour(s.record_time) between 17 and 19
            then 'peak'
        else 'off_peak'
    end as period_type,
    count(distinct s.station_no) as stations_observed,
    count(*) as observations,
    avg(s.bikes_available) as avg_bikes_available,
    avg(s.spaces_available) as avg_spaces_available,
    avg(s.bikes_available / nullif(i.total_spaces, 0)) as avg_filling_rate,
    sum(s.is_stock_out_risk) as stock_out_risk_observations,
    sum(s.is_full_load_risk) as full_load_risk_observations
from {{ ref('stg_station_status') }} as s
inner join {{ ref('stg_station_info') }} as i
    on s.station_no = i.station_no
group by
    i.district,
    date_format(s.record_time, '%Y-%m-%d %H:00:00'),
    hour(s.record_time),
    case
        when hour(s.record_time) between 7 and 9
            or hour(s.record_time) between 17 and 19
            then 'peak'
        else 'off_peak'
    end
