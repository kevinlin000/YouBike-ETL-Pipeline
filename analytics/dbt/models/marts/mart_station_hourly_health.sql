select
    s.station_no,
    i.name_tw,
    i.district,
    date_format(s.record_time, '%Y-%m-%d %H:00:00') as record_hour,
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
    s.station_no,
    i.name_tw,
    i.district,
    date_format(s.record_time, '%Y-%m-%d %H:00:00')
