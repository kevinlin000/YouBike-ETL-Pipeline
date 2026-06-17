select
    station_no,
    name_tw,
    district,
    cast(lat as decimal(10, 7)) as lat,
    cast(lng as decimal(10, 7)) as lng,
    cast(total_spaces as signed) as total_spaces
from {{ source('youbike_raw', 'station_info') }}
