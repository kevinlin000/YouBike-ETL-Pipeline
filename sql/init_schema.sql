CREATE TABLE IF NOT EXISTS station_info (
  station_no      VARCHAR(16)  NOT NULL,
  name_tw         VARCHAR(255) NOT NULL,
  district        VARCHAR(255) NOT NULL,
  lat             DECIMAL(10,7) NOT NULL,
  lng             DECIMAL(10,7) NOT NULL,
  total_spaces    INT NOT NULL,
  PRIMARY KEY (station_no)
);

CREATE TABLE IF NOT EXISTS station_status (
  id               BIGINT AUTO_INCREMENT PRIMARY KEY,
  station_no       VARCHAR(16) NOT NULL,
  bikes_available  INT NOT NULL,
  spaces_available INT NOT NULL,
  record_time      DATETIME NOT NULL,
  UNIQUE KEY uniq_station_time (station_no, record_time),
  CONSTRAINT fk_station
    FOREIGN KEY (station_no) REFERENCES station_info(station_no)
);

