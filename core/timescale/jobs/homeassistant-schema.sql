-- Home Assistant (LTSS) reporting objects on the timescale cluster.
-- Applied by homeassistant-schema-job.yaml as the homeassistant role (owner
-- of the database and of LTSS' ltss table). Idempotent; safe to rerun.
--
-- LTSS creates ltss itself on first start, so everything that depends on it
-- is skipped until the table exists.
\set ON_ERROR_STOP on

-- LTSS commits every state change separately; waiting for Longhorn's 3-way
-- fsync capped it at ~3 rows/s. Losing the last ~second of readings on a
-- crash is fine for sensor history (no corruption risk).
ALTER DATABASE homeassistant SET synchronous_commit = off;

SELECT to_regclass('public.ltss') IS NOT NULL AS has_ltss \gset
\if :has_ltss

-- Compression: sensor rows segmented by entity compress ~10-20x. Settings
-- can't be changed once chunks are compressed, so only set them once.
SELECT NOT compression_enabled AS need_compress
  FROM timescaledb_information.hypertables WHERE hypertable_name = 'ltss' \gset
\if :need_compress
ALTER TABLE ltss SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'entity_id',
  timescaledb.compress_orderby = 'time DESC'
);
\endif
SELECT add_compression_policy('ltss', compress_after => INTERVAL '7 days', if_not_exists => true);
-- Daily chunks so data compresses at ~7-8 days old rather than 7-14.
SELECT set_chunk_time_interval('ltss', INTERVAL '1 day');

-- Latest metadata + state per entity (LTSS only stores changes, and the
-- attributes JSON is too slow to DISTINCT ON over for every panel).
CREATE TABLE IF NOT EXISTS ha_entity (
  entity_id    text PRIMARY KEY,
  domain       text NOT NULL,
  name         text NOT NULL,
  unit         text,
  device_class text,
  state_class  text,
  state        text,
  last_seen    timestamptz NOT NULL
);

CREATE OR REPLACE PROCEDURE ha_refresh_entities(job_id int DEFAULT NULL, config jsonb DEFAULT NULL)
LANGUAGE sql AS $$
  INSERT INTO ha_entity AS e
  SELECT DISTINCT ON (entity_id)
         entity_id,
         split_part(entity_id, '.', 1),
         coalesce(attributes->>'friendly_name', entity_id),
         attributes->>'unit_of_measurement',
         attributes->>'device_class',
         attributes->>'state_class',
         state,
         time
    FROM ltss
   WHERE time > now() - coalesce((config->>'lookback')::interval, INTERVAL '15 minutes')
   ORDER BY entity_id, time DESC
  ON CONFLICT (entity_id) DO UPDATE
     SET domain = excluded.domain, name = excluded.name, unit = excluded.unit,
         device_class = excluded.device_class, state_class = excluded.state_class,
         state = excluded.state, last_seen = excluded.last_seen
   WHERE excluded.last_seen >= e.last_seen;
$$;

SELECT NOT EXISTS (SELECT 1 FROM ha_entity) AS entity_empty \gset
\if :entity_empty
CALL ha_refresh_entities(NULL, '{"lookback": "100 years"}');
\endif
SELECT NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs
                    WHERE proc_name = 'ha_refresh_entities') AS need_entity_job \gset
\if :need_entity_job
SELECT add_job('ha_refresh_entities', INTERVAL '1 minute', config => '{"lookback": "15 minutes"}');
\endif

-- A state as a number, or NULL for "on", "unavailable", etc.
CREATE OR REPLACE FUNCTION ha_num(s text) RETURNS double precision
LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE WHEN s ~ '^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$' THEN s::double precision END
$$;

-- ha_entity plus a short display label for dashboard legends: common
-- suffixes stripped ("Dining room Temperature" -> "Dining room"), and made
-- unique within a device class, since Grafana merges series that share a
-- name.
CREATE OR REPLACE VIEW ha_entities AS
SELECT e.*,
       CASE WHEN count(*) OVER (PARTITION BY e.short, e.device_class) > 1
            THEN e.short || ' (' || split_part(e.entity_id, '.', 2) || ')'
            ELSE e.short END AS label
  FROM (SELECT ha_entity.*,
               coalesce(nullif(trim(regexp_replace(regexp_replace(name,
                 '^Nest Temperature Sensor \((.*?)\s*\)', '\1 (Nest)'),
                 '\s*(Temp(erature)? Sensor\s*)?(Temperature|Humidity|Power Minute Average|Energy Today|Battery( level)?|Instantaneous demand|Power|Device temperature)$',
                 '', 'i')), ''), name) AS short
          FROM ha_entity) e;

-- Hourly rollup of numeric states, for long time ranges and daily energy.
-- Real-time (materialized_only = false) so the current hour is included.
SELECT to_regclass('public.ltss_hourly') IS NULL AS new_hourly \gset
CREATE MATERIALIZED VIEW IF NOT EXISTS ltss_hourly
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket(INTERVAL '1 hour', time) AS bucket,
       entity_id,
       avg(state::double precision)         AS avg,
       min(state::double precision)         AS min,
       max(state::double precision)         AS max,
       first(state::double precision, time) AS first,
       last(state::double precision, time)  AS last,
       count(*)                             AS samples
  FROM ltss
 WHERE state ~ '^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$'
 GROUP BY bucket, entity_id
WITH NO DATA;
SELECT add_continuous_aggregate_policy('ltss_hourly',
  start_offset => INTERVAL '3 days', end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '30 minutes', if_not_exists => true);
\if :new_hourly
CALL refresh_continuous_aggregate('ltss_hourly', NULL, now() - INTERVAL '1 hour');
\endif

-- Numeric series for Grafana: averaged per step, from raw rows for short
-- steps and from ltss_hourly once a step is 30m or more. Each entity also
-- gets its last value before the range at t_from, so lines start at the
-- left edge even for sensors that rarely change.
CREATE OR REPLACE FUNCTION ha_series(ids text[], t_from timestamptz, t_to timestamptz, step interval)
RETURNS TABLE (ts timestamptz, entity_id text, value double precision)
LANGUAGE plpgsql STABLE AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT t_from, i.id, p.v
    FROM unnest(ids) AS i(id)
    CROSS JOIN LATERAL (
      SELECT l.state::double precision AS v
        FROM ltss l
       WHERE l.entity_id = i.id AND l.time < t_from AND l.time > t_from - INTERVAL '30 days'
         AND l.state ~ '^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$'
       ORDER BY l.time DESC LIMIT 1) p;

  IF step >= INTERVAL '30 minutes' THEN
    RETURN QUERY
    SELECT time_bucket(greatest(step, INTERVAL '1 hour'), h.bucket), h.entity_id::text,
           sum(h.avg * h.samples) / sum(h.samples)
      FROM ltss_hourly h
     WHERE h.entity_id = ANY (ids) AND h.bucket >= t_from AND h.bucket < t_to
     GROUP BY 1, 2;
  ELSE
    RETURN QUERY
    SELECT time_bucket(greatest(step, INTERVAL '1 second'), l.time), l.entity_id::text,
           avg(l.state::double precision)
      FROM ltss l
     WHERE l.entity_id = ANY (ids) AND l.time >= t_from AND l.time < t_to
       AND l.state ~ '^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$'
     GROUP BY 1, 2;
  END IF;
END $$;

-- Raw state changes for state timelines (doors, motion, alarms), with the
-- state in force at t_from and a closing row at t_to so bars span the range.
CREATE OR REPLACE FUNCTION ha_states(ids text[], t_from timestamptz, t_to timestamptz)
RETURNS TABLE (ts timestamptz, entity_id text, state text)
LANGUAGE sql STABLE AS $$
  WITH s AS (
    SELECT t_from AS time, i.id AS entity_id, p.state
      FROM unnest(ids) AS i(id)
      CROSS JOIN LATERAL (
        SELECT l.state::text FROM ltss l
         WHERE l.entity_id = i.id AND l.time < t_from
         ORDER BY l.time DESC LIMIT 1) p
    UNION ALL
    SELECT l.time, l.entity_id::text, l.state::text
      FROM ltss l
     WHERE l.entity_id = ANY (ids) AND l.time >= t_from AND l.time < t_to
  )
  SELECT * FROM s
  UNION ALL
  (SELECT DISTINCT ON (s.entity_id) least(t_to, now()), s.entity_id, s.state
     FROM s ORDER BY s.entity_id, s.time DESC);
$$;

-- Consumption per local day from cumulative counters (kWh "today"/"this
-- month"/lifetime totals). Sums positive hour-to-hour increases; a drop is
-- treated as a counter reset, counting the new value.
CREATE OR REPLACE FUNCTION ha_daily_delta(ids text[], t_from timestamptz, t_to timestamptz,
                                          tz text DEFAULT 'America/Chicago')
RETURNS TABLE (day timestamptz, entity_id text, delta double precision)
LANGUAGE sql STABLE AS $$
  WITH h AS (
    SELECT h.bucket, h.entity_id, h.first, h.last,
           lag(h.last) OVER (PARTITION BY h.entity_id ORDER BY h.bucket) AS prev
      FROM ltss_hourly h
     WHERE h.entity_id = ANY (ids)
       AND h.bucket >= t_from - INTERVAL '1 day' AND h.bucket < t_to
  )
  SELECT date_trunc('day', h.bucket AT TIME ZONE tz) AT TIME ZONE tz, h.entity_id,
         sum(CASE WHEN h.prev IS NULL THEN h.last - h.first
                  WHEN h.last >= h.prev THEN h.last - h.prev
                  ELSE h.last END)
    FROM h
   WHERE h.bucket >= t_from
   GROUP BY 1, 2;
$$;

-- Read-only access for Grafana (role managed in cluster.yaml; skipped until
-- CNPG has created it).
SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana') AS has_grafana \gset
\if :has_grafana
GRANT CONNECT ON DATABASE homeassistant TO grafana;
GRANT USAGE ON SCHEMA public TO grafana;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO grafana;
\endif

\else
\echo 'ltss does not exist yet (LTSS has not started); skipping'
\endif
