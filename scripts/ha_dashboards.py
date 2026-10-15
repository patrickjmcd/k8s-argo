#!/usr/bin/env python3
"""Generate the Home Assistant Grafana dashboards.

Writes core/grafana-assets/dashboards/ha-*.json. The data is LTSS' ltss table
on the TimescaleDB cluster, read through the helpers in
core/timescale/homeassistant-schema.sql (ha_entities, ha_series, ha_states,
ha_daily_delta, ltss_hourly). Edit this file and rerun it rather than editing
the JSON:

    python3 scripts/ha_dashboards.py
"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "core/grafana-assets/dashboards"
DS = {"type": "postgres", "uid": "ha-timescale"}
TAGS = ["home-assistant"]

# ---------------------------------------------------------------- entity sets
# SQL predicates over ha_entities (alias e). Kept here so every dashboard
# agrees on what counts as a room, a circuit, etc.
INDOOR_TEMP = (
    "e.domain = 'sensor' AND e.device_class = 'temperature' AND e.entity_id !~ "
    "'(device_temperature|printer|mcdata|fordpass|freezer|fridge|refrigerator|lumi_lumi"
    "|target|apparent|cloud_aqi|outdoor|^sensor\\.home_|filament|dehumidifier)'"
)
INDOOR_HUMIDITY = (
    "e.domain = 'sensor' AND e.device_class = 'humidity' AND e.entity_id !~ "
    "'(printer|freezer|fridge|lumi_lumi|cloud_aqi|outdoor|^sensor\\.home_|filament|humidifier_humidity)'"
)
OUTDOOR_TEMP = ["sensor.outdoor_temperature", "sensor.home_apparent_temperature"]
COLD_STORAGE = [
    "sensor.basement_fridge_refrigerator_temperature",
    "sensor.basement_freezer_temperature",
    "sensor.lumi_lumi_weather_temperature",
]
EQUIPMENT_TEMP = (
    "e.domain = 'sensor' AND e.device_class = 'temperature' AND e.entity_id ~ "
    "'(printer|mcdata|fordpass)' AND e.entity_id !~ 'target'"
)
MAINS_POWER = [
    "sensor.vue_gen_3_left_side_power_minute_average",
    "sensor.vue_gen_3_right_side_power_minute_average",
]
MAINS_ENERGY = [
    "sensor.vue_gen_3_left_side_energy_today",
    "sensor.vue_gen_3_right_side_energy_today",
]
CIRCUIT_POWER = (
    "e.entity_id ~ '_power_minute_average' AND e.entity_id !~ "
    "'^sensor\\.vue_gen_3_(left|right)_side_power_minute_average$'"
)
CIRCUIT_ENERGY = (
    "e.entity_id ~ '_energy_today' AND e.entity_id !~ "
    "'^sensor\\.vue_gen_3_(left|right)_side_energy_today$'"
)
PLUG_POWER = (
    "e.domain = 'sensor' AND e.device_class = 'power' AND e.entity_id !~ "
    "'(minute_average|instantaneous_demand|tripplite)'"
)
OPENINGS = (
    "e.domain = 'binary_sensor' AND e.device_class IN ('door', 'window', 'opening', 'garage_door')"
)
MOTION = (
    "e.domain = 'binary_sensor' AND e.device_class IN ('motion', 'occupancy', 'presence') "
    "AND e.entity_id <> 'binary_sensor.frigate_activity'"
)
SAFETY = (
    "e.domain = 'binary_sensor' AND e.device_class IN "
    "('smoke', 'carbon_monoxide', 'gas', 'moisture', 'safety', 'heat') AND e.entity_id !~ 'rain'"
)
BATTERY = "e.domain = 'sensor' AND e.device_class = 'battery' AND e.unit = '%'"

TO_F = "CASE WHEN e.unit = '°C' THEN s.value * 9 / 5 + 32 ELSE s.value END"
STEP = "make_interval(secs => $__interval_ms / 1000.0)"


def arr(ids):
    return "ARRAY[" + ", ".join(f"'{i}'" for i in ids) + "]"


def sel(pred):
    return f"ARRAY(SELECT e.entity_id FROM ha_entities e WHERE {pred})"


def var_arr(name):
    return f"ARRAY[${{{name}:sqlstring}}]::text[]"


def series(ids_sql, value="s.value"):
    return (
        f'SELECT s.ts AS "time", e.label AS metric, {value} AS value\n'
        f"FROM ha_series({ids_sql}, $__timeFrom(), $__timeTo(), {STEP}) s\n"
        "JOIN ha_entities e USING (entity_id)\nORDER BY 1"
    )


def summed_series(ids_sql, name, agg="sum"):
    return (
        f'SELECT s.ts AS "time", {agg}(s.value) AS "{name}"\n'
        f"FROM ha_series({ids_sql}, $__timeFrom(), $__timeTo(), {STEP}) s\n"
        "GROUP BY 1 ORDER BY 1"
    )


def states(ids_sql, on=1, off=0):
    return (
        'SELECT s.ts AS "time", e.label AS metric,\n'
        f"  CASE s.state WHEN 'on' THEN {on} WHEN 'off' THEN {off} ELSE -1 END AS value\n"
        "FROM ha_states(" + ids_sql + ", $__timeFrom(), $__timeTo()) s\n"
        "JOIN ha_entities e USING (entity_id)\nORDER BY 1"
    )


def current(pred, value="ha_num(e.state)"):
    """Latest value per entity as a single-timestamp series (one field each)."""
    return (
        f'SELECT now() AS "time", e.label AS metric, {value} AS value\n'
        f"FROM ha_entities e WHERE {pred} AND ha_num(e.state) IS NOT NULL\nORDER BY 2"
    )


# ------------------------------------------------------------- panel builders
class Board:
    def __init__(self):
        self.panels = []
        self.y = 0
        self.x = 0
        self.row_h = 0

    def add(self, panel, w, h):
        if self.x + w > 24:
            self.y += self.row_h
            self.x = 0
            self.row_h = 0
        panel["id"] = len(self.panels) + 1
        panel["gridPos"] = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        self.panels.append(panel)

    def row(self, title):
        self.y += self.row_h
        self.x = 0
        self.row_h = 0
        self.panels.append({
            "type": "row", "title": title, "collapsed": False, "panels": [],
            "id": len(self.panels) + 1, "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1},
        })
        self.y += 1


def target(sql, fmt="time_series", ref="A"):
    return {"datasource": DS, "refId": ref, "editorMode": "code", "rawQuery": True,
            "format": fmt, "rawSql": sql}


def thresholds(*steps):
    """steps: color, (value, color), ..."""
    out = [{"color": steps[0], "value": None}]
    out += [{"color": c, "value": v} for v, c in steps[1:]]
    return {"mode": "absolute", "steps": out}


def timeseries(title, sql, unit, desc="", stack=False, bars=False, interp="smooth",
               thresholds_=None, fill=10, interval=None, legend="table", decimals=None,
               min_=None, time_from=None, extra_targets=()):
    custom = {
        "drawStyle": "bars" if bars else "line",
        "lineInterpolation": interp,
        "lineWidth": 1 if stack or bars else 2,
        "fillOpacity": 80 if bars else (60 if stack else fill),
        "gradientMode": "opacity",
        "showPoints": "never",
        "spanNulls": True,
        "stacking": {"mode": "normal" if stack else "none", "group": "A"},
        "axisSoftMin": min_,
        "thresholdsStyle": {"mode": "line+area" if thresholds_ else "off"},
        "barAlignment": -1,
    }
    defaults = {"unit": unit, "custom": custom, "color": {"mode": "palette-classic"}}
    if decimals is not None:
        defaults["decimals"] = decimals
    if thresholds_:
        defaults["thresholds"] = thresholds_
    p = {
        "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "targets": [target(sql)] + [target(s, ref=chr(66 + i)) for i, s in enumerate(extra_targets)],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "legend": ({"displayMode": "table", "placement": "right", "calcs": ["lastNotNull", "min", "max", "mean"],
                        "sortBy": "Last *", "sortDesc": True, "showLegend": True}
                       if legend == "table" else
                       {"displayMode": "list", "placement": "bottom", "showLegend": legend != "none"}),
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }
    if interval:
        p["interval"] = interval
    if time_from:
        p["timeFrom"] = time_from
    return p


def stat(title, sql, unit, thr=None, desc="", fmt="table", spark=False, decimals=None,
         text_mode="auto", color_mode="value", mappings=None, orientation="auto", calc="lastNotNull"):
    defaults = {"unit": unit, "thresholds": thr or thresholds("text"),
                "color": {"mode": "thresholds"}, "mappings": mappings or []}
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "type": "stat", "title": title, "description": desc, "datasource": DS,
        "targets": [target(sql, fmt)],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"reduceOptions": {"calcs": [calc], "fields": "", "values": False},
                    "graphMode": "area" if spark else "none", "colorMode": color_mode,
                    "textMode": text_mode, "justifyMode": "auto", "orientation": orientation,
                    "wideLayout": True},
    }


def timeline(title, sql, on_text, on_color, off_text="Clear", off_color="transparent", desc=""):
    mappings = [{"type": "value", "options": {
        "1": {"text": on_text, "color": on_color, "index": 0},
        "0": {"text": off_text, "color": off_color, "index": 1},
        "-1": {"text": "Unavailable", "color": "#555555", "index": 2},
    }}]
    return {
        "type": "state-timeline", "title": title, "description": desc, "datasource": DS,
        "targets": [target(sql)],
        "fieldConfig": {"defaults": {"mappings": mappings, "color": {"mode": "fixed", "fixedColor": "text"},
                                     "custom": {"fillOpacity": 85, "lineWidth": 0, "spanNulls": True}},
                        "overrides": []},
        "options": {"mergeValues": True, "showValue": "never", "alignValue": "left", "rowHeight": 0.8,
                    "legend": {"showLegend": False}, "tooltip": {"mode": "single"}},
    }


def table(title, sql, overrides=(), sort=None, desc="", time_from=None):
    p = {
        "type": "table", "title": title, "description": desc, "datasource": DS,
        "targets": [target(sql, "table")],
        "fieldConfig": {"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}},
                        "overrides": list(overrides)},
        "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False},
                    "sortBy": [sort] if sort else []},
    }
    if time_from:
        p["timeFrom"] = time_from
    return p


def col(name, **props):
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": k.replace("__", "."), "value": v} for k, v in props.items()]}


def gauge_cell(unit, thr, min_=0, max_=100):
    return {"unit": unit, "min": min_, "max": max_, "thresholds": thr,
            "custom__cellOptions": {"type": "gauge", "mode": "basic", "valueDisplayMode": "text"}}


def query_var(name, label, pred, all_=True, default_all=True, hide=0):
    q = f"SELECT e.label AS __text, e.entity_id AS __value FROM ha_entities e WHERE {pred} ORDER BY 1"
    v = {"type": "query", "name": name, "label": label, "datasource": DS, "query": q,
         "definition": q, "refresh": 1, "multi": True, "includeAll": all_, "sort": 0, "hide": hide,
         "options": []}
    if default_all:
        v["current"] = {"selected": True, "text": ["All"], "value": ["$__all"]}
    return v


def dashboard(uid, title, board, desc, variables=(), time="now-24h", refresh="1m"):
    return {
        "uid": uid, "title": title, "description": desc, "tags": TAGS,
        "schemaVersion": 39, "version": 1, "editable": True, "graphTooltip": 1,
        "timezone": "browser", "refresh": refresh,
        "time": {"from": time, "to": "now"},
        "templating": {"list": list(variables)},
        "links": [{"type": "dashboards", "tags": TAGS, "asDropdown": False, "includeVars": False,
                   "keepTime": True, "title": "Home Assistant", "icon": "external link"}],
        "annotations": {"list": []},
        "panels": board.panels,
    }


TEMP_THR = thresholds("blue", (62, "green"), (76, "orange"), (82, "red"))
HUM_THR = thresholds("orange", (30, "green"), (60, "orange"), (70, "red"))
BATT_THR = thresholds("red", (20, "orange"), (40, "green"))
OK_THR = thresholds("green", (1, "red"))


# ------------------------------------------------------------------ Overview
def overview():
    b = Board()
    b.add(stat("Outdoor", series(arr(["sensor.outdoor_temperature"])), "fahrenheit",
               TEMP_THR, fmt="time_series", spark=True, decimals=1), 3, 4)
    b.add(stat("Indoor (avg)", f"SELECT avg(ha_num(e.state) * CASE WHEN e.unit = '°C' THEN 1.8 ELSE 1 END"
               f" + CASE WHEN e.unit = '°C' THEN 32 ELSE 0 END) FROM ha_entities e WHERE {INDOOR_TEMP}",
               "fahrenheit", TEMP_THR, decimals=1,
               desc="Average of the indoor room temperature sensors."), 3, 4)
    b.add(stat("Outdoor humidity", series(arr(["sensor.outdoor_humidity"])), "humidity",
               thresholds("text"), fmt="time_series", spark=True, decimals=0), 3, 4)
    b.add(stat("House power", f"SELECT sum(ha_num(e.state)) FROM ha_entities e WHERE e.entity_id = ANY ({arr(MAINS_POWER)})",
               "watt", thresholds("green", (3000, "orange"), (6000, "red")), decimals=0,
               desc="Both Emporia Vue mains legs (1-minute average)."), 3, 4)
    b.add(stat("Energy today", f"SELECT sum(ha_num(e.state)) FROM ha_entities e WHERE e.entity_id = ANY ({arr(MAINS_ENERGY)})",
               "kwatth", thresholds("text"), decimals=1), 3, 4)
    b.add(stat("Open doors & windows", f"SELECT count(*) FILTER (WHERE e.state = 'on') FROM ha_entities e WHERE {OPENINGS}",
               "none", thresholds("green", (1, "orange"))), 3, 4)
    b.add(stat("Safety alarms", f"SELECT count(*) FILTER (WHERE e.state = 'on') FROM ha_entities e WHERE {SAFETY}",
               "none", OK_THR, desc="Smoke, CO, gas, leak and heat sensors currently triggered."), 3, 4)
    b.add(stat("Low batteries", f"SELECT count(*) FROM ha_entities e WHERE ({BATTERY} AND ha_num(e.state) < 20)"
               " OR (e.domain = 'binary_sensor' AND e.device_class = 'battery' AND e.state = 'on')",
               "none", thresholds("green", (1, "orange"))), 3, 4)

    b.add(timeseries("House power", summed_series(arr(MAINS_POWER), "House"), "watt",
                     interval="1m", legend="list", fill=25,
                     desc="Sum of both Emporia Vue mains legs."), 12, 9)
    indoor_avg = summed_series(sel(INDOOR_TEMP), "Indoor (avg)", agg="avg")
    b.add(timeseries("Indoor vs outdoor", series(arr(OUTDOOR_TEMP), TO_F), "fahrenheit",
                     interval="5m", legend="list", extra_targets=[indoor_avg]), 12, 9)

    b.add(timeseries("Daily energy (last 14 days)",
                     'SELECT d.day AS "time", sum(d.delta) AS "kWh"\n'
                     f"FROM ha_daily_delta({arr(MAINS_ENERGY)}, $__timeFrom(), $__timeTo()) d\n"
                     "GROUP BY 1 ORDER BY 1",
                     "kwatth", bars=True, legend="none", interval="1d", time_from="14d", decimals=1), 8, 9)
    b.add(table("Open right now",
                f"SELECT e.label AS \"Sensor\", initcap(e.device_class) AS \"Type\", e.last_seen AS \"Since\"\n"
                f"FROM ha_entities e WHERE (({OPENINGS}) OR ({SAFETY})) AND e.state = 'on' ORDER BY 3 DESC",
                [col("Since", unit="dateTimeFromNow")]), 6, 9)
    b.add(table("Recent activity",
                "SELECT l.time AS \"Time\", e.label AS \"Sensor\",\n"
                "  CASE WHEN e.device_class IN ('door','window','opening','garage_door')\n"
                "       THEN CASE l.state WHEN 'on' THEN 'Opened' ELSE 'Closed' END\n"
                "       WHEN e.device_class IN ('motion','occupancy','presence')\n"
                "       THEN CASE l.state WHEN 'on' THEN 'Motion' ELSE 'Clear' END\n"
                "       ELSE CASE l.state WHEN 'on' THEN 'ALARM' ELSE 'Clear' END END AS \"Event\"\n"
                "FROM ltss l JOIN ha_entities e USING (entity_id)\n"
                f"WHERE (({OPENINGS}) OR ({MOTION}) OR ({SAFETY})) AND l.state = 'on'\n"
                "  AND l.time > now() - INTERVAL '24 hours'\nORDER BY l.time DESC LIMIT 25",
                [col("Time", unit="dateTimeFromNow"),
                 col("Event", custom__cellOptions={"type": "color-text"},
                     mappings=[{"type": "value", "options": {
                         "Opened": {"color": "orange", "index": 0}, "Motion": {"color": "blue", "index": 1},
                         "ALARM": {"color": "red", "index": 2}}}])]), 10, 9)

    return dashboard("ha-overview", "Home Assistant — Overview", b,
                     "At-a-glance house status from Home Assistant (LTSS on TimescaleDB).")


# ------------------------------------------------------------------- Climate
def climate():
    b = Board()
    b.add(stat("Rooms now",
               current(f"e.entity_id IN (${{rooms:sqlstring}})",
                       "ha_num(e.state) * CASE WHEN e.unit = '°C' THEN 1.8 ELSE 1 END"
                       " + CASE WHEN e.unit = '°C' THEN 32 ELSE 0 END"),
               "fahrenheit", TEMP_THR, fmt="time_series", decimals=1, text_mode="value_and_name",
               color_mode="background"), 24, 5)
    b.add(timeseries("Room temperature", series(var_arr("rooms"), TO_F), "fahrenheit",
                     interval="2m", decimals=1), 24, 10)
    b.add(timeseries("Outdoor", series(arr(OUTDOOR_TEMP), TO_F), "fahrenheit",
                     interval="5m", legend="list", decimals=1,
                     desc="Outdoor sensor and the weather integration's feels-like temperature."), 12, 8)
    b.add(table("Range summary",
                "SELECT e.label AS \"Room\",\n"
                "  min(h.min) * CASE WHEN e.unit = '°C' THEN 1.8 ELSE 1 END + CASE WHEN e.unit = '°C' THEN 32 ELSE 0 END AS \"Min\",\n"
                "  sum(h.avg * h.samples) / sum(h.samples) * CASE WHEN e.unit = '°C' THEN 1.8 ELSE 1 END + CASE WHEN e.unit = '°C' THEN 32 ELSE 0 END AS \"Mean\",\n"
                "  max(h.max) * CASE WHEN e.unit = '°C' THEN 1.8 ELSE 1 END + CASE WHEN e.unit = '°C' THEN 32 ELSE 0 END AS \"Max\"\n"
                "FROM ltss_hourly h JOIN ha_entities e USING (entity_id)\n"
                "WHERE h.entity_id IN (${rooms:sqlstring}) AND $__timeFilter(h.bucket)\n"
                "GROUP BY e.label, e.unit ORDER BY 1",
                [col("Min", unit="fahrenheit", decimals=1, color={"mode": "thresholds"}, thresholds=TEMP_THR,
                     custom__cellOptions={"type": "color-text"}),
                 col("Mean", unit="fahrenheit", decimals=1),
                 col("Max", unit="fahrenheit", decimals=1, color={"mode": "thresholds"}, thresholds=TEMP_THR,
                     custom__cellOptions={"type": "color-text"})],
                desc="Per-room min/mean/max over the selected time range (hourly rollup)."), 12, 8)

    b.row("Humidity")
    b.add(stat("Humidity now", current(f"e.entity_id IN (${{humidity:sqlstring}})"), "humidity",
               HUM_THR, fmt="time_series", decimals=0, text_mode="value_and_name",
               color_mode="background"), 24, 4)
    b.add(timeseries("Relative humidity", series(var_arr("humidity")), "humidity",
                     interval="2m", decimals=0), 16, 9)
    b.add(timeseries("Outdoor humidity", series(arr(["sensor.outdoor_humidity", "sensor.home_humidity"])),
                     "humidity", interval="5m", legend="list", decimals=0), 8, 9)

    b.row("Fridges, freezers & equipment")
    b.add(timeseries("Fridge & freezer", series(arr(COLD_STORAGE), TO_F), "fahrenheit",
                     interval="2m", legend="list", decimals=1,
                     thresholds_=thresholds("transparent", (40, "red")),
                     desc="Red band: above 40°F, unsafe for a fridge. Freezers should sit near 0°F."), 12, 9)
    b.add(timeseries("Equipment", series(sel(EQUIPMENT_TEMP), TO_F), "fahrenheit",
                     interval="2m", decimals=0,
                     desc="3D printer, NAS drives and car coolant."), 12, 9)

    return dashboard("ha-climate", "Home Assistant — Climate", b,
                     "Room temperatures and humidity, cold storage and equipment temperatures.",
                     variables=[query_var("rooms", "Rooms", INDOOR_TEMP),
                                query_var("humidity", "Humidity sensors", INDOOR_HUMIDITY)])


# -------------------------------------------------------------------- Energy
def energy():
    b = Board()
    b.add(stat("House power", f"SELECT sum(ha_num(e.state)) FROM ha_entities e WHERE e.entity_id = ANY ({arr(MAINS_POWER)})",
               "watt", thresholds("green", (3000, "orange"), (6000, "red")), decimals=0), 4, 4)
    b.add(stat("Unmonitored", "SELECT ha_num(e.state) FROM ha_entities e WHERE e.entity_id = 'sensor.balance_power_minute_average'",
               "watt", thresholds("text"), decimals=0,
               desc="Emporia 'Balance': mains load not on any monitored circuit."), 4, 4)
    b.add(stat("Today", f"SELECT sum(ha_num(e.state)) FROM ha_entities e WHERE e.entity_id = ANY ({arr(MAINS_ENERGY)})",
               "kwatth", thresholds("text"), decimals=1), 4, 4)
    bill = ("SELECT h.last FROM ltss_hourly h WHERE h.entity_id = '{}' ORDER BY h.bucket DESC LIMIT 1")
    b.add(stat("Bill to date", bill.format("sensor.current_bill_electric_cost_to_date_2"), "currencyUSD",
               thresholds("text"), decimals=2, desc="Opower; last value reported."), 4, 4)
    b.add(stat("Bill forecast", bill.format("sensor.current_bill_electric_forecasted_cost_2"), "currencyUSD",
               thresholds("text"), decimals=2, desc="Opower; last value reported."), 4, 4)
    b.add(stat("Typical month", bill.format("sensor.typical_monthly_electric_cost_2"), "currencyUSD",
               thresholds("text"), decimals=2, desc="Opower; last value reported."), 4, 4)

    b.add(timeseries("Mains", series(arr(MAINS_POWER)), "watt", stack=True, interval="1m",
                     legend="list", desc="Emporia Vue mains legs, stacked = whole-house load."), 12, 9)
    b.add(table("Circuits right now",
                f"SELECT e.label AS \"Circuit\", ha_num(e.state) AS \"Power\"\nFROM ha_entities e\n"
                f"WHERE ({CIRCUIT_POWER}) AND ha_num(e.state) > 0\nORDER BY 2 DESC",
                [col("Power", unit="watt", decimals=0, color={"mode": "continuous-GrYlRd"}, min=0,
                     custom__cellOptions={"type": "gauge", "mode": "gradient", "valueDisplayMode": "text"})],
                sort={"displayName": "Power", "desc": True}), 12, 9)
    b.add(timeseries("Circuits", series(var_arr("circuits")), "watt", stack=True, interval="1m",
                     interp="stepAfter", desc="Emporia Vue circuits (1-minute averages), stacked."), 24, 11)

    b.row("Consumption")
    b.add(timeseries("Daily energy by circuit",
                     'SELECT d.day AS "time", e.label AS metric, d.delta AS value\n'
                     "FROM ha_daily_delta(" + var_arr("circuit_energy") + ", $__timeFrom(), $__timeTo()) d\n"
                     "JOIN ha_entities e USING (entity_id)\nORDER BY 1",
                     "kwatth", bars=True, stack=True, interval="1d", time_from="14d", decimals=2,
                     legend="list"), 16, 10)
    b.add(table("Energy by circuit (range)",
                'SELECT e.label AS "Circuit", sum(d.delta) AS "kWh"\n'
                "FROM ha_daily_delta(" + var_arr("circuit_energy") + ", $__timeFrom(), $__timeTo()) d\n"
                "JOIN ha_entities e USING (entity_id)\nGROUP BY 1 ORDER BY 2 DESC",
                [col("kWh", unit="kwatth", decimals=2, color={"mode": "continuous-BlPu"},
                     custom__cellOptions={"type": "gauge", "mode": "basic", "valueDisplayMode": "text"})],
                sort={"displayName": "kWh", "desc": True}), 8, 10)

    b.row("Plugs & UPS")
    b.add(timeseries("Smart plugs", series(sel(PLUG_POWER)), "watt", interval="1m",
                     interp="stepAfter", legend="list"), 12, 8)
    b.add(timeseries("UPS load", series(arr(["sensor.tripplite_ups_power_usage"])), "watt",
                     interval="1m", legend="none", fill=25,
                     extra_targets=[]), 6, 8)
    b.add(timeseries("UPS voltage",
                     series(arr(["sensor.tripplite_input_voltage", "sensor.tripplite_output_voltage"])),
                     "volt", interval="1m", legend="list", decimals=1), 6, 8)

    return dashboard("ha-energy", "Home Assistant — Energy", b,
                     "Emporia Vue mains and circuits, daily consumption, smart plugs and UPS.",
                     variables=[query_var("circuits", "Circuits", CIRCUIT_POWER),
                                query_var("circuit_energy", "Circuits (energy)", CIRCUIT_ENERGY, hide=2)])


# ------------------------------------------------------------------ Activity
def activity():
    b = Board()
    b.add(stat("Open now", f"SELECT count(*) FILTER (WHERE e.state = 'on') FROM ha_entities e WHERE {OPENINGS}",
               "none", thresholds("green", (1, "orange"))), 4, 4)
    b.add(stat("Safety alarms", f"SELECT count(*) FILTER (WHERE e.state = 'on') FROM ha_entities e WHERE {SAFETY}",
               "none", OK_THR), 4, 4)
    b.add(stat("Motion events", "SELECT count(*) FROM ltss l JOIN ha_entities e USING (entity_id)\n"
               f"WHERE {MOTION} AND l.state = 'on' AND $__timeFilter(l.time)", "none",
               thresholds("text"), desc="Motion/occupancy activations in the selected range."), 4, 4)
    b.add(stat("Door/window openings", "SELECT count(*) FROM ltss l JOIN ha_entities e USING (entity_id)\n"
               f"WHERE {OPENINGS} AND l.state = 'on' AND $__timeFilter(l.time)", "none",
               thresholds("text")), 4, 4)
    b.add(stat("Last motion", "SELECT max(l.time) FROM ltss l JOIN ha_entities e USING (entity_id)\n"
               f"WHERE {MOTION} AND l.state = 'on' AND l.time > now() - INTERVAL '7 days'",
               "dateTimeFromNow", thresholds("text")), 8, 4)

    b.add(timeline("Doors & windows", states(var_arr("openings")), "Open", "orange", "Closed", "green"), 24, 8)
    b.add(timeline("Motion & presence", states(var_arr("motion")), "Motion", "blue"), 24, 10)
    b.add(timeline("Safety", states(sel(SAFETY)), "ALARM", "red", "OK", "green",
                   desc="Smoke, CO, gas, leak and heat detectors."), 24, 10)

    b.add(timeseries("Motion per hour",
                     f'SELECT time_bucket({STEP}, l.time) AS "time", e.label AS metric, count(*) AS value\n'
                     "FROM ltss l JOIN ha_entities e USING (entity_id)\n"
                     "WHERE l.entity_id IN (${motion:sqlstring}) AND l.state = 'on' AND $__timeFilter(l.time)\n"
                     "GROUP BY 1, 2 ORDER BY 1",
                     "none", bars=True, stack=True, interval="1h", legend="list", decimals=0), 14, 9)
    b.add(table("Last activity",
                'SELECT e.label AS "Sensor", max(l.time) FILTER (WHERE l.state = \'on\') AS "Last active",\n'
                "  count(*) FILTER (WHERE l.state = 'on') AS \"Activations\"\n"
                "FROM ltss l JOIN ha_entities e USING (entity_id)\n"
                "WHERE (l.entity_id IN (${motion:sqlstring}) OR l.entity_id IN (${openings:sqlstring}))\n"
                "  AND $__timeFilter(l.time)\nGROUP BY 1 ORDER BY 2 DESC NULLS LAST",
                [col("Last active", unit="dateTimeFromNow")]), 10, 9)

    return dashboard("ha-activity", "Home Assistant — Activity & Safety", b,
                     "Doors, windows, motion and safety sensors.",
                     variables=[query_var("openings", "Doors & windows", OPENINGS),
                                query_var("motion", "Motion sensors", MOTION)])


# ------------------------------------------------------------------- Devices
def devices():
    b = Board()
    b.add(stat("Entities tracked", "SELECT count(*) FROM ha_entities e WHERE e.last_seen > now() - INTERVAL '7 days'",
               "none", thresholds("text")), 4, 4)
    b.add(stat("Unavailable", "SELECT count(*) FROM ha_entities e WHERE e.state IN ('unavailable', 'unknown')",
               "none", thresholds("green", (1, "orange"), (25, "red"))), 4, 4)
    b.add(stat("Low batteries", f"SELECT count(*) FROM ha_entities e WHERE ({BATTERY} AND ha_num(e.state) < 20)"
               " OR (e.domain = 'binary_sensor' AND e.device_class = 'battery' AND e.state = 'on')",
               "none", thresholds("green", (1, "orange"))), 4, 4)
    b.add(stat("LTSS lag", "SELECT extract(epoch FROM now() - max(time)) FROM ltss", "s",
               thresholds("green", (60, "orange"), (300, "red")), decimals=0,
               desc="Age of the newest row LTSS has written. Spikes after a Home Assistant restart while it drains its queue."), 4, 4)
    b.add(stat("Stored", "SELECT hypertable_size('ltss') + coalesce((SELECT hypertable_size('ltss_hourly')), 0)",
               "bytes", thresholds("text"), decimals=1), 4, 4)
    b.add(stat("Compression ratio",
               "SELECT before_compression_total_bytes::float / nullif(after_compression_total_bytes, 0)\n"
               "FROM hypertable_compression_stats('ltss')", "x", thresholds("text"), decimals=1,
               desc="Chunks compress after 7 days."), 4, 4)

    b.add(table("Battery levels",
                f"SELECT e.label AS \"Device\", ha_num(e.state) AS \"Battery\", e.last_seen AS \"Reported\"\n"
                f"FROM ha_entities e WHERE {BATTERY} AND ha_num(e.state) IS NOT NULL ORDER BY 2",
                [col("Battery", **gauge_cell("percent", BATT_THR)), col("Reported", unit="dateTimeFromNow")],
                sort={"displayName": "Battery", "desc": False}), 10, 14)
    b.add(timeseries("Battery history", series(var_arr("batteries")), "percent", interval="1h",
                     interp="stepAfter", legend="list", min_=0), 14, 14)

    b.add(table("Unavailable or unknown",
                'SELECT e.label AS "Entity", e.entity_id AS "ID", e.state AS "State", e.last_seen AS "Since"\n'
                "FROM ha_entities e WHERE e.state IN ('unavailable', 'unknown') ORDER BY e.last_seen DESC",
                [col("Since", unit="dateTimeFromNow")]), 12, 10)
    b.add(table("Problems reported",
                'SELECT e.label AS "Entity", e.entity_id AS "ID", e.last_seen AS "Since"\n'
                "FROM ha_entities e WHERE e.domain = 'binary_sensor' AND e.device_class = 'problem'"
                " AND e.state = 'on' ORDER BY e.last_seen DESC",
                [col("Since", unit="dateTimeFromNow")],
                desc="binary_sensors with device_class problem that are on (updates available, faults)."), 12, 10)

    b.row("LTSS pipeline")
    b.add(timeseries("Rows written",
                     f'SELECT time_bucket({STEP}, time) AS "time", count(*) AS "rows"\n'
                     "FROM ltss WHERE $__timeFilter(time) GROUP BY 1 ORDER BY 1",
                     "short", bars=True, interval="5m", legend="none"), 12, 8)
    b.add(table("Top writers (24h)",
                'SELECT e.label AS "Entity", l.entity_id AS "ID", count(*) AS "Rows"\n'
                "FROM ltss l JOIN ha_entities e USING (entity_id)\n"
                "WHERE l.time > now() - INTERVAL '24 hours'\nGROUP BY 1, 2 ORDER BY 3 DESC LIMIT 20",
                [col("Rows", custom__cellOptions={"type": "gauge", "mode": "basic", "valueDisplayMode": "text"},
                     color={"mode": "continuous-BlPu"})],
                desc="Entities writing the most rows: candidates to exclude from LTSS."), 12, 8)

    return dashboard("ha-devices", "Home Assistant — Devices & Health", b,
                     "Batteries, unavailable entities, problems and the LTSS pipeline.",
                     variables=[query_var("batteries", "Batteries", BATTERY)])


# ------------------------------------------------------------------ Explorer
def explorer():
    b = Board()
    b.add(timeseries("Numeric history", series(var_arr("entity")), "short", interval="1m",
                     desc="Numeric states, averaged per interval (hourly rollup for long ranges)."), 24, 11)
    b.add(table("Current state",
                'SELECT e.label AS "Entity", e.entity_id AS "ID", e.state AS "State", e.unit AS "Unit",\n'
                '  e.device_class AS "Class", e.last_seen AS "Changed"\n'
                "FROM ha_entities e WHERE e.entity_id IN (${entity:sqlstring}) ORDER BY 1",
                [col("Changed", unit="dateTimeFromNow")]), 12, 8)
    b.add(table("Latest attributes",
                'SELECT e.label AS "Entity", a.key AS "Attribute", a.value AS "Value"\n'
                "FROM ha_entities e\n"
                "CROSS JOIN LATERAL (SELECT l.attributes FROM ltss l WHERE l.entity_id = e.entity_id\n"
                "                    ORDER BY l.time DESC LIMIT 1) l\n"
                "CROSS JOIN LATERAL jsonb_each_text(l.attributes) a\n"
                "WHERE e.entity_id IN (${entity:sqlstring}) ORDER BY 1, 2"), 12, 8)
    b.add(table("State changes",
                'SELECT l.time AS "Time", e.label AS "Entity", l.state AS "State"\n'
                "FROM ltss l JOIN ha_entities e USING (entity_id)\n"
                "WHERE l.entity_id IN (${entity:sqlstring}) AND $__timeFilter(l.time)\n"
                "ORDER BY l.time DESC LIMIT 1000"), 24, 12)

    domain_q = "SELECT DISTINCT e.domain FROM ha_entities e ORDER BY 1"
    entity_q = ("SELECT e.label || ' — ' || e.entity_id AS __text, e.entity_id AS __value\n"
                "FROM ha_entities e WHERE e.domain IN (${domain:sqlstring}) ORDER BY e.entity_id")
    variables = [
        {"type": "query", "name": "domain", "label": "Domain", "datasource": DS, "query": domain_q,
         "definition": domain_q, "refresh": 1, "multi": True, "includeAll": True, "sort": 1,
         "current": {"selected": True, "text": ["sensor"], "value": ["sensor"]}, "options": []},
        {"type": "query", "name": "entity", "label": "Entity", "datasource": DS, "query": entity_q,
         "definition": entity_q, "refresh": 1, "multi": True, "includeAll": False, "sort": 0,
         "current": {"selected": True, "text": ["Outdoor — sensor.outdoor_temperature"],
                     "value": ["sensor.outdoor_temperature"]}, "options": []},
    ]
    return dashboard("ha-explorer", "Home Assistant — Entity Explorer", b,
                     "Any LTSS entity: numeric history, current state, attributes and changes.",
                     variables=variables)


def main():
    for name, build in [("overview", overview), ("climate", climate), ("energy", energy),
                        ("activity", activity), ("devices", devices), ("explorer", explorer)]:
        path = OUT / f"ha-{name}.json"
        path.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(OUT.parent.parent.parent)}")


if __name__ == "__main__":
    main()
