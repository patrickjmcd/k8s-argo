#!/usr/bin/env python3
"""Build core/grafana-assets/dashboards/sonos.json -- edit this and rerun, not the JSON.

    python3 scripts/sonos_dashboard.py ../sonos-exporter/grafana/sonos-overview-dashboard.json

The source is the upstream dashboard from our fork github.com/patrickjmcd/sonos-exporter,
in Grafana's v2 dashboard schema (dashboard.grafana.app/v2), which the Grafana sidecar's file
provisioning doesn't load. This script:

  * converts it to classic dashboard JSON (rows of grid panels; the Battery and Legacy radio
    rows lose their "show only with data" condition, which classic JSON can't express);
  * wraps every query in `max without (pod, instance) (...)`: series scraped before the
    ServiceMonitor dropped those labels (apps/sonos-exporter/servicemonitor.yaml, Nov 12
    2026) would otherwise show each zone once per exporter pod until they age out;
  * adds a "UniFi (access point view)" row from unpoller, joined to Sonos zone names by IP.
"""
import json
import sys

OUT = "core/grafana-assets/dashboards/sonos.json"
UID = "sonos-overview"
DS = {"type": "prometheus", "uid": "${datasource}"}

# unpoller client series -> Sonos zone label, matched on IP (sonos_device_info's ip_address).
JOIN = ('on (ip) group_left (zone) label_replace(max by (ip_address, zone) (sonos_device_info), '
        '"ip", "$1", "ip_address", "(.+)")')


def dedupe(expr):
    return f"max without (pod, instance) ({expr})"


def convert(src):
    d = json.load(open(src))
    s = d["spec"]
    els = s["elements"]

    def panel(name, gp):
        e = els[name]["spec"]
        v = e["vizConfig"]
        q = e["data"]["spec"]
        targets = []
        for pq in q["queries"]:
            t = dict(pq["spec"]["query"]["spec"])
            t["refId"] = pq["spec"]["refId"]
            if "expr" in t:
                t["expr"] = dedupe(t["expr"])
            if pq["spec"].get("hidden"):
                t["hide"] = True
            targets.append(t)
        p = {"id": e["id"], "type": v["group"], "title": e["title"],
             "description": e.get("description", ""), "links": e.get("links", []),
             "gridPos": gp, "targets": targets, "datasource": DS,
             "fieldConfig": v["spec"].get("fieldConfig", {"defaults": {}, "overrides": []}),
             "options": v["spec"].get("options", {}),
             "transformations": q.get("transformations", [])}
        for k in ("interval", "maxDataPoints", "timeFrom", "timeShift", "hideTimeOverride"):
            if k in q.get("queryOptions", {}):
                p[k] = q["queryOptions"][k]
        return p

    rows = []
    for row in s["layout"]["spec"]["rows"]:
        r = row["spec"]
        kids = []
        for it in r["layout"]["spec"]["items"]:
            i = it["spec"]
            kids.append(panel(i["element"]["name"],
                              {"x": i["x"], "y": i["y"], "w": i["width"], "h": i["height"]}))
        rep = r.get("repeat")
        rows.append({"title": r.get("title", ""), "collapsed": r.get("collapse", False),
                     "repeat": (rep.get("value") if isinstance(rep, dict) else rep) if rep else None,
                     "panels": kids})

    tmpl = [{"name": "datasource", "type": "datasource", "query": "prometheus",
             "label": "Data source", "current": {"text": "default", "value": "default"}}]
    for v in s.get("variables", []):
        vs = v["spec"]
        tmpl.append({"name": vs["name"], "label": vs.get("label"), "type": "query",
                     "datasource": DS, "definition": vs.get("definition"),
                     "query": vs["query"]["spec"], "multi": vs.get("multi"),
                     "includeAll": vs.get("includeAll"),
                     "refresh": 2 if vs.get("refresh") == "onTimeRangeChanged" else 1,
                     "sort": 1 if vs.get("sort") == "alphabeticalAsc" else 0,
                     "regex": vs.get("regex", ""),
                     "current": {"text": ["All"], "value": ["$__all"]}, "options": []})
    ts = s["timeSettings"]
    meta = {"uid": UID, "title": s["title"], "description": s.get("description", ""),
            "tags": s.get("tags", []), "editable": s.get("editable", True),
            "graphTooltip": 1 if s.get("cursorSync") == "Crosshair" else 0,
            "schemaVersion": 41, "version": 1, "time": {"from": ts["from"], "to": ts["to"]},
            "refresh": ts.get("autoRefresh", ""), "timezone": ts.get("timezone", "browser"),
            "templating": {"list": tmpl}, "annotations": {"list": []},
            "links": s.get("links", [])}
    return meta, rows


def unifi_row(first_id):
    ids = iter(range(first_id, first_id + 100))

    def ts(title, expr, unit, desc, x, y, w=12, h=8, extra=None):
        p = {"id": next(ids), "type": "timeseries", "title": title, "description": desc,
             "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
             "targets": [{"refId": "A", "expr": expr, "legendFormat": "{{zone}}", "range": True}],
             "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 1, "fillOpacity": 0,
                                                                   "spanNulls": False}},
                             "overrides": []},
             "options": {"legend": {"displayMode": "table", "placement": "right",
                                    "calcs": ["lastNotNull", "max"]},
                         "tooltip": {"mode": "multi", "sort": "desc"}}}
        if extra:
            p["fieldConfig"]["defaults"].update(extra)
        return p

    def tq(ref, expr):
        return {"refId": ref, "expr": expr, "instant": True, "range": False, "format": "table"}

    table = {
        "id": next(ids), "type": "table", "title": "Access point per speaker",
        "description": ("What the UniFi access point sees for each Wi-Fi speaker (unpoller). "
                        "Satellites (Sub, surrounds) talk only to their soundbar and don't appear. "
                        "Signal is dBm (closer to 0 is stronger); retries are the AP's resends "
                        "to the speaker as a share of packets sent."),
        "datasource": DS, "gridPos": {"x": 0, "y": 0, "w": 24, "h": 7},
        "targets": [
            tq("A", f"max by (zone, ap_name, radio, channel) (unpoller_client_radio_signal_db * {JOIN}) != 0"),
            tq("B", f"max by (zone) (unpoller_client_satisfaction_ratio * {JOIN}) != 0"),
            tq("C", f"sum by (zone) (rate(unpoller_client_transmit_retries_total[15m]) * {JOIN})"
                    f" / sum by (zone) (rate(unpoller_client_transmit_packets_total[15m]) * {JOIN}) > 0"),
            tq("D", f"sum by (zone) (increase(unpoller_client_roam_count_total[1d]) * {JOIN})"),
        ],
        "transformations": [
            {"id": "merge", "options": {}},
            {"id": "filterByValue", "options": {"type": "include", "match": "all", "filters": [
                {"fieldName": "ap_name", "config": {"id": "regex", "options": {"value": ".+"}}}]}},
            {"id": "organize", "options": {
                "excludeByName": {"Time": True},
                "indexByName": {"zone": 0, "ap_name": 1, "radio": 2, "channel": 3,
                                "Value #A": 4, "Value #B": 5, "Value #C": 6, "Value #D": 7},
                "renameByName": {"zone": "Speaker", "ap_name": "Access point", "radio": "Radio",
                                 "channel": "Channel", "Value #A": "Signal",
                                 "Value #B": "Satisfaction", "Value #C": "TX retries",
                                 "Value #D": "Roams (24h)"}}},
        ],
        "fieldConfig": {"defaults": {}, "overrides": [
            {"matcher": {"id": "byName", "options": "Signal"},
             "properties": [{"id": "unit", "value": "dBm"}]},
            {"matcher": {"id": "byName", "options": "Satisfaction"},
             "properties": [{"id": "unit", "value": "percentunit"}]},
            {"matcher": {"id": "byName", "options": "TX retries"},
             "properties": [{"id": "unit", "value": "percentunit"},
                            {"id": "decimals", "value": 1}]},
            {"matcher": {"id": "byName", "options": "Roams (24h)"},
             "properties": [{"id": "decimals", "value": 0}]},
            {"matcher": {"id": "byName", "options": "Radio"},
             "properties": [{"id": "mappings", "value": [{"type": "value", "options": {
                 "ng": {"text": "2.4 GHz"}, "na": {"text": "5 GHz"}, "6e": {"text": "6 GHz"}}}]}]},
        ]},
        "options": {"showHeader": True, "sortBy": [{"displayName": "Speaker", "desc": False}]},
    }
    panels = [
        table,
        ts("Signal at the AP", f"max by (zone) (unpoller_client_radio_signal_db * {JOIN}) != 0", "dBm",
           "Speaker signal as the access point hears it. Below about -70 dBm expect retries.", 0, 7),
        ts("TX retries (AP side)",
           f"sum by (zone) (rate(unpoller_client_transmit_retries_total[15m]) * {JOIN})"
           f" / sum by (zone) (rate(unpoller_client_transmit_packets_total[15m]) * {JOIN})",
           "percentunit",
           "Share of packets the AP had to resend to the speaker: the AP's side of the PHY "
           "error panels above. Only meaningful while the speaker is streaming.", 12, 7),
        ts("Satisfaction", f"max by (zone) (unpoller_client_satisfaction_ratio * {JOIN}) != 0",
           "percentunit", "UniFi's per-client experience score.", 0, 15,
           extra={"min": 0, "max": 1}),
        ts("Roams per hour", f"sum by (zone) (increase(unpoller_client_roam_count_total[1h]) * {JOIN})",
           "short", "Access point changes. A fixed speaker roaming means two APs are about equally "
           "strong where it sits.", 12, 15),
    ]
    return {"title": "UniFi (access point view)", "collapsed": False, "repeat": None, "panels": panels}


def layout(rows):
    out, y, rid = [], 0, 1000
    for r in rows:
        rp = {"id": rid, "type": "row", "title": r["title"], "collapsed": r["collapsed"],
              "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}
        if r["repeat"]:
            rp["repeat"] = r["repeat"]
        rid += 1
        kids = []
        for p in r["panels"]:
            p = dict(p)
            g = p["gridPos"]
            p["gridPos"] = {**g, "y": y + 1 + g["y"]}
            kids.append(p)
        height = max([k["gridPos"]["y"] + k["gridPos"]["h"] for k in kids] + [y + 1]) - y
        if r["collapsed"]:
            rp["panels"] = kids
            out.append(rp)
            y += 1
        else:
            out.append(rp)
            out.extend(kids)
            y += height
    return out


def main():
    meta, rows = convert(sys.argv[1])
    max_id = max(p["id"] for r in rows for p in r["panels"])
    # Before the collapsed per-zone repeat row, which stays last.
    rows.insert(len(rows) - 1 if rows[-1]["repeat"] else len(rows), unifi_row(max_id + 1))
    meta["panels"] = layout(rows)
    with open(OUT, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
