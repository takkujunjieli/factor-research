"""Local, user-editable stock pools. Standard-library only for dashboard APIs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

SYMBOL = re.compile(r"[A-Z][A-Z0-9.\-]{0,14}")
PROFILE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
FILTERS = {"min_price": 5, "min_dollar_volume": 10000000, "volume_window": 20,
           "min_market_cap": 0, "max_market_cap": None, "min_history": 61}


def symbols(values):
    if not isinstance(values, list) or len(values) > 1500:
        raise ValueError("股票列表必须是最多1500项的数组")
    normalized = []
    for value in values:
        if not isinstance(value, str) or not SYMBOL.fullmatch(value.strip().upper()):
            raise ValueError("股票代码格式无效")
        normalized.append(value.strip().upper())
    return sorted(set(normalized))


def validate_settings(document, catalog):
    if not isinstance(document, dict) or set(document) != {"version", "active", "profiles"} or document["version"] != 1:
        raise ValueError("股票池配置格式无效")
    profiles = document["profiles"]
    if not isinstance(profiles, dict) or not 1 <= len(profiles) <= 30 or document["active"] not in profiles:
        raise ValueError("需要有效的当前股票池，最多保存30个")
    allowed = {g["id"] for g in catalog["groups"]}
    cleaned = {"version": 1, "active": document["active"], "profiles": {}}
    for ident, profile in profiles.items():
        if not isinstance(ident, str) or not PROFILE_ID.fullmatch(ident) or not isinstance(profile, dict):
            raise ValueError("股票池ID无效")
        if set(profile) != {"name", "include_base", "groups", "include", "exclude", "filters"}:
            raise ValueError("股票池包含未知字段")
        if not isinstance(profile["name"], str) or not 1 <= len(profile["name"].strip()) <= 80 or type(profile["include_base"]) is not bool:
            raise ValueError("股票池名称或基础池设置无效")
        groups = profile["groups"]
        if not isinstance(groups, list) or any(not isinstance(g, str) for g in groups):
            raise ValueError("未知细分行业或主题")
        groups = [catalog.get("group_aliases", {}).get(g, g) for g in groups]
        if any(g not in allowed for g in groups):
            raise ValueError("未知细分行业或主题")
        filters = {**FILTERS, **profile["filters"]} if isinstance(profile["filters"], dict) else {}
        if set(filters) != set(FILTERS):
            raise ValueError("筛选参数无效")
        for key, value in filters.items():
            if key == "max_market_cap" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 1e16:
                raise ValueError("筛选门槛必须为非负有限数字")
        for key, low, high in (("volume_window", 1, 252), ("min_history", 61, 2520)):
            if type(filters[key]) is not int or not low <= filters[key] <= high:
                raise ValueError(f"{key} 超出允许范围")
        if filters["max_market_cap"] is not None and filters["max_market_cap"] <= filters["min_market_cap"]:
            raise ValueError("市值上限必须大于下限，或留空")
        cleaned["profiles"][ident] = {"name": profile["name"].strip(), "include_base": profile["include_base"],
                                      "groups": sorted(set(groups)), "include": symbols(profile["include"]),
                                      "exclude": symbols(profile["exclude"]), "filters": filters}
    return cleaned


def resolve(base, settings, catalog):
    settings = validate_settings(settings, catalog)
    profile = settings["profiles"][settings["active"]]
    chosen = set(base["symbols"]) if profile["include_base"] else set()
    for group in catalog["groups"]:
        if group["id"] in profile["groups"]:
            chosen.update(symbols(group["symbols"]))
    chosen.update(profile["include"])
    chosen.difference_update(profile["exclude"])
    chosen.discard(base["benchmark"])
    if not chosen or len(chosen) > 1500:
        raise ValueError("请选择至少一只、最多1500只股票")
    config = copy.deepcopy(base)
    config.update(symbols=sorted(chosen), current_screen=profile["filters"],
                  universe_profile=settings["active"], universe_name=profile["name"],
                  universe_selection=profile, screen_current_only=True)
    # A small industry pool must still produce scores. Cross-sectional IC needs
    # at least three stocks; per-stock timing diagnostics remain available.
    config["min_candidates"] = 3
    return config


def fingerprint(config, selected):
    payload = {"profile": config.get("universe_profile", "default"), "symbols": selected,
               "filters": config.get("current_screen", {})}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return "current-" + digest
