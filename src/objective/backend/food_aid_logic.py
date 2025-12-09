# src/objective/backend/food_aid_logic.py
# Small clean logic module for NGO list, supermarket simulation and demands

import os
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# point to your CSV inside datasets/
DEFAULT_NGO_FILE = os.path.join("datasets", "food_banks.csv")
DEFAULT_RADIUS_KM = 100000.0  # ~whole Earth


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def load_ngo_data(file_name: str = DEFAULT_NGO_FILE):
    """
    Load NGO CSV/Excel with columns:
    name, latitude, longitude, address

    We auto-detect common column names and clean the dataframe.
    """
    alt = os.path.join("datasets", "food_banks (1).csv")

    if not os.path.exists(file_name):
        if os.path.exists(alt):
            file_name = alt
        else:
            print(f"[food_aid_logic] NGO file not found: {file_name} / {alt}")
            return pd.DataFrame()

    if file_name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_name)
    else:
        df = pd.read_csv(file_name)

    # try to map column names automatically
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if "lat" in lc and "long" not in lc:
            col_map[c] = "latitude"
        elif "lon" in lc or "long" in lc:
            col_map[c] = "longitude"
        elif "name" in lc:
            col_map[c] = "name"
        elif "address" in lc:
            col_map[c] = "address"

    df = df.rename(columns=col_map)

    # ensure required columns exist
    for col in ["name", "latitude", "longitude", "address"]:
        if col not in df.columns:
            df[col] = None

    df = df[["name", "latitude", "longitude", "address"]].dropna(
        subset=["latitude", "longitude"]
    )
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # add simple numeric NGO id
    df["id"] = df.index + 1
    return df


def get_supermarket_data(ngo_df: pd.DataFrame):
    """Create 3 demo supermarkets roughly around the average NGO location."""
    if ngo_df is not None and not ngo_df.empty:
        avg_lat = float(ngo_df["latitude"].mean())
        avg_lon = float(ngo_df["longitude"].mean())
    else:
        # Bangalore-ish default
        avg_lat, avg_lon = 12.95, 77.65

    supermarkets = {
        101: {"name": "SuperMart Central", "lat": avg_lat + 0.005, "lon": avg_lon - 0.005},
        102: {"name": "Fresh Market East", "lat": avg_lat + 0.015, "lon": avg_lon + 0.010},
        103: {"name": "Big Bazaar South", "lat": avg_lat - 0.008, "lon": avg_lon - 0.002},
    }
    return supermarkets


def get_initial_demands():
    """
    Start platform with NO default demo demands.
    NGOs will add their own.
    """
    return []


def calculate_metrics(demands):
    """Optional aggregate metrics about demands (not heavily used now)."""
    if not demands:
        return {}

    df_demands = pd.DataFrame(demands)
    total_demands = len(df_demands)

    status_counts = df_demands["status"].value_counts()
    fulfilled_available = status_counts.get("Stock Available", 0)
    partial_available = sum(
        1
        for s in df_demands["status"]
        if isinstance(s, str) and "Partially Available" in s
    )
    fulfilled_count = fulfilled_available + partial_available
    pending_count = status_counts.get("Pending Supermarket Review", 0)
    out_of_stock_count = status_counts.get("Out of Stock", 0)
    fulfillment_rate = (fulfilled_count / total_demands) * 100 if total_demands > 0 else 0

    top_items = (
        df_demands[df_demands["item_name"].astype(bool)]["item_name"]
        .value_counts()
        .nlargest(5)
    )
    df_top = top_items.reset_index()
    df_top.columns = ["Item", "Requests"]

    return {
        "total_demands": total_demands,
        "fulfilled_count": fulfilled_count,
        "pending_count": pending_count,
        "out_of_stock_count": out_of_stock_count,
        "fulfillment_rate": fulfillment_rate,
        "top_items_df": df_top,
    }


def get_demands_for_supermarket(sm_id, sm_data, demands, ngo_df, radius_km=DEFAULT_RADIUS_KM):
    """Filter demands to those within radius_km of a supermarket.

    NOTE: Demands that the NGO has already ACCEPTED are skipped so
    they no longer appear in the supermarket update list.
    """
    if ngo_df is None or ngo_df.empty:
        return pd.DataFrame()

    ngo_map = (
        ngo_df[["id", "latitude", "longitude", "name"]]
        .set_index("id")
        .T
        .to_dict("dict")
    )

    out = []
    for d in demands:
        # NEW: skip NGO-accepted demands
        if d.get("ngo_response") == "Accepted":
            continue

        ngo_id = d.get("ngo_id")
        if not ngo_id:
            continue
        if ngo_id not in ngo_map:
            continue

        ngo = ngo_map[ngo_id]
        dist = calculate_distance(
            sm_data["lat"],
            sm_data["lon"],
            float(ngo["latitude"]),
            float(ngo["longitude"]),
        )
        if dist <= radius_km:
            out.append(
                {
                    "Demand ID": d["id"],
                    "NGO": ngo["name"],
                    "Distance (km)": f"{dist:.2f}",
                    "Item": d["item_name"],
                    "Quantity": f"{d['quantity_needed']} {d['unit']}",
                    "Status": d["status"],
                }
            )

    if out:
        return pd.DataFrame(out).set_index("Demand ID")
    return pd.DataFrame()



def create_new_demand(current_demands, selected_ngo_id, item_name, quantity_needed, unit):
    """Kept for reference (not used directly now)."""
    return {
        "id": len(current_demands) + 1,
        "ngo_id": int(selected_ngo_id),
        "item_name": str(item_name),
        "quantity_needed": float(quantity_needed),
        "unit": unit,
        "status": "Pending Supermarket Review",
        "supermarket_response": None,
    }
