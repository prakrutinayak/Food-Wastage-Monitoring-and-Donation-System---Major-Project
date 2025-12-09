# src/objective/frontend/app.py

import os
import sys
import time

import streamlit as st
from PIL import Image
import pandas as pd

# ---- add project root to Python path ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.objective.backend.freshness_predictor import predict_freshness_from_pil
from src.objective.backend.food_aid_logic import (
    load_ngo_data,
    get_supermarket_data,
    get_initial_demands,
    get_demands_for_supermarket,
)

UPLOAD_DIR = "collected_uncertain"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="Food Spoilage Detection & Demand Forecasting",
    page_icon="🍎",
    layout="wide",
)


def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# ---------- SESSION STATE ----------
if "ngo_df" not in st.session_state:
    st.session_state.ngo_df = load_ngo_data()

if "supermarkets" not in st.session_state:
    st.session_state.supermarkets = get_supermarket_data(st.session_state.ngo_df)

if "demands" not in st.session_state:
    st.session_state.demands = get_initial_demands() or []

if st.session_state.demands is None:
    st.session_state.demands = []

if "next_demand_id" not in st.session_state:
    ids = [d.get("id", 0) for d in st.session_state.demands]
    st.session_state.next_demand_id = max(ids) + 1 if ids else 1


def allocate_new_demand_id():
    nid = st.session_state.next_demand_id
    st.session_state.next_demand_id += 1
    return nid


# ---------- UI HELPERS ----------
def map_overview_ui():
    st.header("🌍 Map Overview")

    ngo_df = st.session_state.ngo_df
    if ngo_df is None or ngo_df.empty:
        st.warning("NGO data not loaded. Check datasets/food_banks.csv.")
        return

    ngo_map = ngo_df[["latitude", "longitude", "name"]].rename(
        columns={"latitude": "lat", "longitude": "lon"}
    )

    sm_list = [v for _, v in st.session_state.supermarkets.items()]
    if sm_list:
        sm_df = pd.DataFrame(sm_list).rename(columns={"lat": "lat", "lon": "lon"})
    else:
        sm_df = pd.DataFrame(columns=["lat", "lon", "name"])

    map_df = pd.concat([ngo_map, sm_df], ignore_index=True)

    if map_df.empty:
        st.info("No locations to show on map.")
    else:
        st.map(map_df, latitude="lat", longitude="lon", zoom=12)

    st.markdown("---")


def ngo_interface_ui():
    ngo_df = st.session_state.ngo_df
    if ngo_df is None or ngo_df.empty:
        st.warning("NGO data not loaded.")
        return

    st.title("🤝 NGO — Submit Demands & View Donations")

    ngo_options = ngo_df.set_index("id")["name"].to_dict()
    default_ngo = st.session_state.get("selected_ngo_id", list(ngo_options.keys())[0])

    selected_ngo_id = st.selectbox(
        "Select your NGO:",
        list(ngo_options.keys()),
        index=list(ngo_options.keys()).index(default_ngo),
        format_func=lambda x: ngo_options[x],
    )
    st.session_state["selected_ngo_id"] = selected_ngo_id
    ngo_name = ngo_options[selected_ngo_id]
    st.subheader(f"Logged in as: *{ngo_name}*")

    # ----- create new demand -----
    with st.form("new_demand_form", clear_on_submit=True):
        st.write("### Create New Demand")
        item_name = st.text_input("Item Name (e.g. Apples, Bananas, Tomatoes, Bread, etc.)")
        quantity_needed = st.number_input(
            "Quantity Needed", min_value=1.0, value=10.0, step=1.0
        )
        unit = st.selectbox(
            "Unit", ["kg", "boxes", "crates", "liters", "packets"], index=0
        )
        submitted = st.form_submit_button("Submit Demand")

        if submitted:
            new_id = allocate_new_demand_id()
            new_demand = {
                "id": new_id,
                "ngo_id": int(selected_ngo_id),
                "item_name": item_name or "Unknown item",
                "quantity_needed": float(quantity_needed),
                "unit": unit,
                "status": "Pending Supermarket Review",
                "supermarket_response": None,
                "image_path": None,
                "freshness_label": None,
                "freshness_confidence": None,
                "freshness_source": None,
                "prediction_label": None,
                "prediction_confidence": None,
                "prediction_source": None,
            }
            st.session_state.demands.append(new_demand)
            st.success(f"Demand submitted (ID #{new_id}) for {ngo_name}")
            safe_rerun()

    st.markdown("---")
    st.subheader(f"Current Demands for {ngo_name}")

    demands_for_ngo = [
        d for d in st.session_state.demands if d.get("ngo_id") == int(selected_ngo_id)
    ]

    if not demands_for_ngo:
        st.info("No active demands for this NGO.")
        return

    for d in sorted(demands_for_ngo, key=lambda x: x.get("id", 0)):
        st.markdown(
            f"**Demand #{d['id']} — {d['item_name']}** — "
            f"{d['quantity_needed']} {d['unit']} — "
            f"Status: **{d['status']}**"
        )

        if d.get("supermarket_response"):
            st.write(f"Response from: {d.get('supermarket_response')}")

        if d.get("image_path") and os.path.exists(d["image_path"]):
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(d["image_path"], width=140)

            with cols[1]:
                label = d.get("freshness_label") or d.get("prediction_label")
                conf = d.get("freshness_confidence") or d.get("prediction_confidence")
                source = d.get("freshness_source") or d.get("prediction_source")

                if label is None and conf is None:
                    st.write("**Prediction:** _Not predicted yet or model unsure_")
                    st.write("**Confidence:** —")
                    st.write("**Source:** —")
                else:
                    st.write(f"**Prediction:** {label if label is not None else 'Unsure'}")
                    st.write(
                        f"**Confidence:** {conf}%"
                        if conf is not None
                        else "**Confidence:** —"
                    )
                    st.write(f"**Source:** {source or 'model'}")

                if st.button(
                    f"Open full image #{d['id']}", key=f"open_img_{d['id']}"
                ):
                    st.image(
                        d["image_path"],
                        caption=f"Demand #{d['id']} image",
                        use_container_width=True,
                    )

        st.markdown("---")


def supermarket_interface_ui():
    ngo_df = st.session_state.ngo_df
    if ngo_df is None or ngo_df.empty:
        st.warning("NGO data not loaded.")
        return

    st.title("🛒 Supermarket — Manage Stock & Upload Photos")

    sm_keys = list(st.session_state.supermarkets.keys())
    default_sm = st.session_state.get("selected_supermarket_id", sm_keys[0])

    sm_id = st.selectbox(
        "Select Supermarket:",
        sm_keys,
        index=sm_keys.index(default_sm),
        format_func=lambda x: st.session_state.supermarkets[x]["name"],
    )
    st.session_state["selected_supermarket_id"] = sm_id
    sm_data = st.session_state.supermarkets[sm_id]

    st.subheader(f"Logged in as: *{sm_data['name']}*")
    st.markdown("---")

    st.subheader("Nearby / Pending NGO Demands")

    df_sm_view = get_demands_for_supermarket(
        sm_id, sm_data, st.session_state.demands, ngo_df
    )

    if df_sm_view.empty:
        st.info("No nearby pending demands. Showing global pending demands as fallback.")
        pending_global = [
            d
            for d in st.session_state.demands
            if d.get("status") == "Pending Supermarket Review"
        ]

        if pending_global:
            rows = []
            for d in pending_global:
                ngo_row = ngo_df[ngo_df["id"] == d.get("ngo_id")]
                ngo_name = ngo_row.iloc[0]["name"] if not ngo_row.empty else "Unknown NGO"
                rows.append(
                    {
                        "Demand ID": d["id"],
                        "NGO": ngo_name,
                        "Item": d["item_name"],
                        "Quantity": f"{d['quantity_needed']} {d['unit']}",
                        "Status": d["status"],
                    }
                )
            st.dataframe(pd.DataFrame(rows).set_index("Demand ID"))
        else:
            st.info(
                "No pending demands on platform. "
                "Ask an NGO to submit a demand first."
            )
    else:
        st.dataframe(df_sm_view)

    st.markdown("---")
    st.subheader("Update Demand / Upload Photo (camera or file)")

    if not df_sm_view.empty:
        pending_ids = df_sm_view[
            df_sm_view["Status"] == "Pending Supermarket Review"
        ].index.tolist()
    else:
        pending_ids = [
            d["id"]
            for d in st.session_state.demands
            if d.get("status") == "Pending Supermarket Review"
        ]

    selected_demand_id = None
    demand_item = None

    if pending_ids:
        selected_demand_id = st.selectbox("Select Demand to Update:", pending_ids)
        demand_item = next(
            (d for d in st.session_state.demands if d["id"] == int(selected_demand_id)),
            None,
        )
        if demand_item:
            ngo_row = ngo_df[ngo_df["id"] == demand_item.get("ngo_id")]
            ngo_name = ngo_row.iloc[0]["name"] if not ngo_row.empty else "Unknown NGO"
            st.write(
                f"Selected: **{demand_item['item_name']}** — "
                f"{demand_item['quantity_needed']} {demand_item['unit']} "
                f"for **{ngo_name}**"
            )
    else:
        st.info(
            "No pending demands selected. "
            "You may upload ad-hoc donation photos below (not linked to a demand)."
        )

    # ----- Image upload / camera input -----
    with st.expander("Upload / Capture photo of the item"):
        col_a, col_b = st.columns(2)
        uploaded = col_a.file_uploader(
            "Upload image (file)",
            type=["jpg", "jpeg", "png"],
            key=f"file_uploader_{sm_id}",
        )
        camera_img = col_b.camera_input("Take a photo (camera)", key=f"camera_{sm_id}")

        chosen_image = None
        if camera_img is not None:
            try:
                chosen_image = Image.open(camera_img).convert("RGB")
            except Exception:
                chosen_image = None
        elif uploaded is not None:
            try:
                chosen_image = Image.open(uploaded).convert("RGB")
            except Exception:
                chosen_image = None

        if chosen_image is not None:
            st.image(chosen_image, caption="Preview", width=300)

            if st.button(
                "Confirm: Upload & Mark Stock Available",
                key=f"confirm_upload_{sm_id}_{selected_demand_id}",
            ):
                try:
                    res = predict_freshness_from_pil(chosen_image)
                except Exception as e:
                    res = {
                        "label": "Unsure",
                        "confidence": 0.0,
                        "source": f"error:{e}",
                    }

                filename = (
                    f"sm{sm_id}_d{selected_demand_id or 'adhoc'}_"
                    f"{int(time.time()*1000)}.jpg"
                )
                save_path = os.path.join(UPLOAD_DIR, filename)
                chosen_image.save(save_path)

                if selected_demand_id is not None and demand_item is not None:
                    pred_label = res.get("label")
                    pred_conf = round(float(res.get("confidence", 0)), 2)
                    pred_source = res.get("source")

                    for d in st.session_state.demands:
                        if d["id"] == int(selected_demand_id):
                            d["status"] = "Stock Available"
                            d["supermarket_response"] = st.session_state.supermarkets[
                                sm_id
                            ]["name"]
                            d["image_path"] = save_path

                            d["freshness_label"] = pred_label
                            d["freshness_confidence"] = pred_conf
                            d["freshness_source"] = pred_source

                            d["prediction_label"] = pred_label
                            d["prediction_confidence"] = pred_conf
                            d["prediction_source"] = pred_source
                            break

                    st.success(f"Uploaded & predicted: {pred_label} ({pred_conf}%)")
                    safe_rerun()
                else:
                    pred_label = res.get("label")
                    pred_conf = round(float(res.get("confidence", 0)), 2)
                    pred_source = res.get("source")

                    new_id = allocate_new_demand_id()
                    record = {
                        "id": new_id,
                        "ngo_id": None,
                        "item_name": "Ad-hoc donation",
                        "quantity_needed": 0,
                        "unit": "",
                        "status": "Stock Available",
                        "supermarket_response": st.session_state.supermarkets[sm_id][
                            "name"
                        ],
                        "image_path": save_path,
                        "freshness_label": pred_label,
                        "freshness_confidence": pred_conf,
                        "freshness_source": pred_source,
                        "prediction_label": pred_label,
                        "prediction_confidence": pred_conf,
                        "prediction_source": pred_source,
                    }
                    st.session_state.demands.append(record)
                    st.success(f"Ad-hoc uploaded: {pred_label} ({pred_conf}%)")
                    safe_rerun()

    # ----- Quick status update -----
    if selected_demand_id is not None and demand_item is not None:
        col1, col2, col3 = st.columns(3)

        if col1.button(
            "✅ I Have It In Stock (no photo)",
            key=f"stock_no_photo_{selected_demand_id}",
        ):
            for d in st.session_state.demands:
                if d["id"] == int(selected_demand_id):
                    d["status"] = "Stock Available"
                    d["supermarket_response"] = st.session_state.supermarkets[sm_id][
                        "name"
                    ]
                    break
            st.success("Marked Stock Available (no photo).")
            safe_rerun()

        if col2.button(
            "❌ Out of Stock",
            key=f"out_no_photo_{selected_demand_id}",
        ):
            for d in st.session_state.demands:
                if d["id"] == int(selected_demand_id):
                    d["status"] = "Out of Stock"
                    d["supermarket_response"] = st.session_state.supermarkets[sm_id][
                        "name"
                    ]
                    break
            st.warning("Marked Out of Stock.")
            safe_rerun()

        with col3.expander("Partially Available"):
            default_qty = (
                demand_item.get("quantity_needed", 1.0) / 2
                if demand_item.get("quantity_needed", 1.0) > 1
                else 1.0
            )
            qty = st.number_input(
                "Quantity you can supply",
                min_value=1.0,
                value=default_qty,
                key=f"partial_{selected_demand_id}",
            )
            if st.button(
                "Confirm Partial Supply", key=f"partial_btn_{selected_demand_id}"
            ):
                for d in st.session_state.demands:
                    if d["id"] == int(selected_demand_id):
                        d["status"] = f"Partially Available ({qty} {d.get('unit', '')})"
                        d["supermarket_response"] = st.session_state.supermarkets[sm_id][
                            "name"
                        ]
                        break
                st.info("Marked Partially Available.")
                safe_rerun()


# ---------- SIDEBAR + MAIN ----------
with st.sidebar:
    st.markdown("## 🍱 Menu")
    subpage = st.radio(
        "Select view",
        ["🌍 Map Overview", "🤝 NGO Interface", "🛒 Supermarket Interface"],
        index=0,
    )

st.markdown(
    "<h1 style='margin-top:10px;'>Food Wastage Monitoring and Donation System</h1>",
    unsafe_allow_html=True,
)
st.write("")

if subpage == "🌍 Map Overview":
    map_overview_ui()
elif subpage == "🤝 NGO Interface":
    ngo_interface_ui()
else:
    supermarket_interface_ui()
