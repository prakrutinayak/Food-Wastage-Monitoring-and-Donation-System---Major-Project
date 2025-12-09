# merged_app.py
# Unified UI:
# 1) Food Spoilage Detection & Demand Forecasting
# 2) Donation Recommendation
# 3) Recipe Generation

import os
import sys
import traceback
import io
import base64

import pandas as pd
from PIL import Image
import streamlit as st

# -------------------------------------------------------------------
# PATH SETUP  (project root: food_donation_system)
# -------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(BASE_DIR)

# -------------------------------------------------------------------
# BACKEND IMPORTS
# -------------------------------------------------------------------
from src.objective.backend.food_aid_logic import (
    load_ngo_data,
    get_supermarket_data,
    get_initial_demands,
    calculate_metrics,
    get_demands_for_supermarket,
)
from src.objective.backend.freshness_predictor import predict_freshness_from_pil

from src.objective.backend.backend import (
    load_dataset,
    find_nearby,
    geocode_address,
    geolocate_device,
    google_search_link,
    directions_distance_km,
)
from src.objective.backend.recipe_generator import generate_recipes

from config import API_KEY


# -------------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Food Wastage Monitoring & Donation System",
    page_icon="🍱",
    layout="wide",
)

# -------------------------------------------------------------------
# GLOBAL DATA (NGO CSV)
# -------------------------------------------------------------------
NGO_CSV_PATH = os.path.join(BASE_DIR, "datasets", "food_banks.csv")


# ===================================================================
#  MODULE 1: FOOD SPOILAGE & DEMAND FORECASTING
# ===================================================================

def init_food_module_state():
    """Initialise session_state for module 1."""
    if "ngo_df" not in st.session_state:
        st.session_state.ngo_df = load_ngo_data(NGO_CSV_PATH)

    if "supermarkets" not in st.session_state:
        st.session_state.supermarkets = get_supermarket_data(st.session_state.ngo_df)

    if "demands" not in st.session_state:
        st.session_state.demands = get_initial_demands()

    # ensure new field exists on old demands
    for d in st.session_state.demands:
        if "ngo_response" not in d:
            d["ngo_response"] = None

    if "next_demand_id" not in st.session_state:
        if st.session_state.demands:
            st.session_state.next_demand_id = (
                max(d["id"] for d in st.session_state.demands) + 1
            )
        else:
            st.session_state.next_demand_id = 1


def food_map_overview_ui():
    ngo_df = st.session_state.ngo_df

    st.subheader("🌍 Map Overview")

    if ngo_df is None or ngo_df.empty:
        st.warning("NGO data not loaded. Check datasets/food_banks.csv.")
        return

    st.markdown("**Food Banks / NGOs from the dataset**")
    st.dataframe(ngo_df[["id", "name", "address"]])

    st.markdown("**Location map**")
    try:
        st.map(ngo_df[["latitude", "longitude"]])
    except Exception as e:
        st.error(f"Could not display map: {e}")


def ngo_interface_ui():
    ngo_df = st.session_state.ngo_df

    st.subheader("🏥 NGO Interface")

    if ngo_df is None or ngo_df.empty:
        st.warning("NGO data not loaded. Check datasets/food_banks.csv.")
        return

    demands = st.session_state.demands

    # --- Select NGO ---
    st.markdown("### Select NGO")
    ngo_options = ngo_df[["id", "name"]].copy()
    ngo_options["label"] = ngo_options["id"].astype(str) + " – " + ngo_options["name"]

    selected_label = st.selectbox(
        "Choose your NGO",
        ngo_options["label"].tolist(),
    )
    selected_id = int(selected_label.split("–")[0].strip())
    selected_ngo = ngo_df[ngo_df["id"] == selected_id].iloc[0]

    st.info(f"Selected NGO: **{selected_ngo['name']}**")

    # --- Create new demand ---
    st.markdown("### Create new demand")

    col1, col2, col3 = st.columns(3)
    with col1:
        item_name = st.text_input("Item name", placeholder="tomato, banana, rice, etc.")
    with col2:
        quantity = st.number_input("Quantity", min_value=0.0, value=10.0, step=1.0)
    with col3:
        unit = st.selectbox("Unit", ["kg", "packets", "boxes", "pieces"])

    if st.button("Submit Demand", type="primary"):
        if not item_name.strip():
            st.error("Please enter an item name.")
        else:
            demand = {
                "id": st.session_state.next_demand_id,
                "ngo_id": selected_id,
                "item_name": item_name.strip(),
                "quantity_needed": quantity,
                "unit": unit,
                "status": "Pending Supermarket Review",
                "predicted_label": None,
                "confidence": None,
                "source": None,
                "supermarket_response": None,
                "image_bytes": None,   # filled later by supermarket
                "ngo_response": None,  # NEW: NGO accepts / rejects
            }
            st.session_state.demands.append(demand)
            st.session_state.next_demand_id += 1
            st.success("Demand submitted successfully!")

    # --- Show current demands for this NGO ---
    st.markdown("### Current Demands for this NGO")
    ngo_demands = [d for d in st.session_state.demands if d["ngo_id"] == selected_id]

    if not ngo_demands:
        st.info("No demands yet for this NGO.")
        return

    df_display = pd.DataFrame()

    for d in ngo_demands:
        if "ngo_response" not in d:
            d["ngo_response"] = None

        img_html = ""
        if d.get("image_bytes"):
            img_html = (
                f"<img src='data:image/png;base64,{d['image_bytes']}' "
                f"width='60' style='border-radius:8px;'>"
            )

        resp_text = d["ngo_response"] if d["ngo_response"] else "Pending"

        row = {
            "Demand ID": d["id"],
            "Item": d["item_name"],
            "Quantity": f"{d['quantity_needed']} {d['unit']}",
            "Status": d["status"],
            "Predicted": d.get("predicted_label", ""),
            "Confidence": (
                f"{d.get('confidence', 0):.2f}"
                if d.get("confidence") is not None
                else ""
            ),
            "Response": resp_text,
            "Image": img_html,
        }

        df_display = pd.concat(
            [df_display, pd.DataFrame([row])],
            ignore_index=True,
        )

    # render table with images + response text
    st.write(
        df_display.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

    # ---- NGO Accept / Reject panel ----
    st.markdown("#### Respond to supermarket stock")

    # only demands where supermarket has updated status
    respondable = [
        d
        for d in ngo_demands
        if d["status"] != "Pending Supermarket Review"
        and d.get("ngo_response") is None
    ]

    if not respondable:
        st.info("No new supermarket updates waiting for NGO response.")
        return

    resp_ids = [d["id"] for d in respondable]
    chosen_id = st.selectbox("Choose Demand ID to respond:", resp_ids)

    choice = st.radio(
        "Your decision:",
        ["Accept", "Reject"],
        horizontal=True,
    )

    if st.button("Submit NGO response", type="primary"):
        for d in st.session_state.demands:
            if d["id"] == chosen_id:
                d["ngo_response"] = "Accepted" if choice == "Accept" else "Rejected"
                break
        st.success(f"Response saved for Demand #{chosen_id}: {choice}")


def supermarket_interface_ui():
    ngo_df = st.session_state.ngo_df
    supermarkets = st.session_state.supermarkets
    demands = st.session_state.demands

    st.subheader("🏬 Supermarket Interface")

    if ngo_df is None or ngo_df.empty:
        st.warning("NGO data not loaded. Check datasets/food_banks.csv.")
        return

    if not supermarkets:
        st.warning("Supermarket list is empty.")
        return

    # ---- select supermarket ----
    st.markdown("### Select supermarket")
    sm_ids = list(supermarkets.keys())
    sm_labels = [f"{sid} – {supermarkets[sid]['name']}" for sid in sm_ids]
    selected_sm_label = st.selectbox("Supermarket", sm_labels)
    selected_sm_id = int(selected_sm_label.split("–")[0].strip())
    sm = supermarkets[selected_sm_id]

    st.info(f"Selected supermarket: **{sm['name']}**")

    # ---- Metrics ----
    st.markdown("### Overall Demand Metrics")
    metrics = calculate_metrics(demands)
    if not metrics:
        st.info("No demands yet in the system.")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Demands", metrics["total_demands"])
        col2.metric("Fulfilled / Available", metrics["fulfilled_count"])
        col3.metric("Pending Review", metrics["pending_count"])
        col4.metric("Out of Stock", metrics["out_of_stock_count"])
        col5.metric("Fulfillment Rate", f"{metrics['fulfillment_rate']:.1f}%")

        if not metrics["top_items_df"].empty:
            st.markdown("**Top requested items**")
            st.dataframe(metrics["top_items_df"], use_container_width=True)

    # ---- Nearby pending demands (excluding NGO-accepted ones in backend) ----
    st.markdown("### Nearby / Pending NGO Demands ↩️")
    nearby_df = get_demands_for_supermarket(
        selected_sm_id,
        sm,
        demands,
        ngo_df,
        radius_km=30.0,
    )

    if nearby_df.empty:
        st.info("No nearby demands found within the radius.")
        return

    st.dataframe(nearby_df, use_container_width=True)

    # ----------------------------------------------------------------
    # Select a demand to update + OPTIONAL photo (file or camera)
    # ----------------------------------------------------------------
    st.markdown("### Update Demand / Upload Photo (camera or file) ↩️")

    demand_ids = nearby_df.index.tolist()
    selected_demand_id = st.selectbox("Select Demand to Update:", demand_ids)

    selected_row = nearby_df.loc[selected_demand_id]
    st.write(
        f"Selected: **{selected_row['Item']}** — {selected_row['Quantity']} "
        f"for **{selected_row['NGO']}**"
    )

    # ---- Manual stock status options (no 'use model prediction') ----
    status_choice = st.radio(
        "Set stock status for this demand:",
        ["I have it in stock", "Out of stock", "Partially available"],
        index=0,
        horizontal=True,
    )

    image_file = None

    # Only show camera / uploader when NOT "Out of stock"
    if status_choice != "Out of stock":
        source_choice = st.radio(
            "Choose image source:",
            ["Upload from file", "Use live camera"],
            index=0,
            horizontal=True,
        )

        if source_choice == "Upload from file":
            uploaded_file = st.file_uploader(
                "Upload / Capture photo of the item",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
                key="supermarket_upload",
            )
            image_file = uploaded_file
        else:
            camera_image = st.camera_input(
                "Capture photo from camera",
                key="supermarket_camera",
            )
            image_file = camera_image
    else:
        st.info("For **Out of stock**, no photo is required. Just click update.")

    run_update = st.button("Run Freshness Prediction & Update", type="primary")

    # ----------------------------------------------------------------
    # CASE 1: Out of stock  → no image, no prediction needed
    # ----------------------------------------------------------------
    if status_choice == "Out of stock" and run_update:
        try:
            demands_list = st.session_state.demands
            selected_index = next(
                (i for i, d in enumerate(demands_list) if d["id"] == selected_demand_id),
                None,
            )

            if selected_index is None:
                st.error("Could not find this demand in session_state.demands.")
                return

            d = demands_list[selected_index]
            d["status"] = "Out of Stock"
            d["predicted_label"] = "Not OK"
            d["confidence"] = None
            d["source"] = "manual_out_of_stock"
            # keep existing image_bytes
            st.session_state.demands = demands_list
            st.success("Demand updated to **Out of Stock** successfully!")
        except Exception:
            st.error("Error while updating demand.")
            st.write(traceback.format_exc())
        return

    # ----------------------------------------------------------------
    # CASE 2: In stock / Partially available  → need image + prediction
    # ----------------------------------------------------------------
    if status_choice != "Out of stock" and run_update:
        if not image_file:
            st.error("Please upload or capture an image first.")
            return

        try:
            image = Image.open(image_file)
            st.image(image, caption="Item image", width=220)

            # default prediction values
            label = "Unsure"
            confidence = 0.0
            source = "not_run"

            # try model prediction
            try:
                result = predict_freshness_from_pil(image)
                label = result.get("label", "Unsure")
                confidence = result.get("confidence", 0.0)
                source = result.get("source", "model")
            except Exception:
                st.warning("Model prediction failed, using default 'Unsure'.")
                source = "error"

            # show result
            if label == "OK to eat":
                st.success(f"Prediction: **OK to eat**  (confidence: {confidence:.2f}%)")
            elif label == "Not OK":
                st.error(f"Prediction: **Not OK**  (confidence: {confidence:.2f}%)")
            else:
                st.warning(
                    f"Prediction: **Unsure**  (confidence: {confidence:.2f}%)"
                )

            st.caption(f"Source: `{source}`")

            status_map = {
                "I have it in stock": "Stock Available",
                "Partially available": "Partially Available",
            }
            final_status = status_map.get(status_choice, "Pending Supermarket Review")

            demands_list = st.session_state.demands
            selected_index = next(
                (i for i, d in enumerate(demands_list) if d["id"] == selected_demand_id),
                None,
            )

            if selected_index is None:
                st.error("Could not find this demand in session_state.demands.")
                return

            d = demands_list[selected_index]
            d["status"] = final_status
            d["predicted_label"] = label
            d["confidence"] = confidence
            d["source"] = source

            # store image bytes for NGO thumbnail
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            d["image_bytes"] = base64.b64encode(buf.getvalue()).decode("utf-8")

            st.session_state.demands = demands_list
            st.success(f"Demand updated successfully! (Status: {final_status})")

        except Exception:
            st.error("Error during prediction or update.")
            st.write(traceback.format_exc())


def food_module_ui(sub_page: str):
    init_food_module_state()

    st.markdown(
        "<h1 style='margin-top:10px;'>Food Wastage Monitoring and Donation System</h1>",
        unsafe_allow_html=True,
    )

    if sub_page == "🌍 Map Overview":
        food_map_overview_ui()
    elif sub_page == "🏥 NGO Interface":
        ngo_interface_ui()
    else:
        supermarket_interface_ui()


# ===================================================================
#  MODULE 2: DONATION RECOMMENDATION
# ===================================================================

@st.cache_data(ttl=600)
def load_food_bank_data(path: str):
    return load_dataset(path)


def donation_ui():
    st.markdown("## 🙏 Donation Recommendation")
    st.caption(
        "Find the nearest food banks / NGOs for supermarkets, restaurants "
        "and stores to donate excess food."
    )
    st.write("---")

    DATASET_PATH = os.path.join(BASE_DIR, "datasets", "food_banks.csv")

    try:
        data = load_food_bank_data(DATASET_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    st.subheader("📍 Location")
    use_current = st.checkbox("Use My Current Location (auto-detect)")

    user_lat = user_lng = None

    if use_current:
        try:
            with st.spinner("Detecting your location..."):
                lat, lng, acc = geolocate_device(API_KEY)
                user_lat, user_lng = lat, lng
                st.success(f"✅ Auto-detected location (accuracy ~ {acc} meters).")
                st.info(f"Coordinates: {user_lat:.6f}, {user_lng:.6f}")
        except Exception as e:
            st.error("Auto-detection failed: " + str(e))

    if not user_lat:
        address = st.text_input(
            "Or enter your address:",
            placeholder="Example: Yelahanka, Bengaluru, Karnataka",
        )
        if address:
            try:
                with st.spinner("Geocoding address..."):
                    lat, lng, formatted = geocode_address(address, API_KEY)
                    user_lat, user_lng = lat, lng
                    st.success(f"✅ Location found: {formatted}")
                    st.info(f"Coordinates: {user_lat:.6f}, {user_lng:.6f}")
            except ValueError as ve:
                if str(ve) == "ZERO_RESULTS":
                    st.warning(
                        "Address not found. Try simplifying (area + city + state)."
                    )
                else:
                    st.error("Geocoding failed: " + str(ve))
            except Exception as ex:
                st.error("Geocoding error: " + str(ex))

    if user_lat and user_lng:
        st.info("⏳ Calculating road distances to all NGOs... please wait.")
        try:
            nearby = find_nearby(data, user_lat, user_lng, API_KEY, radius_km=5.0)

            if not nearby.empty:
                nearby = nearby.reset_index(drop=True)
                TOPN = min(5, len(nearby))
                for i in range(TOPN):
                    row = nearby.loc[i]
                    precise = directions_distance_km(
                        user_lat, user_lng, row["latitude"], row["longitude"], API_KEY
                    )
                    if precise is not None:
                        nearby.at[i, "distance_km"] = precise

            if nearby.empty:
                st.warning("No nearby food banks within 5 km.")
                st.info(
                    "You can also search for composting centers, animal shelters or "
                    "community kitchens in your area."
                )
                st.subheader("♻️ Composting Centers")
                st.write(google_search_link("composting centers", user_lat, user_lng))
                st.subheader("🐄 Animal Shelters")
                st.write(google_search_link("animal shelters", user_lat, user_lng))
                st.subheader("🏡 Community Kitchens")
                st.write(google_search_link("community kitchens", user_lat, user_lng))
            else:
                st.success(f"Found {len(nearby)} nearby NGOs within 5 km.")
                st.subheader("🏠 Nearest Food Banks / NGOs")
                st.dataframe(
                    nearby[["name", "address", "phone", "website", "distance_km"]]
                )

                st.subheader("🗺️ Map")
                st.map(nearby[["latitude", "longitude"]])

                st.subheader("🔗 Quick Links")
                for _, row in nearby.iterrows():
                    lat = row["latitude"]
                    lng = row["longitude"]
                    name = row["name"]
                    website = row.get("website", "")

                    if isinstance(website, str) and website.strip():
                        if not website.lower().startswith(("http://", "https://")):
                            website = "https://" + website.strip()
                    else:
                        website = ""

                    st.markdown(
                        f"**{name}**  \n"
                        f"🗺️ [Open in Google Maps]"
                        f"(https://www.google.com/maps/search/?api=1&query={lat},{lng})  \n"
                        f"🚗 [Get Directions]"
                        f"(https://www.google.com/maps/dir/?api=1&destination={lat},{lng})"
                        + (f"  \n🌐 [Website]({website})" if website else "")
                        + f"  \n**Distance:** {row['distance_km']} km"
                    )

        except Exception:
            st.error("Error while computing distances.")
            st.write(traceback.format_exc())


# ===================================================================
#  MODULE 3: RECIPE GENERATION
# ===================================================================

def recipe_ui():
    st.markdown("## 🍽 Recipe Generation")
    st.caption("Generate creative recipes from leftover ingredients.")
    st.write("---")

    ingredients = st.text_area(
        "Enter ingredients:",
        placeholder="Example: rice, tomato, onion, capsicum",
        height=120,
    )

    if st.button("Generate Recipes"):
        if not ingredients.strip():
            st.error("Please enter at least one ingredient.")
            return

        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            st.error("Missing GROQ_API_KEY in .streamlit/secrets.toml")
            return

        with st.spinner("Cooking some ideas..."):
            try:
                result = generate_recipes(ingredients, api_key)
                st.markdown("### 📝 Suggested Recipes")
                st.write(result)
            except Exception as e:
                st.error(f"Error generating recipes: {e}")


# ===================================================================
#  SIDEBAR NAVIGATION
# ===================================================================
with st.sidebar:
    st.markdown("## 🍱 Menu")

    main_page = st.radio(
        "Select module",
        [
            "🍎 Food Spoilage Detection & Demand Forecasting",
            "🙏 Donation Recommendation",
            "🍽 Recipe Generation",
        ],
    )

    sub_page = None
    if main_page.startswith("🍎"):
        st.write("Sub-module")
        sub_page = st.radio(
            "",
            ["🌍 Map Overview", "🏥 NGO Interface", "🏬 Supermarket Interface"],
        )

    st.write("---")
    if st.button("🏠 Home", use_container_width=True):
        st.experimental_rerun()


# ===================================================================
#  MAIN ROUTER
# ===================================================================
if main_page.startswith("🍎"):
    food_module_ui(sub_page)
elif main_page.startswith("🙏"):
    donation_ui()
else:
    recipe_ui()
