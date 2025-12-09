# freshness_predictor.py
# Simple wrapper around your Keras model + CLIP fallback.
# Public function: predict_freshness_from_pil(pil.Image) -> dict

import os, traceback
from PIL import Image
import numpy as np
import pickle

# -------------------------------------------------
# Make paths robust: point to models/ relative to project root
# -------------------------------------------------
# this file is in: src/objective/backend/freshness_predictor.py
THIS_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))

MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATHS = [
    os.path.join(MODELS_DIR, "fruit_model_retrained.h5"),
    os.path.join(MODELS_DIR, "fruit_model.h5"),
]

LABEL_ENCODER_PATHS = [
    os.path.join(MODELS_DIR, "label_encoder_v5.pkl"),
    os.path.join(MODELS_DIR, "label_encoder_v2.pkl"),
    os.path.join(MODELS_DIR, "label_encoder.pkl"),
]

IMG_SIZE = (150, 150)
FRESH_THRESHOLD = 0.62
NOT_FRESH_THRESHOLD = 0.38

# -------------------------------------------------
# Try to import Keras / TensorFlow
# -------------------------------------------------
KERAS_OK = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    KERAS_OK = True
except Exception:
    KERAS_OK = False

# -------------------------------------------------
# Try to import CLIP
# -------------------------------------------------
CLIP_OK = False
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_OK = True
except Exception:
    CLIP_OK = False

# ---------------- loaders ----------------
def load_any_keras_model(paths):
    if not KERAS_OK:
        print("[predictor] TensorFlow / Keras not available, skipping keras model load")
        return None
    for p in paths:
        if os.path.exists(p):
            try:
                m = load_model(p)
                print("[predictor] Loaded keras model:", p)
                return m
            except Exception as e:
                print("[predictor] Failed to load keras model", p, ":", e)
        else:
            print("[predictor] Model path does not exist:", p)
    return None

def load_any_label_encoder(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    le = pickle.load(f)
                print("[predictor] Loaded label encoder:", p)
                return le
            except Exception as e:
                print("[predictor] Failed to load label encoder", p, ":", e)
        else:
            print("[predictor] Label encoder path does not exist:", p)
    return None

keras_model = load_any_keras_model(MODEL_PATHS)
label_encoder = load_any_label_encoder(LABEL_ENCODER_PATHS)

# CLIP caching
_clip_cache = None
def load_clip():
    global _clip_cache
    if _clip_cache is not None:
        return _clip_cache
    if not CLIP_OK:
        print("[predictor] CLIP not available (transformers/torch missing), skipping")
        return None
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    _clip_cache = {"model": model, "processor": processor, "device": device}
    print("[predictor] Loaded CLIP on device:", device)
    return _clip_cache

# ---------------- utilities ----------------
def multi_to_fresh_prob(pred_vector, label_encoder):
    if label_encoder is None:
        return None
    classes = list(label_encoder.classes_)
    if len(pred_vector) != len(classes):
        return None
    fresh_prob = 0.0
    for i, c in enumerate(classes):
        cname = c.lower()
        if cname.startswith("fresh") or "fresh" in cname:
            fresh_prob += float(pred_vector[i])
    return float(fresh_prob)

def clip_zero_shot(pil_img):
    clip = load_clip()
    if clip is None:
        return None, None
    fruits = [
        "apple", "banana", "cucumber", "tomato", "potato", "orange",
        "grape", "capsicum", "okra", "bittergourd", "carrot", "pear", "mango"
    ]
    prompts, mapping = [], []
    for f in fruits:
        prompts.append(f"a photo of fresh {f}")
        mapping.append((f, "fresh"))
        prompts.append(f"a photo of rotten {f}")
        mapping.append((f, "rotten"))

    prompts += ["a photo of fresh fruit", "a photo of rotten fruit"]
    for gp in ["generic_fresh", "generic_rotten"]:
        mapping.append((gp, gp))

    proc = clip["processor"]
    model = clip["model"]
    device = clip["device"]

    inputs = proc(text=prompts, images=pil_img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        sims = (100.0 * img_emb @ txt_emb.T).cpu().numpy().flatten()
    best = int(np.argmax(sims))
    best_score = float(sims[best])
    fruit, lab = mapping[best]
    prob = 1.0 / (1.0 + np.exp(-best_score / 9.0))
    if "rotten" in prompts[best]:
        prob = 1.0 - prob
    return float(prob), fruit

# ---------------- public function ----------------
def predict_freshness_from_pil(pil_img):
    """
    Input:  PIL.Image
    Output: dict {label, confidence (0-100), source, debug}
    """
    try:
        img = pil_img.convert("RGB").resize(IMG_SIZE)
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, 0)

        final_prob = None
        source = None
        debug = {}

        # 1) KERAS MODEL
        if keras_model is not None:
            try:
                pred = keras_model.predict(arr)
                pred = np.array(pred)
                debug["keras_shape"] = pred.shape

                # binary output
                if pred.ndim == 1 or (pred.ndim == 2 and pred.shape[-1] == 1):
                    p = float(pred.reshape(-1)[0])
                    final_prob = p
                    source = "keras_sigmoid"
                    debug["keras_sig"] = p
                else:
                    # multi-class – map to fresh prob
                    pred_vec = pred.reshape(-1)
                    mp = multi_to_fresh_prob(pred_vec, label_encoder)
                    if mp is not None:
                        final_prob = mp
                        source = "keras_multiclass_mapped"
                        try:
                            pred_idx = int(np.argmax(pred_vec))
                            predicted_name = label_encoder.inverse_transform([pred_idx])[0]
                            debug["keras_pred_label"] = predicted_name
                        except Exception:
                            pass
                    else:
                        # cannot compute fresh prob; keep label only
                        try:
                            pred_idx = int(np.argmax(pred_vec))
                            predicted_name = label_encoder.inverse_transform([pred_idx])[0]
                            debug["keras_pred_label"] = predicted_name
                        except Exception:
                            pass
            except Exception as e:
                debug["keras_error"] = str(e)

        # 2) CLIP FALLBACK / COMBINATION
        if final_prob is None or final_prob < 0.55:
            try:
                clip_prob, fruit_guess = clip_zero_shot(pil_img)
                if clip_prob is not None:
                    if final_prob is None:
                        final_prob = clip_prob
                        source = (source or "") + "clip"
                    else:
                        final_prob = max(final_prob, clip_prob)
                        source = (source or "") + "+clip"
                    debug["clip"] = (clip_prob, fruit_guess)
            except Exception as e:
                debug["clip_error"] = str(e)

        # 3) FINAL DECISION
        if final_prob is None:
            # nothing worked
            return {
                "label": "Unsure",
                "confidence": 0.0,
                "source": "none",
                "debug": debug,
            }

        conf_percent = round(float(final_prob * 100), 2)
        if final_prob >= FRESH_THRESHOLD:
            label = "OK to eat"
        elif final_prob <= NOT_FRESH_THRESHOLD:
            label = "Not OK"
        else:
            label = "Unsure"

        return {
            "label": label,
            "confidence": conf_percent,
            "source": source,
            "debug": debug,
        }

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "label": "Unsure",
            "confidence": 0.0,
            "source": "error",
            "debug": {"error": str(e), "trace": tb},
        }
