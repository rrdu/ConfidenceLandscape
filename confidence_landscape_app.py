#streamlit_app.py

'''Streamlit App Interface'''

import os
import io
import base64
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from torchvision.models import (
    ResNet18_Weights,
    MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights,
)

import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_image_select import image_select

from PIL import Image

from config import(
    MODELS,
    IMAGE_CLASSES,
    IMAGE_ROOT,
    PERTURB_AXES
)

from models.model_map import get_model
from utils.cache import load_from_cache, save_to_cache
from utils.landscape import build_confidence_landscape
from utils.plotting import make_plot
from utils.perturbations import PERTURB_FUNCS
from utils.gradcam import GradCAM
from utils.heatmap_overlay import gradcam_overlay_cv2
from utils.heatmap_overlay import colormap_legend
from utils.heatmap_overlay import apply_colormap_to_cam

# ---------- ImageNet labels for readable top-3 ----------
IMAGENET_LABELS = None
try:
    from torchvision.models import ResNet18_Weights
    IMAGENET_LABELS = ResNet18_Weights.DEFAULT.meta["categories"]
except Exception:
    # fallback: leave as None
    IMAGENET_LABELS = None
#############################################################
if "selected_image" not in st.session_state:
    st.session_state['selected_image'] = None
if "selected_class" not in st.session_state:
    st.session_state["selected_class"] = None
if "last_click" not in st.session_state:
    st.session_state["last_click"] = None
if "Z" not in st.session_state:
    st.session_state["Z"] = None
if "model_obj" not in st.session_state:
    st.session_state["model_obj"] = None
if "class_idx" not in st.session_state:
    st.session_state["class_idx"] = None
if "active_image" not in st.session_state:
    st.session_state["active_image"] = None
if "active_class" not in st.session_state:
    st.session_state["active_class"] = None
if "active_model" not in st.session_state:
    st.session_state["active_model"] = None
if "active_x" not in st.session_state:
    st.session_state["active_x"] = None
if "active_y" not in st.session_state:
    st.session_state["active_y"] = None
if "pred_box_html" not in st.session_state:
    st.session_state["pred_box_html"] = None
#############################################################
#Initialize page
st.set_page_config(page_title='Confidence Landscape', layout='wide')
st.title('**Confidence Landscape**')

#############################################################
# ---------- one-time welcome message ----------
if "show_welcome" not in st.session_state:
    st.session_state["show_welcome"] = True

if st.session_state["show_welcome"]:
    st.info(
        """
        👋 **Welcome to the Confidence Landscape app**

        **How to use it:**
        1. Pick a model, two perturbations (X/Y), and an image class.
        2. Pick an image from the preview row.
        3. Press **Run ▶** to generate the surface.
        4. **Click** any point on the surface
        5. View the GradCAM of the perturbed image to see where the model 'looked' to make its decision.
        """
    )
    # little "close" button under the info box
    close_col1, close_col2 = st.columns([0.15, 0.85])
    with close_col1:
        if st.button("Close"):
            st.session_state["show_welcome"] = False

#############################################################
#Top control bar
c_model, c_x, c_y, c_class, c_preview, c_run, c_info = st.columns(
    [1.1, 1.1, 1.1, 1.1, 2.4, 0.6, 0.25]
)

#Info button
with c_info:
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    with st.popover("ℹ️"):
        st.markdown(
            """
            **How to use this app**

            1. **Select Model** – pick a pretrained backbone.
            2. **1st / 2nd Perturbation** – choose the 2 axes for the confidence surface.
            3. **Select Image Class** – pick the folder of images.
            4. **Preview images** – click the image to select.
            5. **Run ▶** – builds the landscape.
            6. **Click the 3D plot** – to run **GradCAM** for that perturbation.
            7. Right panel shows:
               - Base image
               - Perturbed GradCAM
               - Legend
               - Top-3 predictions

            🔁 If you change images or axes, **press Run again** to update.
            """,
            unsafe_allow_html=False,
        )


#1) Select model
with c_model:
    model_name = st.selectbox("**Select Model**", MODELS)

#2) Select X axis (1st perturbation)
all_axes = list(PERTURB_AXES.keys())
with c_x:
    x_axis = st.selectbox("**1st Perturbation (X-Axis)**", all_axes, index=0)
    
#3) Select Y axis (2nd perturbation) – must be different from X
with c_y:
    y_choices = [a for a in all_axes if a != x_axis]
    prev_y = st.session_state.get("y_axis_selected")
    if prev_y in y_choices:
        y_default = y_choices.index(prev_y)
    else:
        y_default = 0
    y_axis = st.selectbox(
        "**2nd Perturbation (Y-Axis)**",
        y_choices,
        index=y_default,
        key="y_axis_selected",
    )

#4) Select image class
with c_class:
    image_class = st.selectbox("**Select Image Class**", IMAGE_CLASSES)

img_dir = os.path.join(IMAGE_ROOT, image_class)
img_files = sorted(os.listdir(img_dir))

#If user changed class, reset selected image
if st.session_state["selected_class"] != image_class:
    st.session_state["selected_class"] = image_class
    st.session_state["selected_image"] = None

#If no image selected yet, pick the first
if st.session_state["selected_image"] is None and img_files:
    st.session_state["selected_image"] = img_files[0]

#5) Preview images in class (thumbnails with select buttons)
with c_preview:
    st.markdown("**Preview images in the class**")

    display_files = img_files[:5]
    current_sel = st.session_state.get("selected_image")

    #Default selection if none
    if current_sel not in display_files:
        current_sel = display_files[0] if display_files else None
        st.session_state["selected_image"] = current_sel

    #Show thumbnails
    cols = st.columns(len(display_files))
    for i, fname in enumerate(display_files):
        col = cols[i]
        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        col.image(img, use_column_width=True)

    #Centered radio buttons under images
    st.markdown(
        """
        <style>
        /* only touch radios in this preview block */
        .img-radio-wrap .stRadio > div {
            display: flex;
            justify-content: center;
            gap: 0.35rem;
        }
        .img-radio-wrap .stRadio label {
            flex: 1;
            max-width: 95px;
            text-align: center;
            font-size: 0.68rem;      /* smaller */
            line-height: 1.0rem;
            white-space: normal;
            word-break: break-word;
            margin-top: 0.25rem;
        }
        .img-radio-wrap .stRadio label span {
            font-size: 0.68rem !important;   /* make sure inner span also shrinks */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='img-radio-wrap'>", unsafe_allow_html=True)
    selected_fname = st.radio(
        "",
        options=display_files,
        index=display_files.index(current_sel),
        horizontal=True,
        label_visibility="collapsed",
        key="image_radio",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["selected_image"] = selected_fname

#6) Run app
with c_run:
    # small spacer to push the button down to the image row
    st.markdown("<div style='height:43px'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            font-weight: 600;
            font-size: 16px;
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    run_button = st.button("**Run ▶**", use_container_width=True)
#############################################################
#Load image and preprocess
# Determine what to actually display (frozen state from last Run)
disp_class = st.session_state.get("active_class") or image_class
disp_image = st.session_state.get("active_image") or st.session_state["selected_image"]
disp_model_name = st.session_state.get("active_model") or model_name
disp_x = st.session_state.get("active_x") or x_axis
disp_y = st.session_state.get("active_y") or y_axis

# Load corresponding image
img_dir = os.path.join(IMAGE_ROOT, disp_class)
img_path = os.path.join(img_dir, disp_image)
base_img = Image.open(img_path).convert("RGB")
image_name = st.session_state["selected_image"]

#Preprocess image for model
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

#############################################################
#Setup plot
x_cfg = PERTURB_AXES[disp_x]
y_cfg = PERTURB_AXES[disp_y]

x_vals = np.linspace(x_cfg["min"], x_cfg["max"], x_cfg["steps"])
y_vals = np.linspace(y_cfg["min"], y_cfg["max"], y_cfg["steps"])
#############################################################
#Get top model prediction
Z = None
model = None
class_idx = None

def get_top_class_idx(model, pil_img):
    with torch.no_grad():
        t = preprocess(pil_img).unsqueeze(0)
        out = model(t)
        return out.argmax(dim=1).item()
    
#############################################################
#Run pipeline for model
results_placeholder = st.empty()

if run_button:
    #Freeze states
    st.session_state["active_image"] = st.session_state["selected_image"]
    st.session_state["active_class"] = image_class
    st.session_state["active_model"] = model_name
    st.session_state["active_x"] = x_axis
    st.session_state["active_y"] = y_axis

    #Set active values
    active_class = st.session_state["active_class"]
    active_image = st.session_state["active_image"]
    active_model = st.session_state["active_model"]
    active_x = st.session_state["active_x"]
    active_y = st.session_state["active_y"]

    #Reload image + axis grids for run
    img_dir = os.path.join(IMAGE_ROOT, active_class)
    img_path = os.path.join(img_dir, active_image)
    base_img = Image.open(img_path).convert("RGB")

    x_cfg = PERTURB_AXES[active_x]
    y_cfg = PERTURB_AXES[active_y]
    x_vals = np.linspace(x_cfg["min"], x_cfg["max"], x_cfg["steps"])
    y_vals = np.linspace(y_cfg["min"], y_cfg["max"], y_cfg["steps"])

    with results_placeholder.container():
        with st.spinner('Running model and generating surface...'):
            print('[APP] Run button pressed')
            print(f'[APP] Loading model: {model_name}')

            model = get_model(active_model)

            #Detect class on BASE image
            print('[APP] Classifying base image...')
            class_idx = get_top_class_idx(model, base_img)
            print(f'[APP] Predicted class_idx = {class_idx}')

            config = {
                "model": active_model,
                "image_class": active_class,
                "image_name": active_image,
                "x_axis": active_x,
                "y_axis": active_y,
                "x_vals": x_vals.tolist(),
                "y_vals": y_vals.tolist(),
                "class_idx": class_idx,
            }

            #Make confidence landscape
            Z_cached, key = load_from_cache(config)
            if Z_cached is not None:
                print(f'[APP] Loaded landscape from cache: {key}')
                Z = Z_cached
            else:
                print('[APP] Building confidence landscape (may take few secs)')
                Z = build_confidence_landscape(
                    model,
                    base_img,
                    class_idx,
                    x_axis,
                    y_axis,
                    x_vals,
                    y_vals,
                    preprocess,
                    device="cpu",
                )
                print('[APP] Done building confidence landscape')
                print('[APP] Saving to cache...')
                save_to_cache(key, Z)
    
            st.session_state["Z"] = Z
            st.session_state["model_obj"] = model
            st.session_state["class_idx"] = class_idx
            st.session_state["pred_box_html"] = None
            st.session_state["last_click"] = None
#############################################################
#Find confidence
def find_nearest_index(arr, value):
    arr = np.asarray(arr)
    return (np.abs(arr - value)).argmin()
#############################################################
#Find target layer for GradCAM
def pick_target_layer(model, model_name:str):
    '''Pick model layer to hook GradCAM to'''
    canon = model_name.lower().replace(" ", "").replace("-", "").replace("__", "_")
    
    if "mobilenet_v3_large" in canon:
        #Torchvision mobilenet_v3_large
        return model.features[-1]
    if "resnet18" in canon:
        #Torchvision resnet18
        return model.layer4[-1]
    if "efficientnet_b0" in canon:
        #Torchvision efficientnet_b0
        return model.features[-1]
    
    #Fallback: try common attributes
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    
    raise ValueError(f"Don't know how to pick target layer for model '{model_name}'")
#############################################################
def get_imagenet_labels_for_model(model_name: str):
    name = model_name.lower()
    if "resnet18" in name:
        return ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    if "mobilenet" in name or "mbilenet" in name:
        return MobileNet_V3_Large_Weights.IMAGENET1K_V1.meta["categories"]
    if "efficientnet" in name:
        return EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]
    # fallback: 0..999
    return [f"class {i}" for i in range(1000)]
#############################################################
# -------------------------------------------------------
# DISPLAY / INTERACTION AFTER THE SURFACE EXISTS
# -------------------------------------------------------
Z = st.session_state.get("Z")
model = st.session_state.get("model_obj")
class_idx = st.session_state.get("class_idx")


if Z is not None:
    # main 2-column layout: plot on left, images/preds on right
    left, right = st.columns([2.2, 1], vertical_alignment="top")

    # ================= LEFT: 3D SURFACE =================
    with left:
        fig = make_plot(
            x_vals,
            y_vals,
            Z,
            x_label=PERTURB_AXES[disp_x]["label"],
            y_label=PERTURB_AXES[disp_y]["label"],
        )

        clicked_points = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            key="surface",
            override_height=520,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if (not clicked_points) and (st.session_state.get("last_click") is None):
            st.info("Click the surface to see point details.")

    # ================= RIGHT: IMAGES + LEGEND + PREDS =================
    with right:
        # 1) image row (two columns) – force top alignment
        img_col, grad_col = st.columns([1, 1], vertical_alignment="top")

        with img_col:
            st.image(base_img, caption="Base image", use_column_width=True)

        #Fill this only when user clicks
        with grad_col:
            grad_placeholder = st.empty()
        if st.session_state.get("last_click") is None:
            st.markdown(
                """
                <div style="
                    background-color: #111417;
                    padding: 0.9rem 1.2rem;
                    border-radius: 0.5rem;
                    border: 1px solid #2f3335;
                    margin-top: 0.75rem;
                    width: 100%;
                    text-align: center;
                    color: #cccccc;
                    font-size: 0.9rem;
                ">
                    Click a point on the plot to see the GradCAM visualization here.
                </div>
                """,
                unsafe_allow_html=True,
            )


        # 2) legend under BOTH images, centered
        legend_img = colormap_legend(width=220, height=20)  
        buf = io.BytesIO()
        legend_img.save(buf, format="PNG")
        legend_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        st.markdown(
            f"""
            <div style="width:100%; display:flex; justify-content:center; margin-top:0.5rem; margin-bottom:0.75rem;">
                <div style="text-align:center;">
                    <img src="data:image/png;base64,{legend_b64}" style="width:220px; height:20px;" />
                    <div style="font-size:0.75rem; color:#cccccc; margin-top:0.25rem;">
                        Red = more useful to classification
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3) predictions block – fill later
        preds_placeholder = st.empty()
        if st.session_state.get("pred_box_html"):
            preds_placeholder.markdown(
                st.session_state["pred_box_html"],
                unsafe_allow_html=True,
            )

    # ================= HANDLE CLICK =================
    if clicked_points:
        pt = clicked_points[0]
        clicked_x = float(pt["x"])
        clicked_y = float(pt["y"])

        # find nearest indices in the perturbation grids
        x_idx = (np.abs(x_vals - clicked_x)).argmin()
        y_idx = (np.abs(y_vals - clicked_y)).argmin()
        clicked_conf = float(Z[x_idx, y_idx])

        # store for the bottom "Current point" card
        st.session_state["last_click"] = {
            "x_axis": disp_x,
            "x_value": clicked_x,
            "y_axis": disp_y,
            "y_value": clicked_y,
            "confidence": clicked_conf,
        }

        # ---------- GRADCAM: render into placeholder ----------
        with grad_placeholder.container():
            with st.spinner("Running GradCAM..."):
                x_key = disp_x.lower()
                y_key = disp_y.lower()

                # 1) apply the two perturbations the user clicked
                pert = PERTURB_FUNCS[x_key](base_img, clicked_x)
                pert = PERTURB_FUNCS[y_key](pert, clicked_y)

                # 2) make sure we have a model + class_idx
                model = st.session_state.get("model_obj") or get_model(model_name)
                model.eval()
                st.session_state["model_obj"] = model

                class_idx = st.session_state.get("class_idx")
                if class_idx is None:
                    # re-predict from base image if somehow missing
                    with torch.no_grad():
                        tmp = preprocess(base_img).unsqueeze(0)
                        class_idx = model(tmp).argmax(1).item()
                        st.session_state["class_idx"] = class_idx

                # 3) choose correct layer
                target_layer = pick_target_layer(model, model_name)

                # 4) run gradcam
                gc = GradCAM(model, target_layer)
                inp = preprocess(pert).unsqueeze(0)
                cam = gc.generate(inp, class_idx)

                # 5) overlay (you already wrote gradcam_overlay_cv2)
                overlaid = gradcam_overlay_cv2(
                    pert.resize((224, 224)).convert("L").convert("RGB"),
                    cam,
                    alpha=0.6,
                )

            st.image(overlaid, caption="GradCAM overlay", use_column_width=True)

        # ---------- TOP-3 PREDICTIONS: render into placeholder ----------
        with preds_placeholder.container():
            with torch.no_grad():
                logits = model(inp)
                probs = torch.softmax(logits, dim=1)[0]
                topk = torch.topk(probs, 3)

            # try to use ImageNet names if available
            labels_src = globals().get("IMAGENET_LABELS", None)

            html = """
                <div style="
                    background-color:#111417;
                    padding:0.9rem 1.2rem;
                    border-radius:0.5rem;
                    border:1px solid #2f3335;
                    margin-top:1.25rem;
                    width:100%;
                    text-align:center;
                ">
                    <p style="margin:0 0 0.5rem 0;font-weight:600;">
                        Top-3 predictions at this point:
                    </p>
                """

            for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                if labels_src and idx < len(labels_src):
                    label = labels_src[idx]
                else:
                    label = f"class {idx}"
                html += f"<p style='margin:0;'>• <b>{label}</b>: {p*100:.3f}% confidence</p>"

            html += "</div>"

            st.markdown(html, unsafe_allow_html=True)
            st.session_state["pred_box_html"] = html

    # ================= CURRENT POINT CARD (CENTERED) =================
    last = st.session_state.get("last_click")
    if last:
        with left:  
            st.markdown(
                f"""
                <div style="
                    background-color: #111417;
                    padding: 0.9rem 1.2rem;
                    border-radius: 0.5rem;
                    border: 1px solid #2f3335;
                    margin-top: 1.25rem;
                    width: 100%;
                    text-align: center;
                ">
                    <p style="margin:0 0 0.5rem 0; font-weight: 600;">Current point</p>
                    <p style="margin:0;">{last['x_axis']}: <b>{last['x_value']:.3f}</b></p>
                    <p style="margin:0;">{last['y_axis']}: <b>{last['y_value']:.3f}</b></p>
                    <p style="margin:0;">Confidence: <b>{last['confidence']:.3f}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )