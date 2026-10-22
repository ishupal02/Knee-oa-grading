"""
Streamlit frontend for the knee osteoarthritis severity classifier.

Setup (once, with the venv active):
    pip install streamlit

Run (keep knee_oa_model_best.pt in the same folder as this file):
    streamlit run app.py
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

MODEL_PATH = Path(__file__).parent / "knee_oa_model_best.pt"
SAMPLES_DIR = Path(__file__).parent / "samples"
METRICS_PATH = Path(__file__).parent / "metrics.json"
NUM_CLASSES = 5
IMAGE_SIZE = 224
LOW_CONFIDENCE = 0.60

# Severity ramp: cool teal for healthy, warming to deep red for severe.
COLORS = ["#2F8F83", "#86A24E", "#D8A03A", "#D06A2E", "#A93636"]
TEXT_ON = ["#FFFFFF", "#17242F", "#17242F", "#FFFFFF", "#FFFFFF"]

GRADES = [
    ("healthy", "No visible signs of osteoarthritis."),
    ("doubtful", "Possible tiny bone spurs; the joint space looks normal."),
    ("mild", "Clear bone spurs and possible narrowing of the joint space."),
    ("moderate", "Several bone spurs and definite narrowing of the joint space."),
    ("severe", "Large bone spurs, marked narrowing and possible bone deformity."),
]

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,500;7..72,600&family=Public+Sans:wght@400;500;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.stApp {
    background:
        linear-gradient(rgba(23, 36, 47, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(23, 36, 47, 0.025) 1px, transparent 1px),
        #F3F5F3;
    background-size: 28px 28px;
    font-family: 'Public Sans', system-ui, sans-serif;
    color: #17242F;
}
.block-container { max-width: 1100px; padding-top: 2.3rem; padding-bottom: 3.5rem; }

.masthead { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 2.8rem; }
.mark { width: 42px; height: 42px; display: grid; place-items: center; background: #17242F;
                color: #F3F5F3; font-family: 'Literata', Georgia, serif; font-weight: 600; font-size: 1.1rem; }
.brand { color: #17242F; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.08em; }
.brand small { display: block; color: #687781; font-size: 0.68rem; font-weight: 500; margin-top: 0.2rem; letter-spacing: 0.12em; }
.eyebrow { color: #2F8F83; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 0.65rem; }
.intro { display: grid; grid-template-columns: minmax(0, 1fr) 220px; align-items: end; gap: 2rem; margin-bottom: 2rem; }
.intro-note { border-left: 2px solid #2F8F83; padding: 0.15rem 0 0.15rem 0.9rem; color: #5B6B78;
                         font-size: 0.86rem; line-height: 1.5; }
.intro-note strong { display: block; color: #17242F; font-size: 0.78rem; margin-bottom: 0.25rem; }
.section-label { color: #5B6B78; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
                                 text-transform: uppercase; margin: 1.4rem 0 0.65rem; }
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid #CBD5DC; }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 0.2rem; color: #687781; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #17242F !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #2F8F83; height: 3px; }
.stFileUploader > section { border: 1px dashed #9BAAB1; border-radius: 2px; background: rgba(255,255,255,0.62); }
.stFileUploader > section:hover { border-color: #2F8F83; }
.stButton > button { border: 1px solid #CBD5DC; border-radius: 2px; background: #FFFFFF;
                                         color: #17242F; font-weight: 500; min-height: 40px; }
.stButton > button:hover { border-color: #2F8F83; color: #236E66; background: #F5FAF8; }
.stToggle label { color: #17242F; }
.stButton > button[kind="primary"] { background: #236E66; border-color: #236E66; color: #FFFFFF; }
.stButton > button[kind="primary"]:hover { background: #1A5B55; border-color: #1A5B55; color: #FFFFFF; }

.home-hero { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.8fr); gap: 3rem;
             align-items: center; padding: 2.3rem; background: #E7EFEC; border: 1px solid #D3DFDA; }
.home-copy { padding: 0.5rem 0; }
.home-title { max-width: 13ch; font-family: 'Literata', Georgia, serif; font-size: 3.25rem; font-weight: 600;
              line-height: 1.08; color: #17242F; margin: 0 0 1rem; }
.home-lede { max-width: 56ch; color: #52646E; font-size: 1rem; line-height: 1.7; margin: 0; }
.home-image { min-width: 0; background: #101619; border: 1px solid #BAC8C8; }
.home-image img { display: block; width: 100%; aspect-ratio: 1.22; object-fit: cover; }
.image-caption { display: flex; justify-content: space-between; gap: 1rem; padding: 0.7rem 0.85rem;
                 color: #E9F0EE; font-size: 0.7rem; letter-spacing: 0.09em; }
.home-cta { margin-top: 1.35rem; }
.home-section { margin-top: 3.2rem; }
.home-section-head { display: flex; justify-content: space-between; align-items: end; gap: 1rem; margin-bottom: 1.1rem; }
.home-section-title { font-family: 'Literata', Georgia, serif; color: #17242F; font-size: 1.75rem;
                      font-weight: 600; line-height: 1.2; margin: 0; }
.home-section-note { max-width: 42ch; color: #5B6B78; font-size: 0.86rem; line-height: 1.5; margin: 0; }
.guide-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; align-items: stretch; }
.guide-card { height: 100%; background: #FFFFFF; border: 1px solid #D8E0E2; }
.guide-card img { width: 100%; height: 165px; display: block; object-fit: cover; background: #101619; }
.guide-body { padding: 1rem 1.05rem 1.15rem; }
.guide-step { color: #2F8F83; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; }
.guide-title { color: #17242F; font-size: 1rem; font-weight: 600; margin: 0.4rem 0; }
.guide-text { color: #5B6B78; font-size: 0.86rem; line-height: 1.55; margin: 0; }
.usecase-strip { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; padding: 1.5rem 0;
                 border-top: 1px solid #CBD5DC; border-bottom: 1px solid #CBD5DC; }
.usecase { display: flex; gap: 0.9rem; align-items: start; }
.usecase-index { color: #2F8F83; font-family: 'Literata', Georgia, serif; font-size: 1.35rem; line-height: 1; }
.usecase strong { display: block; color: #17242F; font-size: 0.9rem; margin-bottom: 0.3rem; }
.usecase p { color: #5B6B78; font-size: 0.84rem; line-height: 1.55; margin: 0; }
.home-disclaimer { color: #5B6B78; font-size: 0.82rem; line-height: 1.6; margin-top: 1.3rem; }
.page-nav { display: flex; justify-content: flex-end; margin: -1.4rem 0 0.8rem; }

.title { font-family: 'Literata', Georgia, serif; font-weight: 600; font-size: 2.7rem;
         line-height: 1.15; letter-spacing: -0.01em; color: #17242F; margin: 0 0 0.7rem; }
.lede { color: #5B6B78; font-size: 1rem; line-height: 1.65; max-width: 60ch; margin: 0; }
.hint { color: #5B6B78; font-size: 0.95rem; line-height: 1.55; margin-top: 0.6rem; }
.h2 { font-family: 'Literata', Georgia, serif; font-weight: 600; font-size: 1.2rem;
      color: #17242F; margin: 2.2rem 0 0.8rem; }

.result { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
          gap: 2rem; align-items: center; margin-top: 1.4rem; padding: 1.4rem;
          background: #FFFFFF; border: 1px solid #D8E0E2; border-top: 3px solid #2F8F83; }
.xray { width: 100%; max-height: 430px; object-fit: contain; display: block; border: 1px solid #CBD5DC; background: #101619; }
.grade-num { font-family: 'Literata', Georgia, serif; font-size: 5.5rem; line-height: 1;
             font-weight: 600; margin: 0; }
.grade-name { font-size: 1.2rem; font-weight: 600; color: #17242F; margin: 0.5rem 0 0.3rem; }
.grade-desc { color: #5B6B78; line-height: 1.55; margin: 0 0 1rem; }
.conf { color: #17242F; font-weight: 500; margin: 0; }
.notice { border-left: 3px solid #D8A03A; padding: 0.2rem 0 0.2rem 0.8rem; color: #17242F;
          margin: 1rem 0 0; font-size: 0.95rem; line-height: 1.5; }

.scale { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px;
         align-items: end; margin: 1.5rem 0 0.4rem; }
.seg { position: relative; height: 38px; display: flex; align-items: center;
       justify-content: center; color: #17242F; font-weight: 600; }
.seg::before { content: ""; position: absolute; inset: 0; background: var(--c); opacity: 0.3; }
.seg span { position: relative; }
.seg.active { height: 62px; font-size: 1.2rem; }
.seg.active::before { opacity: 1; }

.prow { display: grid; grid-template-columns: 5.5rem 1fr 3.6rem; align-items: center;
        gap: 0.8rem; margin: 0.45rem 0; font-size: 0.95rem; color: #17242F; }
.track { height: 10px; background: #DCE4E9; }
.fill { height: 100%; }
.pv { text-align: right; font-variant-numeric: tabular-nums; }

.cm { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 0.9rem; }
.cm th { font-weight: 500; color: #5B6B78; padding: 0.4rem 0.2rem; text-align: center; }
.cm td { padding: 0.65rem 0.2rem; text-align: center; border: 2px solid #EDF1F4; color: #17242F; }

.fine { color: #5B6B78; font-size: 0.82rem; line-height: 1.6; margin-top: 2.6rem;
        border-top: 1px solid #CBD5DC; padding-top: 1rem; }

@media (max-width: 640px) {
    .block-container { padding-top: 1.4rem; }
    .masthead { margin-bottom: 2rem; }
    .intro { grid-template-columns: 1fr; gap: 1rem; }
        .result { grid-template-columns: 1fr; }
    .result { padding: 1rem; gap: 1rem; }
    .title { font-size: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
        .home-hero { grid-template-columns: 1fr; gap: 1.5rem; padding: 1.2rem; }
        .home-title { font-size: 2.3rem; }
        .home-section { margin-top: 2.3rem; }
        .home-section-head { display: block; }
        .home-section-note { margin-top: 0.5rem; }
        .guide-grid { grid-template-columns: 1fr; }
        .usecase-strip { grid-template-columns: 1fr; gap: 1.3rem; }
}
</style>
"""

FOOTER = (
    '<p class="fine">ResNet-18 fine-tuned on about 5,800 knee X-rays; 68.1% accuracy on '
    "1,656 held-out test images. Neighbouring grades are easy to confuse, even for doctors. "
    "This is a learning project, not a medical device, and it cannot replace a clinician. "
    "Data: Knee Osteoarthritis Dataset with Severity on Kaggle (CC BY 4.0).</p>"
)


@st.cache_resource(show_spinner="Loading model")
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(model, image):
    x = TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0].tolist()


def image_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# NOTE: HTML strings below are built without leading indentation or blank
# lines on purpose; Markdown would treat indented lines as a code block.
def result_html(img_b64, pred, probs):
    name, desc = GRADES[pred]
    conf = probs[pred]
    order = sorted(range(NUM_CLASSES), key=lambda g: probs[g], reverse=True)

    notice = ""
    if conf < LOW_CONFIDENCE:
        other = order[1]
        notice = (
            f'<p class="notice">The model is not sure about this one. '
            f"Grade {other} ({probs[other] * 100:.0f}%) is also plausible.</p>"
        )

    return (
        '<div class="result">'
        f'<img class="xray" src="data:image/png;base64,{img_b64}" alt="Uploaded knee X-ray">'
        "<div>"
        f'<p class="grade-num" style="color:{COLORS[pred]}">{pred}</p>'
        f'<p class="grade-name">Kellgren-Lawrence grade {pred}: {name}</p>'
        f'<p class="grade-desc">{desc}</p>'
        f'<p class="conf">Confidence {conf * 100:.1f}%</p>'
        f"{notice}"
        "</div></div>"
    )


def scale_html(pred):
    segs = []
    for g in range(NUM_CLASSES):
        active = " active" if g == pred else ""
        style = f"--c:{COLORS[g]};"
        if g == pred:
            style += f"color:{TEXT_ON[g]};"
        segs.append(f'<div class="seg{active}" style="{style}"><span>{g}</span></div>')
    return f'<div class="scale">{"".join(segs)}</div>'


def probs_html(probs):
    rows = []
    for g in range(NUM_CLASSES):
        pct = probs[g] * 100
        rows.append(
            '<div class="prow">'
            f"<span>Grade {g}</span>"
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;background:{COLORS[g]}"></div></div>'
            f'<span class="pv">{pct:.1f}%</span>'
            "</div>"
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Grad-CAM: which parts of the image pushed the model toward its answer
# ---------------------------------------------------------------------------
def gradcam(model, image, target):
    feats, grads = [], []

    def forward_hook(module, inputs, output):
        feats.append(output)
        output.register_hook(lambda g: grads.append(g))

    handle = model.layer4.register_forward_hook(forward_hook)
    try:
        model.zero_grad()
        out = model(TRANSFORM(image).unsqueeze(0))
        out[0, target].backward()
    finally:
        handle.remove()

    weights = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats[0]).sum(dim=1))[0].detach().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def heat_colors(v):
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay(image, cam):
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(image.size, Image.BILINEAR)
    v = np.asarray(cam_img, dtype="float32") / 255.0
    base = np.asarray(image, dtype="float32") / 255.0
    alpha = (v * 0.65)[..., None]
    out = (base * (1 - alpha) + heat_colors(v) * alpha) * 255
    return Image.fromarray(out.astype("uint8"))


# ---------------------------------------------------------------------------
# Performance tab
# ---------------------------------------------------------------------------
def confusion_html(cm):
    head_row = "<tr><th></th>" + "".join(f"<th>Predicted {g}</th>" for g in range(NUM_CLASSES)) + "</tr>"
    rows = []
    for t, row in enumerate(cm):
        total = sum(row) or 1
        cells = []
        for count in row:
            frac = count / total
            text = "#FFFFFF" if frac > 0.55 else "#17242F"
            cells.append(f'<td style="background:rgba(43,90,120,{frac * 0.9:.2f});color:{text}">{count}</td>')
        rows.append(f"<tr><th>True {t}</th>{''.join(cells)}</tr>")
    return f'<table class="cm">{head_row}{"".join(rows)}</table>'


def render_performance():
    if not METRICS_PATH.exists():
        st.markdown(
            '<p class="hint">No results yet. Run <code>python evaluate.py</code> once to create '
            "metrics.json, then reload this page.</p>",
            unsafe_allow_html=True,
        )
        return
    m = json.loads(METRICS_PATH.read_text())
    cm = m["confusion"]
    total = sum(map(sum, cm))
    accs = [(row[g] / sum(row)) if sum(row) else 0 for g, row in enumerate(cm)]
    within_one = sum(cm[i][j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if abs(i - j) <= 1) / total
    st.markdown(
        f'<p class="lede">Accuracy {m["accuracy"] * 100:.1f}% on {m["n"]:,} test images the model never saw '
        f"in training. {within_one * 100:.1f}% of predictions are within one grade of the true grade.</p>"
        '<h2 class="h2">Accuracy by true grade</h2>' + probs_html(accs)
        + '<h2 class="h2">Where the model gets confused</h2>' + confusion_html(cm)
        + '<p class="hint">Each row is the true grade. Darker cells hold more of that row’s images.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Grader tab
# ---------------------------------------------------------------------------
def render_grader():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found: {MODEL_PATH.name}. Copy it into the same folder as app.py, "
            "then reload this page."
        )
        return

    uploaded = st.file_uploader(
        "Choose a knee X-ray (PNG or JPG)",
        type=["png", "jpg", "jpeg"],
        help="Works best on cropped single-knee images like the ones in the training set.",
    )

    available = [g for g in range(NUM_CLASSES) if (SAMPLES_DIR / f"grade_{g}.png").exists()]
    if available:
        st.markdown('<p class="section-label">Or start with a sample</p>', unsafe_allow_html=True)
        for col, g in zip(st.columns(len(available)), available):
            if col.button(f"True grade {g}", key=f"sample_{g}"):
                st.session_state["sample"] = g

    source, truth = None, None
    if uploaded is not None:
        source = uploaded
        st.session_state.pop("sample", None)
    elif "sample" in st.session_state:
        truth = st.session_state["sample"]
        source = SAMPLES_DIR / f"grade_{truth}.png"

    if source is None:
        st.markdown(
            '<p class="hint">Your image preview and grade breakdown will appear here once an X-ray is selected.</p>',
            unsafe_allow_html=True,
        )
        return

    try:
        image = Image.open(source).convert("RGB")
    except Exception:
        st.error("That file could not be read as an image. Try a PNG or JPG.")
        return

    model = load_model()
    with st.spinner("Grading"):
        probs = predict(model, image)
    pred = max(range(NUM_CLASSES), key=lambda g: probs[g])

    show_cam = st.toggle("Show where the model looked (Grad-CAM)", value=True)
    shown = overlay(image, gradcam(model, image, pred)) if show_cam else image

    st.markdown(result_html(image_to_base64(shown), pred, probs), unsafe_allow_html=True)
    if show_cam:
        st.markdown(
            '<p class="hint">Warm areas pushed the model toward this grade. It is a hint about the '
            "model, not proof that it reads the X-ray the way a doctor does.</p>",
            unsafe_allow_html=True,
        )
    if truth is not None:
        st.markdown(f'<p class="hint">This sample is a true grade {truth} image.</p>', unsafe_allow_html=True)

    st.markdown(scale_html(pred), unsafe_allow_html=True)
    st.markdown('<h2 class="h2">Probability by grade</h2>' + probs_html(probs), unsafe_allow_html=True)


def render_home():
    hero_path = SAMPLES_DIR / "grade_2.png"
    if not hero_path.exists():
        hero_path = SAMPLES_DIR / "grade_0.png"

    st.markdown(
        '<div class="masthead"><div class="mark">K</div><div class="brand">KNEE / OA<small>IMAGING REVIEW</small></div></div>',
        unsafe_allow_html=True,
    )
    if hero_path.exists():
        hero_b64 = image_to_base64(Image.open(hero_path).convert("RGB"))
        image_markup = (
            f'<div class="home-image"><img src="data:image/png;base64,{hero_b64}" alt="Example knee X-ray">'
            '<div class="image-caption"><span>EXAMPLE RADIOGRAPH</span><span>KL SCALE · 0–4</span></div></div>'
        )
    else:
        image_markup = '<div class="home-image"><div class="image-caption">KNEE RADIOGRAPH</div></div>'

    st.markdown(
        '<section class="home-hero"><div class="home-copy"><p class="eyebrow">A simple research demonstration</p>'
        '<h1 class="home-title">Explore knee X-ray grading.</h1>'
        '<p class="home-lede">See how an image classifier estimates a Kellgren-Lawrence grade, then review its '
        'confidence and the regions that influenced its output.</p></div>' + image_markup + '</section>',
        unsafe_allow_html=True,
    )
    if st.button("Try demo", type="primary", key="home_try_demo"):
        st.session_state["page"] = "grader"
        st.rerun()

    st.markdown(
        '<section class="home-section"><div class="home-section-head"><div>'
        '<p class="eyebrow">Getting started</p><h2 class="home-section-title">Three steps, one clear result</h2></div>'
        '<p class="home-section-note">Use a cropped, single-knee frontal X-ray for the most relevant demonstration. '
        'PNG and JPG images are supported.</p></div></section>',
        unsafe_allow_html=True,
    )
    guide = [
        (0, "01 / Choose an image", "Select a de-identified knee X-ray from your device, or use one of the built-in examples."),
        (2, "02 / Run the estimate", "The model returns one of five KL grades. This is an experimental estimate, not a clinical reading."),
        (4, "03 / Review the output", "Compare the grade probabilities and optional heatmap; warm regions show model attention, not proof."),
    ]
    guide_cards = []
    for grade, title, description in guide:
        image_path = SAMPLES_DIR / f"grade_{grade}.png"
        if not image_path.exists():
            continue
        image_b64 = image_to_base64(Image.open(image_path).convert("RGB"))
        guide_cards.append(
            '<div class="guide-card">'
            f'<img src="data:image/png;base64,{image_b64}" alt="Example knee X-ray for grade {grade}">'
            f'<div class="guide-body"><span class="guide-step">{title.split(" / ")[0]}</span>'
            f'<h3 class="guide-title">{title.split(" / ", 1)[1]}</h3>'
            f'<p class="guide-text">{description}</p></div></div>'
        )
    st.markdown('<div class="guide-grid">' + "".join(guide_cards) + '</div>', unsafe_allow_html=True)

    st.markdown(
        '<section class="home-section"><div class="home-section-head"><div>'
        '<p class="eyebrow">Designed for exploration</p><h2 class="home-section-title">Useful for learning, not diagnosis</h2></div></div>'
        '<div class="usecase-strip"><div class="usecase"><span class="usecase-index">01</span><div>'
        '<strong>Education and demonstrations</strong><p>Explore the five-level grading scale and see how an image model expresses uncertainty.</p>'
        '</div></div><div class="usecase"><span class="usecase-index">02</span><div>'
        '<strong>Model behavior review</strong><p>Inspect confidence scores and Grad-CAM visualizations as a starting point for discussion.</p>'
        '</div></div></div>'
        '<p class="home-disclaimer">This project is not a medical device and cannot diagnose, screen, or guide treatment. '
        'Do not upload identifiable patient information. Always rely on a qualified clinician for medical interpretation.</p>'
        '</section>',
        unsafe_allow_html=True,
    )
    if st.button("Open the demo", type="primary", key="home_bottom_demo"):
        st.session_state["page"] = "grader"
        st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Knee OA | Imaging Review", page_icon="K", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state["page"] = "home"

if st.session_state["page"] == "home":
    render_home()
else:
    st.markdown(
        '<div class="masthead"><div class="mark">K</div><div class="brand">KNEE / OA<small>IMAGING REVIEW</small></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Home", key="grader_home"):
        st.session_state["page"] = "home"
        st.rerun()
    st.markdown(
        '<div class="intro"><div><p class="eyebrow">Radiographic assessment · KL scale</p>'
        '<h1 class="title">Knee X-ray grading</h1>'
        '<p class="lede">An AI-assisted estimate of osteoarthritis severity from a cropped knee X-ray, '
        'organized across the five Kellgren-Lawrence grades.</p></div>'
        '<div class="intro-note"><strong>Research tool</strong>For education and exploration only. '
        'Not a diagnosis or substitute for clinical review.</div></div>',
        unsafe_allow_html=True,
    )

    tab_grade, tab_perf = st.tabs(["Grade an X-ray", "Model performance"])
    with tab_perf:
        render_performance()
    with tab_grade:
        render_grader()

st.markdown(FOOTER, unsafe_allow_html=True)
