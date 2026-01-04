# app.py — Malaria DIP (Hough Circles RBC + HSV Parasites/WBC)
# Adds: FOV filtering + nested-circle suppression + Original vs Annotated side-by-side
# Adds: 5 debug views (FOV mask, CLAHE gray, Blur gray, Parasite mask, WBC mask)
#
# Run:
#   pip install streamlit opencv-python-headless numpy pandas
#   streamlit run app.py

import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Utilities
# ----------------------------
def read_bgr(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_max_dim(img, max_dim=1600):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img, 1.0
    s = max_dim / float(m)
    new_w, new_h = int(w * s), int(h * s)
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, s


def compute_fov_mask(img_bgr, thresh=15):
    """
    Create a binary field-of-view mask (white = inside microscope circle).
    Uses gray threshold + closing to fill the circular FOV.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    k = np.ones((11, 11), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m


def apply_fov(img_bgr, fov_mask):
    out = img_bgr.copy()
    out[fov_mask == 0] = (0, 0, 0)
    return out


def circle_mask(shape_hw, cx, cy, r):
    h, w = shape_hw
    Y, X = np.ogrid[:h, :w]
    return ((X - cx) ** 2 + (Y - cy) ** 2) <= (r ** 2)


# ----------------------------
# Circle post-processing
# ----------------------------
def filter_circles_by_fov(circles, fov_mask, inside_ratio=0.85, center_only=False):
    """
    Keep circles that lie inside the field-of-view mask.
    - center_only=True: just check center pixel is inside fov
    - otherwise: require >= inside_ratio of sampled perimeter points inside fov
    """
    if circles is None or len(circles) == 0 or fov_mask is None:
        return circles

    h, w = fov_mask.shape[:2]
    kept = []

    for (x, y, r) in circles:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if fov_mask[y, x] == 0:
            continue

        if center_only:
            kept.append((x, y, r))
            continue

        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        xs = (x + r * np.cos(angles)).astype(np.int32)
        ys = (y + r * np.sin(angles)).astype(np.int32)

        ok = 0
        total = 0
        for xi, yi in zip(xs, ys):
            if 0 <= xi < w and 0 <= yi < h:
                total += 1
                if fov_mask[yi, xi] > 0:
                    ok += 1

        if total > 0 and (ok / total) >= inside_ratio:
            kept.append((x, y, r))

    return np.array(kept, dtype=np.int32)


def suppress_nested_circles(circles, center_dist_thresh=10, radius_sim_thresh=0.30):
    """
    Remove duplicates / nested circles.
    Strategy:
    - sort by radius descending (keep larger first)
    - if another circle has very close center and similar radius -> drop it
    - if a smaller circle center lies inside a larger one with close-ish radius -> drop smaller
    """
    if circles is None or len(circles) == 0:
        return circles

    circles = np.array(circles, dtype=np.int32)
    circles = circles[np.argsort(-circles[:, 2])]  # big radius first

    kept = []
    for (x, y, r) in circles:
        drop = False
        for (kx, ky, kr) in kept:
            dx = x - kx
            dy = y - ky
            d2 = dx * dx + dy * dy

            # Duplicate: near-identical center, similar radius -> keep the first (larger first)
            if d2 <= center_dist_thresh ** 2:
                if abs(r - kr) / max(kr, 1) < radius_sim_thresh:
                    drop = True
                    break
                if r < kr:
                    drop = True
                    break

            # Nested: small circle center inside big circle and radii not wildly different
            if d2 < (kr * 0.6) ** 2 and r < kr and abs(r - kr) / max(kr, 1) < 0.45:
                drop = True
                break

        if not drop:
            kept.append((x, y, r))

    return np.array(kept, dtype=np.int32)


# ----------------------------
# Core detection
# ----------------------------
def detect_rbc_hough(gray_u8, dp, minDist, p1, p2, minR, maxR):
    circles = cv2.HoughCircles(
        gray_u8,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=p1,
        param2=p2,
        minRadius=minR,
        maxRadius=maxR,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.int32)
    return np.round(circles[0]).astype(np.int32)


def parasite_mask_hsv(img_bgr, h_low, h_high, s_low, v_low, fov_mask=None):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, 255, 255], dtype=np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    if fov_mask is not None:
        m = cv2.bitwise_and(m, fov_mask)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m


def wbc_mask_from_purple(purple_mask, wbc_area_min=900):
    cnts, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wbc = np.zeros_like(purple_mask)
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= wbc_area_min:
            cv2.drawContours(wbc, [c], -1, 255, -1)
    return wbc


def count_infected_rbcs(img_shape, circles, parasite_mask, infected_min_pixels, wbc_mask=None):
    h, w = img_shape[:2]
    infected = 0
    rows = []

    for i, (cx, cy, r) in enumerate(circles, start=1):
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue

        m = circle_mask((h, w), cx, cy, r)

        pm = (parasite_mask > 0) & m
        if wbc_mask is not None:
            pm = pm & (wbc_mask == 0)

        parasite_pixels = int(np.sum(pm))
        is_inf = parasite_pixels >= infected_min_pixels
        if is_inf:
            infected += 1

        rows.append(
            {
                "rbc_id": i,
                "cx": int(cx),
                "cy": int(cy),
                "r": int(r),
                "parasite_pixels": parasite_pixels,
                "infected": bool(is_inf),
            }
        )

    df = pd.DataFrame(rows)
    return infected, df


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Malaria DIP (Hough + HSV)", layout="wide")
st.title("Automated Malaria Parasite Detection & RBC Counting (DIP — Hough + HSV)")
st.caption(
    "Pipeline: FOV masking → CLAHE/blur → Hough circles for RBC → HSV purple mask for parasites/WBC → infected RBC by overlap. "
    "Educational prototype only."
)

with st.sidebar:
    up = st.file_uploader("Upload smear image", type=["jpg", "jpeg", "png"])

    st.header("Performance / Scaling")
    max_dim = st.slider("Max processing dimension", 800, 2600, 1600, 100)

    st.header("FOV (microscope circle)")
    use_fov = st.checkbox("Apply FOV mask", value=True)
    fov_thr = st.slider("FOV threshold", 1, 80, 15, 1)
    fov_inside_ratio = st.slider("Circle inside-FOV ratio", 0.50, 0.98, 0.85, 0.01)

    st.header("Preprocess")
    clahe_clip = st.slider("CLAHE clipLimit", 1.0, 6.0, 3.0, 0.1)
    blur_k = st.slider("Gaussian blur k", 3, 15, 7, 2)

    st.header("RBC detection (Hough Circles)")
    dp = st.slider("dp", 1.0, 2.5, 1.2, 0.1)
    minDist = st.slider("minDist (px)", 8, 80, 16, 1)
    p1 = st.slider("param1 (Canny high)", 30, 200, 80, 5)
    p2 = st.slider("param2 (accumulator)", 10, 80, 22, 1)
    minR = st.slider("minRadius", 5, 100, 14, 1)
    maxR = st.slider("maxRadius", 10, 140, 36, 1)

    st.header("Duplicate / nested circle suppression")
    dup_center_dist = st.slider("Center-distance threshold", 6, 30, 10, 1)
    dup_radius_sim = st.slider("Radius similarity threshold", 0.10, 0.60, 0.30, 0.01)

    st.header("Parasite/WBC (HSV purple)")
    h_low, h_high = st.slider("Hue range", 0, 179, (120, 170), 1)
    s_low = st.slider("Min S", 0, 255, 25, 1)
    v_low = st.slider("Min V", 0, 255, 25, 1)
    wbc_area_min = st.slider("WBC min area (on purple mask)", 200, 8000, 1000, 50)

    st.header("Infection Rule A")
    infected_min_pixels = st.slider("Min purple pixels inside RBC", 1, 80, 2, 1)

    show_debug = st.checkbox("Show debug views", value=True)

if not up:
    st.info("Upload an image to begin.")
    st.stop()

# ----------------------------
# Process image
# ----------------------------
t0 = time.perf_counter()
img0 = read_bgr(up)

# Resize for processing
img, scale = resize_max_dim(img0, max_dim=max_dim)

# FOV mask
if use_fov:
    fov = compute_fov_mask(img, thresh=fov_thr)
    img_proc = apply_fov(img, fov)
else:
    fov = None
    img_proc = img.copy()

# Preprocess for Hough
gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)

k = blur_k if blur_k % 2 == 1 else blur_k + 1
gray_blur = cv2.GaussianBlur(gray_eq, (k, k), 0)

# Detect circles
circles_raw = detect_rbc_hough(gray_blur, dp=dp, minDist=minDist, p1=p1, p2=p2, minR=minR, maxR=maxR)

# Filter circles outside microscope circle
circles_fov = circles_raw
if use_fov and fov is not None and len(circles_fov):
    circles_fov = filter_circles_by_fov(circles_fov, fov, inside_ratio=float(fov_inside_ratio), center_only=False)

# Remove nested/duplicate circles
circles = suppress_nested_circles(
    circles_fov, center_dist_thresh=int(dup_center_dist), radius_sim_thresh=float(dup_radius_sim)
)

# Parasite / WBC masks
par_mask = parasite_mask_hsv(img_proc, h_low, h_high, s_low, v_low, fov_mask=fov)
wbc_mask = wbc_mask_from_purple(par_mask, wbc_area_min=wbc_area_min)

# Infection
infected_count, df_rbc = count_infected_rbcs(img_proc.shape, circles, par_mask, infected_min_pixels, wbc_mask=wbc_mask)

# Overlay
annot = img_proc.copy()

# RBC circles (green) + infected centers (red)
for (x, y, r) in circles:
    cv2.circle(annot, (int(x), int(y)), int(r), (0, 255, 0), 2)

if len(df_rbc):
    for _, row in df_rbc[df_rbc["infected"] == True].iterrows():
        cv2.circle(annot, (int(row["cx"]), int(row["cy"])), 5, (0, 0, 255), -1)

# Magenta tint parasites
mag = np.zeros_like(annot)
mag[par_mask > 0] = (255, 0, 255)
annot = cv2.addWeighted(annot, 1.0, mag, 0.55, 0)

# Blue tint WBC
blu = np.zeros_like(annot)
blu[wbc_mask > 0] = (255, 0, 0)
annot = cv2.addWeighted(annot, 1.0, blu, 0.35, 0)

runtime = time.perf_counter() - t0

total_rbc = int(len(circles))
parasitemia = (infected_count / total_rbc * 100.0) if total_rbc > 0 else 0.0

# ----------------------------
# Display: Original + Annotated side-by-side
# ----------------------------
cL, cR = st.columns(2, gap="large")
with cL:
    st.subheader("Original (processed resolution)")
    st.image(bgr_to_rgb(img_proc), use_container_width=True)

with cR:
    st.subheader("Annotated Output")
    st.image(bgr_to_rgb(annot), use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total RBC (Hough)", total_rbc)
m2.metric("Infected RBC", int(infected_count))
m3.metric("Parasitemia %", f"{parasitemia:.2f}")
m4.metric("Runtime (s)", f"{runtime:.2f}")

st.caption(
    "Green circles = RBC candidates. Red dots = infected RBC centroids. "
    "Magenta = purple mask (parasite + nuclei). Blue tint = WBC regions (excluded from infection overlap)."
)

# Table + download
st.subheader("Per-RBC Table")
st.dataframe(df_rbc, use_container_width=True, height=340)
st.download_button(
    "Download RBC table (CSV)",
    data=df_rbc.to_csv(index=False).encode("utf-8"),
    file_name="rbc_results.csv",
    mime="text/csv",
)

# ----------------------------
# Debug Views (5 views)
# ----------------------------
if show_debug:
    st.divider()
    st.subheader("Debug Views")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.caption("FOV mask (white = inside microscope circle)")
        if fov is not None:
            st.image(fov, use_container_width=True)
        else:
            st.info("FOV disabled.")

    with d2:
        st.caption("Gray (CLAHE)")
        st.image(gray_eq, use_container_width=True)

    with d3:
        st.caption("Gray (CLAHE + blur)")
        st.image(gray_blur, use_container_width=True)

    d4, d5, d6 = st.columns(3)
    with d4:
        st.caption("Parasite/Purple mask (HSV)")
        st.image(par_mask, use_container_width=True)

    with d5:
        st.caption("WBC mask (large purple blobs)")
        st.image(wbc_mask, use_container_width=True)

    with d6:
        st.caption("RBC circles: raw vs filtered count")
        raw_n = int(len(circles_raw))
        fov_n = int(len(circles_fov)) if circles_fov is not None else 0
        final_n = int(len(circles))
        st.write(f"- Raw circles: **{raw_n}**")
        st.write(f"- After FOV filter: **{fov_n}**")
        st.write(f"- After nested suppression: **{final_n}**")

st.info("Disclaimer: Prototype for educational/research use only; not for clinical diagnosis.")
