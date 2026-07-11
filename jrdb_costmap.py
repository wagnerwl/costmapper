"""
JRDB Costmap Viewer
====================
Erstellt eine visuelle Kostenkarte (Bird's-Eye-View) aus Kamerabildern.

Pipeline:
  1. Alle 5 Kameras → 2D-Positionen (X, Y) im Ego-Frame
  2. Richtung aus Combined (bildbasiert) oder Ground-Truth (rot_z)
  3. Geschwindigkeit aus Track-History × 15 fps
  4. Lineare Prädiktion: pos(t) = pos + fwd × speed × t  für t = 0..1.5 s
  5. Gaussian-Blobs an allen Prädiktorpunkten → Costmap-Grid
  6. Visualisierung als Heatmap + Skelette + Pfeile

Kostenkarten-Schichten:
  STATISCH:  Gaussian(σ=0.4m) an aktueller Position
  DYNAMISCH: Gaussian-Ellipse entlang des Pfades (σ_quer=0.3m, σ_längs wächst mit t)

Richtungsquellen (umschaltbar per Button):
  COMBINED  — bildbasiert (Schulter + Augen + Hüfte + Velocity)
  GT        — Ground-Truth rot_z (nur für Evaluation/Präsentation)

Aufruf:
    python jrdb_costmap.py
    python jrdb_costmap.py --scene stlc-111-2019-04-19_0
"""

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import yaml
from matplotlib.widgets import Button, Slider

# ── Pfade ─────────────────────────────────────────────────────────────────────
BASE             = Path(__file__).parent / "jrdb"
IMG_BASE         = BASE / "images"
POSE_BASE        = BASE / "labels" / "labels_2d_pose_coco"
POSE_STITCH_BASE = BASE / "labels" / "labels_2d_pose_stitched_coco"
L3D_BASE         = BASE / "labels" / "labels_3d"
CALIB_CAM        = BASE / "calibration" / "cameras.yaml"
CALIB_LID        = BASE / "calibration" / "lidars.yaml"

CAMERAS    = ["image_0", "image_2", "image_4", "image_6", "image_8"]
CAM_SUFFIX = {"image_0":"image0","image_2":"image2","image_4":"image4",
              "image_6":"image6","image_8":"image8"}
CAM_SENSOR = {"image_0":"sensor_0","image_2":"sensor_2","image_4":"sensor_4",
              "image_6":"sensor_6","image_8":"sensor_8"}

CAM_HEIGHT  = 0.82
JRDB_FPS    = 15.0          # Aufnahme-Framerate JRDB

# ── Costmap-Parameter ─────────────────────────────────────────────────────────
GRID_SIZE    = 200           # Pixel pro Achse
GRID_RANGE   = 8.0           # Meter: Karte zeigt ±8m
GRID_RES     = 2 * GRID_RANGE / GRID_SIZE   # Meter pro Pixel

PRED_STEPS   = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]  # feinere Schritte, längerer Horizont
SIG_STATIC   = 0.45          # σ statischer Blob (Meter)
SIG_TRANSV   = 0.45          # σ quer zur Bewegung (breiter)
SIG_LONG_0   = 0.50          # σ längs bei t=0 (größer)
SIG_LONG_K   = 0.50          # σ wächst schneller pro Sekunde

# ── Skelett ───────────────────────────────────────────────────────────────────
SKELETON_EDGES = [
    (1,2),(0,4),(3,4),(8,10),(5,7),(10,13),(14,16),
    (4,5),(7,12),(4,8),(3,6),(13,15),(11,14),(6,9),(8,11),
]
SHOULDER_WIDTH_NORM = 0.44
EYE_WIDTH_NORM      = 0.12
W_WITH_VEL    = {"vel": 0.60, "shldr": 0.25, "eye": 0.10, "hip": 0.05}
W_WITHOUT_VEL = {"shldr": 0.60, "eye": 0.25, "hip": 0.15}

PERSON_COLORS_HEX = [
    "#e57373","#f06292","#ba68c8","#9575cd",
    "#64b5f6","#4dd0e1","#4db6ac","#81c784",
    "#dce775","#ffb74d","#ff8a65","#a1887f",
]
def person_color(tid): return PERSON_COLORS_HEX[int(tid) % len(PERSON_COLORS_HEX)]

MODE_COMBINED = "Combined (bildbasiert)"
MODE_GT       = "Ground-Truth"

# ── Colormap ──────────────────────────────────────────────────────────────────
# Schwarz → Dunkelblau → Gelb → Rot (wie Wärmebildkamera / Heatmap)
COSTMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "costmap",
    [(0.00, "#050508"),    # leer = fast schwarz
     (0.15, "#0a1a3a"),    # niedrig = dunkelblau
     (0.40, "#1a4a8a"),    # mittel = blau
     (0.65, "#d4a020"),    # erhöht = gelb-orange
     (0.85, "#e84040"),    # hoch = rot
     (1.00, "#ffffff")],   # Maximum = weiß
)

# ── Kalibrierung & Datenladen ─────────────────────────────────────────────────

def load_calibration(camera):
    with open(CALIB_CAM) as f: cam_data = yaml.safe_load(f)
    with open(CALIB_LID) as f: lid_data = yaml.safe_load(f)
    sensor = CAM_SENSOR[camera]
    s = cam_data["cameras"][sensor]
    K       = np.array(list(map(float, s["K"].split()))).reshape(3,3)
    D       = np.array(list(map(float, s["D"].split())))
    cam2ego = np.array(lid_data[sensor]["cam2ego"])
    return K, D, cam2ego

def load_all_calibrations():
    return {cam: load_calibration(cam) for cam in CAMERAS}

def list_scenes():
    p = IMG_BASE / "image_0"
    return sorted(d.name for d in p.iterdir() if d.is_dir()) if p.exists() else []

def load_pose_labels(scene, camera):
    path = POSE_BASE / f"{scene}_{CAM_SUFFIX[camera]}.json"
    if not path.exists(): return {}
    with open(path) as f: data = json.load(f)
    id2fn = {img["id"]: Path(img["file_name"]).name for img in data["images"]}
    out = {}
    for ann in data["annotations"]:
        fn = id2fn.get(ann["image_id"], "")
        if fn: out.setdefault(fn, []).append(ann)
    return out

def load_labels_3d(scene):
    path = L3D_BASE / f"{scene}.json"
    if not path.exists(): return {}
    with open(path) as f: data = json.load(f)
    result = {}
    for pcd_name, anns in data["labels"].items():
        stem = Path(pcd_name).stem
        by_id = {}
        for a in anns:
            try: tid = int(a["label_id"].split(":")[1])
            except: continue
            by_id[tid] = a
        result[stem] = by_id
    return result

# ── Projektion ────────────────────────────────────────────────────────────────

def pixel_ray_ego(u, v, K, D, cam2ego):
    pts   = np.array([[[u, v]]], dtype=np.float32)
    pts_r = cv2.undistortPoints(pts, K, D, P=K)
    u_r, v_r = pts_r[0,0]
    ray_cam = np.array([(u_r-K[0,2])/K[0,0], (v_r-K[1,2])/K[1,1], 1.0])
    return cam2ego[:3,3].copy(), cam2ego[:3,:3] @ ray_cam

def foot_depth(u, v, K, D, cam2ego):
    origin, ray = pixel_ray_ego(u, v, K, D, cam2ego)
    gz = origin[2] - CAM_HEIGHT
    if abs(ray[2]) < 1e-6: return None
    t = (gz - origin[2]) / ray[2]
    return t if t > 0.1 else None

def project_keypoints_3d(kps, K, D, cam2ego):
    depths = [foot_depth(kps[fi,0], kps[fi,1], K, D, cam2ego)
              for fi in [15,16] if kps[fi,2] > 0]
    depths = [d for d in depths if d is not None]
    if not depths: return None
    t = float(np.mean(depths))
    origin, _ = pixel_ray_ego(kps[0,0], kps[0,1], K, D, cam2ego)
    pts = np.full((17,3), np.nan)
    for i in range(17):
        if kps[i,2] > 0:
            _, ray = pixel_ray_ego(kps[i,0], kps[i,1], K, D, cam2ego)
            pts[i] = origin + t * ray
    for i in range(17):
        if np.isnan(pts[i,0]):
            nb = [j for (a,b) in SKELETON_EDGES
                  for j in ([b] if a==i else [a] if b==i else [])
                  if not np.isnan(pts[j,0])]
            if nb: pts[i] = np.nanmean([pts[j] for j in nb], axis=0)
    known = pts[~np.isnan(pts[:,0])]
    if not len(known): return None
    c = known.mean(axis=0)
    for i in range(17):
        if np.isnan(pts[i,0]): pts[i] = c
    return pts

# ── Orientierungsschätzung ────────────────────────────────────────────────────

def _kp_height(kps):
    vis = kps[kps[:,2] > 0]
    if len(vis) < 2: return None
    h = float(vis[:,1].max() - vis[:,1].min())
    return h if h > 10 else None

def yaw_shoulder(kps):
    r_s, l_s = kps[3], kps[5]
    if r_s[2] == 0 or l_s[2] == 0: return None, 0.0
    kp_h = _kp_height(kps)
    if kp_h is None: return None, 0.0
    apparent_w = abs(r_s[0] - l_s[0])
    expected_w = kp_h * SHOULDER_WIDTH_NORM
    ratio   = float(np.clip(apparent_w / (expected_w + 1e-6), 0.0, 1.0))
    sign    = 1.0 if r_s[0] < l_s[0] else -1.0
    return sign * np.arccos(ratio), min(1.0, apparent_w / (expected_w + 1e-6))

def yaw_eye(kps):
    r_e, l_e = kps[1], kps[2]
    if r_e[2] == 0 or l_e[2] == 0: return None, 0.0
    kp_h = _kp_height(kps)
    if kp_h is None: return None, 0.0
    apparent_w = abs(r_e[0] - l_e[0])
    expected_w = kp_h * EYE_WIDTH_NORM
    ratio   = float(np.clip(apparent_w / (expected_w + 1e-6), 0.0, 1.0))
    sign    = 1.0 if r_e[0] < l_e[0] else -1.0
    return sign * np.arccos(ratio), min(1.0, apparent_w / (expected_w + 1e-6))

def yaw_hip(pts3d):
    r_hip, l_hip = pts3d[10], pts3d[11]
    if np.isnan(r_hip[0]) or np.isnan(l_hip[0]): return None, 0.0
    dY = r_hip[1] - l_hip[1]
    if abs(dY) < 0.005: return None, 0.0
    yaw  = 0.0 if -np.sign(dY) > 0 else np.pi
    return yaw, min(1.0, abs(dY) / 0.1) * 0.3

def yaw_velocity(history):
    if len(history) < 2: return None, 0.0
    recent  = list(history)[-3:]
    deltas  = [np.array(recent[i+1]) - np.array(recent[i]) for i in range(len(recent)-1)]
    weights = list(range(1, len(deltas)+1))
    weighted = sum(w * d for w, d in zip(weights, deltas))
    speed = np.linalg.norm(weighted)
    if speed < 0.02: return None, 0.0
    return float(np.arctan2(weighted[1], weighted[0])), min(1.0, speed / 0.15)

def yaw_gt(label_3d):
    if label_3d is None: return None, 0.0
    return -label_3d["box"]["rot_z"] + np.pi, 1.0

def combined_yaw(yaws_confs):
    vecs = []
    for yaw, w, conf in yaws_confs:
        if yaw is None: continue
        eff_w = w * conf
        if eff_w < 0.001: continue
        vecs.append(eff_w * np.array([np.cos(yaw), np.sin(yaw)]))
    if not vecs: return None
    sumvec = sum(vecs)
    n = np.linalg.norm(sumvec)
    if n < 1e-6: return None
    return float(np.arctan2(sumvec[1], sumvec[0]))

# ── Costmap-Berechnung ────────────────────────────────────────────────────────

def world_to_grid(x, y):
    """Welt-Koordinaten (Ego-Frame) → Grid-Pixel-Index (row, col)."""
    col = int((-y + GRID_RANGE) / GRID_RES)   # Y=links → Spalte (negiert)
    row = int((-x + GRID_RANGE) / GRID_RES)   # X=vorwärts → Zeile (negiert)
    return row, col

def add_gaussian_world(grid, wx, wy, sigma, weight=1.0):
    """
    Runder isotroper Gaussian an Welt-Koordinaten (wx, wy).
    sigma in Metern.
    """
    row, col = world_to_grid(wx, wy)
    sig_px = sigma / GRID_RES
    r_rad  = int(4 * sig_px) + 1

    r0, r1 = max(0, row - r_rad), min(GRID_SIZE, row + r_rad + 1)
    c0, c1 = max(0, col - r_rad), min(GRID_SIZE, col + r_rad + 1)
    if r0 >= r1 or c0 >= c1:
        return  # Punkt außerhalb des Grids

    rs = np.arange(r0, r1)
    cs = np.arange(c0, c1)
    rr, cc = np.meshgrid(rs, cs, indexing="ij")
    dist_sq = ((rr - row)**2 + (cc - col)**2) / (sig_px**2 + 1e-6)
    g = np.exp(-0.5 * dist_sq)
    grid[r0:r1, c0:c1] += weight * g

def add_ellipse_world(grid, wx, wy, yaw, sigma_long, sigma_transv, weight=1.0):
    """
    Rotierte elliptische Gaussian an Welt-Koordinaten.
    Alle Sigma in Metern, yaw in Radiant.
    Arbeitet komplett in Welt-Metern → organische Form unabhängig von Grid-Alignment.
    """
    row, col = world_to_grid(wx, wy)
    max_sig  = max(sigma_long, sigma_transv)
    r_rad    = int(4 * max_sig / GRID_RES) + 1

    r0, r1 = max(0, row - r_rad), min(GRID_SIZE, row + r_rad + 1)
    c0, c1 = max(0, col - r_rad), min(GRID_SIZE, col + r_rad + 1)
    if r0 >= r1 or c0 >= c1:
        return  # Punkt außerhalb des Grids

    rs = np.arange(r0, r1)
    cs = np.arange(c0, c1)
    rr, cc = np.meshgrid(rs, cs, indexing="ij")

    # Pixel-Delta in Welt-Meter umrechnen
    # Grid-Row+ = Welt-X- (vorwärts nach oben), Grid-Col+ = Welt-Y- (links nach rechts)
    dx_world = -(rr - row) * GRID_RES   # Welt-X Differenz (Meter)
    dy_world = -(cc - col) * GRID_RES   # Welt-Y Differenz (Meter)

    # Rotation in lokales Gaussian-KS (longitudinal/transversal)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    d_long  =  dx_world * cos_y + dy_world * sin_y    # entlang Bewegung
    d_trans = -dx_world * sin_y + dy_world * cos_y    # quer zur Bewegung

    # Gaussian mit unterschiedlichen Sigmas
    g = np.exp(-0.5 * ((d_long / (sigma_long + 1e-6))**2 +
                        (d_trans / (sigma_transv + 1e-6))**2))
    grid[r0:r1, c0:c1] += weight * g

def compute_costmap(persons_data):
    """
    persons_data: Liste von Dicts mit pos, yaw, speed, confidence.
    Gibt (GRID_SIZE × GRID_SIZE) Float-Array zurück, normiert [0,1].
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for p in persons_data:
        px, py = p["pos"]          # Ego-Frame Meter
        yaw    = p["yaw"]          # Radiant (None wenn unbekannt)
        speed  = p["speed"]        # m/s
        conf   = p["yaw_conf"]     # 0..1

        # ── Statischer Blob (aktuelle Position) ──
        add_gaussian_world(grid, px, py, SIG_STATIC, weight=1.5)

        if yaw is None:
            continue

        # ── Dynamische Ellipsen entlang des Prädiktionspfades ──
        # Bei stehendem Ziel: Mindest-Speed ansetzen damit Richtungsstrahl sichtbar
        eff_speed = max(speed, 0.6)   # mindestens 0.6 m/s für Darstellung
        fwd = np.array([np.cos(yaw), np.sin(yaw)])

        for t in PRED_STEPS[1:]:
            pred_pos = np.array([px, py]) + fwd * eff_speed * t
            ppx, ppy = float(pred_pos[0]), float(pred_pos[1])

            sig_l = (SIG_LONG_0 + SIG_LONG_K * t) / (conf + 0.3)
            sig_t = SIG_TRANSV + 0.15 * t
            # Gewicht: hoch damit Schweif deutlich sichtbar bleibt
            speed_factor = np.clip(speed / 1.0, 0.4, 1.0)
            w = 1.8 * speed_factor * np.exp(-0.15 * t)   # langsamer Abfall

            add_ellipse_world(grid, ppx, ppy, yaw, sig_l, sig_t, weight=w)

    # Normierung
    if grid.max() > 0:
        grid /= grid.max()
    return grid

# ── Viewer ────────────────────────────────────────────────────────────────────

class CostmapViewer:
    def __init__(self, init_scene=""):
        self.scenes    = list_scenes()
        if not self.scenes: raise RuntimeError("Keine Szenen gefunden")
        self.scene_idx = self.scenes.index(init_scene) if init_scene in self.scenes else 0
        self.cam_idx   = 0    # für Kamerabild
        self.frame_idx = 0
        self.playing   = False
        self.fps       = 8
        self.mode      = MODE_COMBINED

        self.calibrations = load_all_calibrations()
        self.track_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=6))

        self._load_scene()
        self._build_ui()
        self._draw()

    # ── Laden ─────────────────────────────────────────────────────────────────

    def _load_scene(self):
        scene  = self.scenes[self.scene_idx]
        camera = CAMERAS[self.cam_idx]
        self.image_paths = sorted((IMG_BASE / camera / scene).glob("*.jpg"))
        self.pose_labels_all = {cam: load_pose_labels(scene, cam) for cam in CAMERAS}
        self.pose_labels_cam = self.pose_labels_all[camera]
        self.labels_3d       = load_labels_3d(scene)
        self.track_hist.clear()
        self.frame_idx = 0
        if hasattr(self, "slider"):
            self.slider.valmax = max(len(self.image_paths)-1, 1)
            self.slider.set_val(0)

    # ── Personendaten für aktuellen Frame ─────────────────────────────────────

    def _get_persons(self):
        if not self.image_paths: return []
        fname = self.image_paths[self.frame_idx].name
        stem  = Path(fname).stem
        l3d_f = self.labels_3d.get(stem, {})
        result   = []
        seen_ids: set[int] = set()

        for cam in CAMERAS:
            K, D, cam2ego = self.calibrations[cam]
            for ann in self.pose_labels_all[cam].get(fname, []):
                track_id = ann.get("track_id", 0)
                if track_id in seen_ids: continue

                kps   = np.array(ann["keypoints"], dtype=float).reshape(17,3)
                pts3d = project_keypoints_3d(kps, K, D, cam2ego)
                if pts3d is None: continue
                seen_ids.add(track_id)

                # Bodenposition aus Fußmitte
                feet = pts3d[[15,16]]
                vf   = feet[~np.isnan(feet[:,0])]
                if not len(vf): continue
                pos_xy = vf.mean(axis=0)[:2]

                # Track-History → Geschwindigkeit
                hist = self.track_hist[track_id]
                hist.append(pos_xy.tolist())
                speed = 0.0
                if len(hist) >= 2:
                    delta = np.array(hist[-1]) - np.array(hist[-2])
                    speed = float(np.linalg.norm(delta)) * JRDB_FPS

                # Richtung
                yaw_mirror = None
                standing   = False
                if self.mode == MODE_GT:
                    yaw, yaw_conf = yaw_gt(l3d_f.get(track_id))
                else:
                    y_s, c_s = yaw_shoulder(kps)
                    y_e, c_e = yaw_eye(kps)
                    y_h, c_h = yaw_hip(pts3d)
                    y_v, c_v = yaw_velocity(hist)
                    if y_v is not None:
                        sources = [(y_v,W_WITH_VEL["vel"],c_v),(y_s,W_WITH_VEL["shldr"],c_s),
                                   (y_e,W_WITH_VEL["eye"],c_e),(y_h,W_WITH_VEL["hip"],c_h)]
                    else:
                        sources = [(y_s,W_WITHOUT_VEL["shldr"],c_s),
                                   (y_e,W_WITHOUT_VEL["eye"],c_e),(y_h,W_WITHOUT_VEL["hip"],c_h)]
                        standing = True
                    yaw       = combined_yaw(sources)
                    yaw_conf  = max(c_s, c_v if y_v is not None else 0.0)

                    # Spiegel-Yaw für Alternativpfeil (arccos-Ambiguität)
                    if standing and y_s is not None:
                        r_s, l_s = kps[3], kps[5]
                        if r_s[2] > 0 and l_s[2] > 0:
                            kp_h = _kp_height(kps)
                            if kp_h and kp_h > 10:
                                apparent_w = abs(r_s[0] - l_s[0])
                                expected_w = kp_h * SHOULDER_WIDTH_NORM
                                ratio = float(np.clip(apparent_w / (expected_w+1e-6), 0, 1))
                                yaw_mag = np.arccos(ratio)
                                sign = 1.0 if r_s[0] < l_s[0] else -1.0
                                yaw_mirror = -sign * yaw_mag

                result.append({
                    "track_id":   track_id,
                    "pos":        pos_xy.tolist(),
                    "pts3d":      pts3d,
                    "yaw":        yaw,
                    "yaw_mirror": yaw_mirror,
                    "standing":   standing,
                    "speed":      speed,
                    "yaw_conf":   yaw_conf,
                    "kps":        kps,
                })
        return result

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(14, 9), facecolor="#07070e")
        self.fig.canvas.manager.set_window_title("JRDB Costmap Viewer")

        # BEV + Costmap (links)
        self.ax_bev = self.fig.add_axes([0.02, 0.13, 0.55, 0.84])
        self.ax_bev.set_facecolor("#050508")

        # Kamerabild (rechts oben)
        self.ax_cam = self.fig.add_axes([0.60, 0.52, 0.38, 0.44])
        self.ax_cam.axis("off")

        # Info (rechts unten)
        self.ax_info = self.fig.add_axes([0.60, 0.13, 0.38, 0.36])
        self.ax_info.axis("off")
        self.ax_info.set_facecolor("#0d0d18")

        # Colorbar
        ax_cb = self.fig.add_axes([0.575, 0.13, 0.015, 0.84])
        sm = plt.cm.ScalarMappable(cmap=COSTMAP_CMAP, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cb = self.fig.colorbar(sm, cax=ax_cb)
        cb.set_label("Kosten", color="#666688", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="#444466", labelsize=6)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#666688")

        # Slider
        ax_sl = self.fig.add_axes([0.02, 0.075, 0.55, 0.025], facecolor="#0d0d18")
        n = len(self.image_paths)
        self.slider = Slider(ax_sl, "", 0, max(n-1,1), valinit=0, valstep=1, color="#4fc3f7")
        self.slider.label.set_color("#333355")
        self.slider.valtext.set_color("#666688")
        self.slider.on_changed(self._on_slider)

        bs = dict(color="#10101e", hovercolor="#1e1e38")

        # Frame-Buttons
        self.btn_prev = Button(self.fig.add_axes([0.12, 0.025, 0.06, 0.04]), "◀◀", **bs)
        self.btn_play = Button(self.fig.add_axes([0.19, 0.025, 0.07, 0.04]), "▶ Play", **bs)
        self.btn_next = Button(self.fig.add_axes([0.27, 0.025, 0.06, 0.04]), "▶▶", **bs)
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            b.label.set_color("#aaaacc"); b.label.set_fontsize(9)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_play.on_clicked(self._on_play)
        self.btn_next.on_clicked(self._on_next)

        # Szene / Kamera
        self.btn_sp    = Button(self.fig.add_axes([0.60, 0.075, 0.08, 0.038]), "◀ Szene", **bs)
        self.btn_sn    = Button(self.fig.add_axes([0.69, 0.075, 0.08, 0.038]), "Szene ▶", **bs)
        self.btn_cam_p = Button(self.fig.add_axes([0.79, 0.075, 0.045, 0.038]), "◀ K", **bs)
        self.btn_cam_n = Button(self.fig.add_axes([0.84, 0.075, 0.045, 0.038]), "K ▶", **bs)
        for b in (self.btn_sp, self.btn_sn, self.btn_cam_p, self.btn_cam_n):
            b.label.set_color("#aaaacc"); b.label.set_fontsize(8)
        self.btn_sp.on_clicked(self._on_scene_p)
        self.btn_sn.on_clicked(self._on_scene_n)
        self.btn_cam_p.on_clicked(self._on_cam_prev)
        self.btn_cam_n.on_clicked(self._on_cam)

        # Richtungs-Modus Toggle
        self.btn_mode = Button(self.fig.add_axes([0.60, 0.025, 0.20, 0.04]),
                               self.mode, **bs)
        self.btn_mode.label.set_color("#44ee55"); self.btn_mode.label.set_fontsize(8)
        self.btn_mode.on_clicked(self._on_mode)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Zeichnen ──────────────────────────────────────────────────────────────

    def _draw(self):
        persons = self._get_persons()

        # Costmap berechnen
        costmap = compute_costmap(persons)

        # ── BEV-Plot ──
        self.ax_bev.cla()
        self.ax_bev.set_facecolor("#050508")

        # Costmap als Hintergrund
        extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE]
        self.ax_bev.imshow(
            costmap, origin="upper",
            extent=extent,
            cmap=COSTMAP_CMAP, vmin=0, vmax=1,
            aspect="equal", interpolation="bilinear", alpha=0.92,
        )

        # Grid-Linien
        for v in np.arange(-GRID_RANGE, GRID_RANGE+1, 2):
            self.ax_bev.axhline(v, color="#ffffff08", lw=0.4)
            self.ax_bev.axvline(v, color="#ffffff08", lw=0.4)

        # Abstandsringe
        for r in [2, 4, 6]:
            self.ax_bev.add_patch(plt.Circle(
                (0,0), r, fill=False, color="#ffffff15", lw=0.6, ls="--"))
            self.ax_bev.text(0.08, r+0.1, f"{r}m",
                             color="#ffffff30", fontsize=6, ha="left", va="bottom")

        # Roboter
        self.ax_bev.scatter([0],[0], s=130, c="#ffffff", marker="^", zorder=10)
        self.ax_bev.annotate("", xy=(0, 1.3), xytext=(0,0),
                             arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=1.5))

        # Skelette + Pfeile pro Person
        for p in persons:
            px, py = p["pos"]
            color  = person_color(p["track_id"])

            # BEV: plot_x = -Y_ego, plot_y = X_ego
            bx = -py
            by =  px
            self.ax_bev.scatter([bx], [by], s=35, c=color, zorder=8, linewidths=0)

            # ── Richtungspfeil (immer wenn yaw vorhanden) ──
            if p["yaw"] is not None:
                fwd_x = np.cos(p["yaw"])
                fwd_y = np.sin(p["yaw"])
                # Mindestlänge 1.0m für Sichtbarkeit, max 2.0m
                arr_len = max(1.0, min(2.0, p["speed"] * 1.2))
                # Hauptpfeil (durchgezogen)
                self.ax_bev.annotate(
                    "", xy=(bx - fwd_y*arr_len, by + fwd_x*arr_len),
                    xytext=(bx, by),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, alpha=0.9))

                # ── Alternativpfeil (gestrichelt) wenn stehend ──
                standing = p.get("standing", False)
                yaw_mirror = p.get("yaw_mirror")
                if standing and yaw_mirror is not None:
                    mfwd_x = np.cos(yaw_mirror)
                    mfwd_y = np.sin(yaw_mirror)
                    self.ax_bev.annotate(
                        "", xy=(bx - mfwd_y*arr_len*0.8, by + mfwd_x*arr_len*0.8),
                        xytext=(bx, by),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.2, alpha=0.4, linestyle="dashed"))

            # ID-Label
            self.ax_bev.text(bx, by+0.2, f"#{p['track_id']}",
                             color=color, fontsize=6.5, ha="center", va="bottom",
                             fontweight="bold")

            # ── Prädiktionspunkte entlang des Pfades ──
            if p["yaw"] is not None:
                fwd = np.array([np.cos(p["yaw"]), np.sin(p["yaw"])])
                eff_speed = max(p["speed"], 0.6)
                for t in PRED_STEPS[1:]:
                    pred = np.array([px, py]) + fwd * eff_speed * t
                    alpha = 0.6 * np.exp(-0.3 * t)
                    self.ax_bev.scatter([-pred[1]], [pred[0]],
                                        s=12, c=color, alpha=alpha,
                                        zorder=7, linewidths=0)

        scene  = self.scenes[self.scene_idx]
        n      = len(self.image_paths)
        self.ax_bev.set_title(
            f"{scene}  [{CAMERAS[self.cam_idx]}]  Frame {self.frame_idx+1}/{n}"
            f"  |  {len(persons)} Personen  |  Modus: {self.mode}",
            color="#666688", fontsize=7.5, pad=5)
        self.ax_bev.set_xlim(-GRID_RANGE, GRID_RANGE)
        self.ax_bev.set_ylim(-GRID_RANGE, GRID_RANGE)
        self.ax_bev.set_xlabel("← links        rechts →", color="#444466", fontsize=7)
        self.ax_bev.set_ylabel("vorwärts →", color="#444466", fontsize=7)
        self.ax_bev.tick_params(colors="#222244", labelsize=6)

        # ── Kamerabild ──
        self.ax_cam.cla(); self.ax_cam.axis("off")
        if self.frame_idx < len(self.image_paths):
            img = cv2.imread(str(self.image_paths[self.frame_idx]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fname = self.image_paths[self.frame_idx].name
                for ann in self.pose_labels_cam.get(fname, []):
                    kps = np.array(ann["keypoints"], dtype=float).reshape(17,3)
                    tid = ann.get("track_id", 0)
                    h   = person_color(tid).lstrip("#")
                    col = (int(h[:2],16), int(h[2:4],16), int(h[4:],16))
                    for a, b in SKELETON_EDGES:
                        if kps[a,2]>0 and kps[b,2]>0:
                            cv2.line(img,(int(kps[a,0]),int(kps[a,1])),
                                     (int(kps[b,0]),int(kps[b,1])), col, 1)
                    for i in range(17):
                        if kps[i,2]>0:
                            cv2.circle(img,(int(kps[i,0]),int(kps[i,1])),3,col,-1)
                    vis = [(kps[i,0],kps[i,1]) for i in range(17) if kps[i,2]>0]
                    if vis:
                        tx = int(sum(q[0] for q in vis)/len(vis))
                        ty = int(min(q[1] for q in vis))-6
                        cv2.putText(img,f"#{tid}",(tx,max(ty,8)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1,cv2.LINE_AA)
                self.ax_cam.imshow(img, aspect="auto")
        self.ax_cam.set_title(f"Kamera: {CAMERAS[self.cam_idx]}", color="#444466", fontsize=7, pad=2)

        # ── Info-Panel ──
        self.ax_info.cla(); self.ax_info.axis("off")
        self.ax_info.set_facecolor("#0d0d18")

        lines = [
            ("Personen",  str(len(persons))),
            ("Modus",     self.mode),
            ("Prädiktion",f"{PRED_STEPS[-1]}s Horizont"),
            ("Grid",      f"{GRID_SIZE}px / {2*GRID_RANGE:.0f}m"),
        ]
        for i, (k, v) in enumerate(lines):
            y = 0.90 - i*0.18
            self.ax_info.text(0.04, y, k+":", color="#333355", fontsize=7.5,
                              va="top", transform=self.ax_info.transAxes)
            col = "#44ee55" if k == "Modus" and self.mode == MODE_GT else "#aaaacc"
            self.ax_info.text(0.04, y-0.09, v, color=col, fontsize=8,
                              va="top", fontweight="bold", transform=self.ax_info.transAxes)

        # Legende
        self.ax_info.text(0.04, 0.22, "Legende:", color="#333355", fontsize=7,
                          va="top", transform=self.ax_info.transAxes)
        legend = [("#ffffff", "Roboter"),
                  ("#e57373", "Person (Fußpunkt)"),
                  ("#e57373", "→  Richtungspfeil (Länge ∝ Speed)"),
                  ("#e5737360", "●  Prädiktionspunkte")]
        for j, (c, txt) in enumerate(legend):
            self.ax_info.text(0.04, 0.12 - j*0.08, f"  {txt}",
                              color=c, fontsize=6.5, va="top",
                              transform=self.ax_info.transAxes)

        self.fig.canvas.draw_idle()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_slider(self, val):
        self.frame_idx = int(val); self._draw()

    def _on_prev(self, _):
        self.playing = False
        self.frame_idx = max(0, self.frame_idx-1)
        self.slider.set_val(self.frame_idx)

    def _on_next(self, _):
        self.playing = False
        self.frame_idx = min(len(self.image_paths)-1, self.frame_idx+1)
        self.slider.set_val(self.frame_idx)

    def _on_play(self, _):
        self.playing = not self.playing

    def _on_scene_p(self, _):
        self.scene_idx = (self.scene_idx-1) % len(self.scenes)
        self._load_scene(); self._draw()

    def _on_scene_n(self, _):
        self.scene_idx = (self.scene_idx+1) % len(self.scenes)
        self._load_scene(); self._draw()

    def _on_cam(self, _):
        self.cam_idx = (self.cam_idx+1) % len(CAMERAS)
        self._switch_cam()

    def _on_cam_prev(self, _):
        self.cam_idx = (self.cam_idx-1) % len(CAMERAS)
        self._switch_cam()

    def _switch_cam(self):
        scene  = self.scenes[self.scene_idx]
        camera = CAMERAS[self.cam_idx]
        self.image_paths     = sorted((IMG_BASE / camera / scene).glob("*.jpg"))
        self.pose_labels_cam = self.pose_labels_all[camera]
        self._draw()

    def _on_mode(self, _):
        self.mode = MODE_GT if self.mode == MODE_COMBINED else MODE_COMBINED
        self.btn_mode.label.set_text(self.mode)
        col = "#44ee55" if self.mode == MODE_GT else "#4fc3f7"
        self.btn_mode.label.set_color(col)
        self._draw()

    def _on_key(self, event):
        if   event.key == " ":     self.playing = not self.playing
        elif event.key == "right": self._on_next(None)
        elif event.key == "left":  self._on_prev(None)
        elif event.key == "n":     self._on_scene_n(None)
        elif event.key == "p":     self._on_scene_p(None)
        elif event.key == "c":     self._on_cam(None)
        elif event.key == "m":     self._on_mode(None)

    # ── Hauptschleife ─────────────────────────────────────────────────────────

    def run(self):
        plt.show(block=False)
        while plt.fignum_exists(self.fig.number):
            if self.playing and self.image_paths:
                self.frame_idx = (self.frame_idx+1) % len(self.image_paths)
                self.slider.set_val(self.frame_idx)
                self._draw()
            self.fig.canvas.flush_events()
            time.sleep(1.0 / self.fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="")
    args = parser.parse_args()
    CostmapViewer(init_scene=args.scene).run()


if __name__ == "__main__":
    main()
