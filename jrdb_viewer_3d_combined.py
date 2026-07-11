"""
JRDB 3D Combined Orientation Viewer
=====================================
Zeigt pro Person bis zu 5 Richtungspfeile gleichzeitig:

  ORANGE  — Schulter-Asymmetrie  (|yaw| aus Pixelbreite, Vorzeichen aus Bild)
  CYAN    — Augen-Asymmetrie     (analog Schulter, aber Augenabstand)
  LILA    — Hüft-Richtung        (senkrecht zum 3D-Hüftvektor, Vorzeichen)
  GELB    — Velocity             (gewichtetes Delta aus Track-History)
  GRÜN    — Ground-Truth         (−rot_z + π, nur Evaluation, ein/ausblendbar)
  WEISS   — Combined             (zirkularer gewichteter Mittelwert der Quellen)

Gewichtung im Combined-Modus:
  Mit Velocity:    vel×0.6  + shldr×0.25 + eye×0.10 + hip×0.05
  Ohne Velocity:   shldr×0.60 + eye×0.25 + hip×0.15

Roboter steht in der Mitte, ±8 m Sichtbereich.

Aufruf:
    python jrdb_viewer_3d_combined.py
    python jrdb_viewer_3d_combined.py --scene stlc-111-2019-04-19_0 --camera image_0
"""

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Button, CheckButtons, Slider

# ── Pfade ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent / "jrdb"
IMG_BASE  = BASE / "images"
POSE_BASE = BASE / "labels" / "labels_2d_pose_coco"
L3D_BASE  = BASE / "labels" / "labels_3d"
CALIB_CAM = BASE / "calibration" / "cameras.yaml"
CALIB_LID = BASE / "calibration" / "lidars.yaml"

CAMERAS    = ["image_0", "image_2", "image_4", "image_6", "image_8"]
CAM_SUFFIX = {"image_0":"image0","image_2":"image2","image_4":"image4",
              "image_6":"image6","image_8":"image8"}
CAM_SENSOR = {"image_0":"sensor_0","image_2":"sensor_2","image_4":"sensor_4",
              "image_6":"sensor_6","image_8":"sensor_8"}

CAM_HEIGHT = 0.82
BEV_RANGE  = 8.0   # ±8 m um den Roboter

# ── Kamera-Perspektiven ───────────────────────────────────────────────────────
# (elev, azim, dist, label)
VIEW_PRESETS = [
    (35,  -60,  9.0,  "Iso"),        # 0 — isometrisch (Standard)
    (90,  -90,  8.5,  "Top ↓"),      # 1 — Vogelperspektive
    (10,  -90,  8.5,  "Frontal"),    # 2 — leicht schräg von vorne
    (10,    0,  8.5,  "Links →"),    # 3 — von links
    (10,  180,  8.5,  "Rechts ←"),   # 4 — von rechts
    (10,   90,  8.5,  "Hinten"),     # 5 — von hinten
]

# ── Pfeil-Farben ──────────────────────────────────────────────────────────────
COL_SHOULDER = "#ff8c00"   # orange
COL_EYE      = "#00d4ff"   # cyan
COL_HIP      = "#cc66ff"   # lila
COL_VELOCITY = "#f5e642"   # gelb
COL_GT       = "#44ee55"   # grün
COL_COMBINED = "#ffffff"   # weiß

ARROW_LENGTH = 1.2         # Meter

SHOULDER_WIDTH_NORM = 0.44
EYE_WIDTH_NORM      = 0.12

# Gewichte Combined
W_WITH_VEL    = {"vel": 0.60, "shldr": 0.25, "eye": 0.10, "hip": 0.05}
W_WITHOUT_VEL = {"shldr": 0.60, "eye": 0.25, "hip": 0.15}

# ── JRDB Skelett ─────────────────────────────────────────────────────────────
# idx: 0=head,1=r_eye,2=l_eye,3=r_shldr,4=c_shldr,5=l_shldr,
#      6=r_elbow,7=l_elbow,8=c_hip,9=r_wrist,10=r_hip,11=l_hip,...
SKELETON_EDGES = [
    (1,2),(0,4),(3,4),(8,10),(5,7),(10,13),(14,16),
    (4,5),(7,12),(4,8),(3,6),(13,15),(11,14),(6,9),(8,11),
]

PERSON_COLORS_HEX = [
    "#e57373","#f06292","#ba68c8","#9575cd",
    "#64b5f6","#4dd0e1","#4db6ac","#81c784",
    "#dce775","#ffb74d","#ff8a65","#a1887f",
]
def person_color(tid): return PERSON_COLORS_HEX[int(tid) % len(PERSON_COLORS_HEX)]

# ── Kalibrierung ──────────────────────────────────────────────────────────────

def load_calibration(camera):
    with open(CALIB_CAM) as f: cam_data = yaml.safe_load(f)
    with open(CALIB_LID) as f: lid_data = yaml.safe_load(f)
    sensor = CAM_SENSOR[camera]
    s = cam_data["cameras"][sensor]
    K       = np.array(list(map(float, s["K"].split()))).reshape(3,3)
    D       = np.array(list(map(float, s["D"].split())))
    cam2ego = np.array(lid_data[sensor]["cam2ego"])
    return K, D, cam2ego

# ── Datenladen ────────────────────────────────────────────────────────────────

def list_scenes():
    p = IMG_BASE / "image_0"
    return sorted(d.name for d in p.iterdir() if d.is_dir()) if p.exists() else []

def load_pose_labels(scene, camera):
    """Lädt Pose-Labels einer einzelnen Kamera."""
    path = POSE_BASE / f"{scene}_{CAM_SUFFIX[camera]}.json"
    if not path.exists(): return {}
    with open(path) as f: data = json.load(f)
    id2fn = {img["id"]: Path(img["file_name"]).name for img in data["images"]}
    out = {}
    for ann in data["annotations"]:
        fn = id2fn.get(ann["image_id"], "")
        if fn: out.setdefault(fn, []).append(ann)
    return out

def load_all_pose_labels(scene):
    """Lädt Pose-Labels aller 5 Kameras."""
    return {cam: load_pose_labels(scene, cam) for cam in CAMERAS}

def load_all_calibrations():
    """Gibt {camera: (K, D, cam2ego)} für alle 5 Kameras zurück."""
    return {cam: load_calibration(cam) for cam in CAMERAS}

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

# ── Orientierungsquellen ──────────────────────────────────────────────────────

def _kp_height(kps):
    vis = kps[kps[:,2] > 0]
    if len(vis) < 2: return None
    h = float(vis[:,1].max() - vis[:,1].min())
    return h if h > 10 else None

def yaw_shoulder(kps):
    """
    Schulter-Asymmetrie. Gibt (yaw, yaw_mirror, confidence) zurück.
    yaw       = wahrscheinlichste Richtung (aus Vorzeichen)
    yaw_mirror = gespiegelte Alternative (± yaw_mag) — bei Stillstand beide gültig
    """
    r_s, l_s = kps[3], kps[5]
    if r_s[2] == 0 or l_s[2] == 0: return None, None, 0.0
    kp_h = _kp_height(kps)
    if kp_h is None: return None, None, 0.0
    apparent_w = abs(r_s[0] - l_s[0])
    expected_w = kp_h * SHOULDER_WIDTH_NORM
    ratio   = float(np.clip(apparent_w / (expected_w + 1e-6), 0.0, 1.0))
    yaw_mag = np.arccos(ratio)
    sign    = 1.0 if r_s[0] < l_s[0] else -1.0
    conf    = min(1.0, apparent_w / (expected_w + 1e-6))
    return sign * yaw_mag, -sign * yaw_mag, conf

def yaw_eye(kps):
    """
    Augen-Asymmetrie. Gibt (yaw, yaw_mirror, confidence) zurück.
    """
    r_e, l_e = kps[1], kps[2]
    if r_e[2] == 0 or l_e[2] == 0: return None, None, 0.0
    kp_h = _kp_height(kps)
    if kp_h is None: return None, None, 0.0
    apparent_w = abs(r_e[0] - l_e[0])
    expected_w = kp_h * EYE_WIDTH_NORM
    ratio   = float(np.clip(apparent_w / (expected_w + 1e-6), 0.0, 1.0))
    yaw_mag = np.arccos(ratio)
    sign    = 1.0 if r_e[0] < l_e[0] else -1.0
    conf    = min(1.0, apparent_w / (expected_w + 1e-6))
    return sign * yaw_mag, -sign * yaw_mag, conf

def yaw_hip(pts3d):
    """
    Senkrechter zum 3D-Hüftvektor im XY-Raum.
    Da alle KPs auf einer YZ-Scheibe liegen (X konstant), gibt der Hüftvektor
    nur die Y-Lateralkomponente → Vorzeichen-Information.
    Liefert (yaw, confidence) — confidence ist niedrig (nur Vorzeichen).
    """
    r_hip, l_hip = pts3d[10], pts3d[11]
    if np.isnan(r_hip[0]) or np.isnan(l_hip[0]): return None, 0.0
    dY = r_hip[1] - l_hip[1]
    if abs(dY) < 0.005: return None, 0.0
    # Senkrecht zu (0, dY) in XY: Person schaut in ±X-Richtung
    # dY < 0 → r_hip weiter rechts (kleineres Y) → Person zeigt leicht nach rechts
    # → Blickrichtung hat negativen Y-Anteil → Vorzeichen negativ
    sign = -np.sign(dY)
    # Yaw ≈ 0 oder π, je nach Vorzeichen — keine Winkelinformation, nur Vorzeichen
    # Wir geben 0 oder π zurück, je nach ob Person zur Kamera schaut
    yaw = 0.0 if sign > 0 else np.pi
    conf = min(1.0, abs(dY) / 0.1)  # schwache Confidence
    return yaw, conf * 0.3  # Hüfte hat immer niedrige Confidence

def yaw_velocity(history):
    """Gewichtetes Delta aus Track-History. Gibt (yaw, confidence)."""
    if len(history) < 2: return None, 0.0
    recent = list(history)[-3:]
    deltas = [np.array(recent[i+1]) - np.array(recent[i])
              for i in range(len(recent)-1)]
    weights = list(range(1, len(deltas)+1))
    weighted = sum(w * d for w, d in zip(weights, deltas))
    speed = np.linalg.norm(weighted)
    if speed < 0.02: return None, 0.0
    conf = min(1.0, speed / 0.15)  # volle Confidence ab 0.15 m/frame
    return float(np.arctan2(weighted[1], weighted[0])), conf

def yaw_gt(label_3d):
    """Ground-Truth: -rot_z + π. Gibt (yaw, 1.0)."""
    if label_3d is None: return None, 0.0
    return -label_3d["box"]["rot_z"] + np.pi, 1.0

def combined_yaw(yaws_confs):
    """
    Zirkularer gewichteter Mittelwert über alle verfügbaren Quellen.
    yaws_confs: list von (yaw, weight, confidence) — weight ist der Basisgewicht.
    Effektivgewicht = weight * confidence.
    """
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

# ── Pfeil zeichnen ────────────────────────────────────────────────────────────

def draw_arrow_3d(ax, origin3d, yaw, color, length=ARROW_LENGTH, lw=2.0,
                  alpha=0.9, linestyle="-"):
    """Zeichnet einen Richtungspfeil aus origin3d in Yaw-Richtung."""
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    tip = origin3d + fwd * length
    ax.plot([origin3d[0], tip[0]], [origin3d[1], tip[1]], [origin3d[2], tip[2]],
            color=color, lw=lw, alpha=alpha, linestyle=linestyle)
    # Pfeilspitze nur bei ausgezogenen Linien
    if linestyle == "-":
        perp = np.array([-fwd[1], fwd[0], 0.0]) * 0.12
        back = tip - fwd * 0.16
        ax.plot([tip[0], back[0]+perp[0]], [tip[1], back[1]+perp[1]], [tip[2], back[2]+perp[2]],
                color=color, lw=lw, alpha=alpha)
        ax.plot([tip[0], back[0]-perp[0]], [tip[1], back[1]-perp[1]], [tip[2], back[2]-perp[2]],
                color=color, lw=lw, alpha=alpha)

def draw_3d_scene(ax, persons, show_gt, ground_z, combined_only=False):
    """
    persons: Liste von PersonData-Dicts mit allen Richtungsquellen.
    """
    ax.cla()
    ax.set_facecolor("#07070e")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#1a1a2e")
    ax.yaxis.pane.set_edgecolor("#1a1a2e")
    ax.zaxis.pane.set_edgecolor("#1a1a2e")
    ax.grid(True, color="#111122", linewidth=0.4)

    # Boden-Grid
    for v in np.arange(-BEV_RANGE, BEV_RANGE+1, 2):
        ax.plot([v,v], [-BEV_RANGE,BEV_RANGE], [ground_z,ground_z],
                color="#1a1a2e", lw=0.5, zorder=0)
        ax.plot([-BEV_RANGE,BEV_RANGE], [v,v], [ground_z,ground_z],
                color="#1a1a2e", lw=0.5, zorder=0)

    # Roboter + Vorwärtspfeil
    ax.scatter([0],[0],[ground_z], s=100, c="#ffffff", marker="^", zorder=10, depthshade=False)
    ax.quiver(0, 0, ground_z, 1.0, 0, 0, color="#4fc3f7", linewidth=1.5, arrow_length_ratio=0.3)

    for p in persons:
        pts3d    = p["pts3d"]
        color    = person_color(p["track_id"])

        # Knochen
        for a, b in SKELETON_EDGES:
            if not (np.isnan(pts3d[a,0]) or np.isnan(pts3d[b,0])):
                ax.plot([pts3d[a,0],pts3d[b,0]], [pts3d[a,1],pts3d[b,1]],
                        [pts3d[a,2],pts3d[b,2]], color=color, lw=1.5, alpha=0.85)

        # Keypoints
        valid = ~np.isnan(pts3d[:,0])
        ax.scatter(pts3d[valid,0], pts3d[valid,1], pts3d[valid,2],
                   s=10, c=color, depthshade=True, zorder=5)
        if not np.isnan(pts3d[0,0]):
            ax.scatter([pts3d[0,0]], [pts3d[0,1]], [pts3d[0,2]],
                       s=40, c=color, depthshade=True, zorder=6,
                       edgecolors="#ffffff", linewidths=0.5)

        # Körperachse Hüfte→Kopf (grau gestrichelt)
        hip, head = pts3d[8], pts3d[0]
        if not (np.isnan(hip[0]) or np.isnan(head[0])):
            ax.plot([hip[0],head[0]], [hip[1],head[1]], [hip[2],head[2]],
                    color="#555566", lw=1.0, ls="--", alpha=0.5)

        # Pfeile aus Hüftmittelpunkt
        origin = pts3d[8].copy()
        if np.isnan(origin[0]): continue

        # Einzelne Quellen — nur wenn nicht im combined_only Modus
        if not combined_only:
            standing = p.get("standing", True)

            if p["yaw_shoulder"] is not None:
                draw_arrow_3d(ax, origin, p["yaw_shoulder"], COL_SHOULDER, lw=2.0)
                # Bei Stillstand: gespiegelte Alternative gestrichelt — leicht versetzt in Z
                if standing and p["yaw_shoulder_m"] is not None:
                    draw_arrow_3d(ax, origin + np.array([0,0,0.08]), p["yaw_shoulder_m"],
                                  COL_SHOULDER, lw=1.8, alpha=0.55,
                                  linestyle="--")

            if p["yaw_eye"] is not None:
                draw_arrow_3d(ax, origin + np.array([0,0,0.18]), p["yaw_eye"], COL_EYE, lw=2.0)
                if standing and p["yaw_eye_m"] is not None:
                    draw_arrow_3d(ax, origin + np.array([0,0,0.26]), p["yaw_eye_m"],
                                  COL_EYE, lw=1.8, alpha=0.55,
                                  linestyle="--")

            if p["yaw_hip"] is not None:
                draw_arrow_3d(ax, origin + np.array([0,0,0.36]), p["yaw_hip"], COL_HIP, lw=1.8, alpha=0.75)
            if p["yaw_velocity"] is not None:
                draw_arrow_3d(ax, origin + np.array([0,0,0.46]), p["yaw_velocity"], COL_VELOCITY, lw=2.2)
            if show_gt and p["yaw_gt"] is not None:
                draw_arrow_3d(ax, origin + np.array([0,0,0.56]), p["yaw_gt"], COL_GT, lw=2.2)

        # Combined immer zeigen
        if p["yaw_combined"] is not None:
            z_off = 0.0 if combined_only else 0.66
            draw_arrow_3d(ax, origin + np.array([0,0,z_off]), p["yaw_combined"],
                          COL_COMBINED, lw=3.0 if combined_only else 2.5, alpha=1.0)

        # Im combined_only Modus: Alternativpfeil (Spiegel der Schulter) gestrichelt zeigen
        if combined_only and p.get("standing") and p["yaw_shoulder_m"] is not None:
            draw_arrow_3d(ax, origin + np.array([0,0,0.12]), p["yaw_shoulder_m"],
                          COL_SHOULDER, lw=1.8, alpha=0.55, linestyle="--")

        # ID-Label
        if not np.isnan(pts3d[0,0]):
            ax.text(pts3d[0,0], pts3d[0,1], pts3d[0,2]+0.12, f"#{p['track_id']}",
                    color=color, fontsize=6.5, ha="center", va="bottom", fontweight="bold")

    # Achsen und Limits — Roboter in der Mitte
    ax.set_xlim(-BEV_RANGE, BEV_RANGE)
    ax.set_ylim(-BEV_RANGE, BEV_RANGE)
    ax.set_zlim(ground_z, ground_z + 2.5)
    ax.set_xlabel("X vorwärts", color="#444466", fontsize=7, labelpad=4)
    ax.set_ylabel("Y links",    color="#444466", fontsize=7, labelpad=4)
    ax.set_zlabel("Z oben",     color="#444466", fontsize=7, labelpad=4)
    ax.tick_params(colors="#333355", labelsize=6)

# ── Viewer ────────────────────────────────────────────────────────────────────

class CombinedViewer3D:
    def __init__(self, init_scene="", init_camera="image_0"):
        self.scenes = list_scenes()
        if not self.scenes: raise RuntimeError("Keine Szenen gefunden")

        self.scene_idx  = self.scenes.index(init_scene) if init_scene in self.scenes else 0
        self.cam_idx    = CAMERAS.index(init_camera) if init_camera in CAMERAS else 0
        self.frame_idx      = 0
        self.playing        = False
        self.fps            = 6
        self.show_gt        = True
        self.combined_only  = False
        self.view_idx       = 0       # aktiver Preset-Index
        self.view_dist      = VIEW_PRESETS[0][2]  # Zoom-Distanz

        self.track_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=4))
        self.all_calibrations = load_all_calibrations()

        self._load_scene()
        self._build_ui()
        self._draw()

    # ── Laden ─────────────────────────────────────────────────────────────────

    def _load_scene(self):
        scene  = self.scenes[self.scene_idx]
        camera = CAMERAS[self.cam_idx]   # nur für Kamerabild rechts
        self.image_paths = sorted((IMG_BASE / camera / scene).glob("*.jpg"))
        # cam2ego / ground_z von sensor_0 (Referenz, alle Kameras transformieren in selben Ego-Frame)
        _, _, cam2ego_ref = self.all_calibrations["image_0"]
        self.ground_z = cam2ego_ref[2,3] - CAM_HEIGHT
        # Alle Kamera-Labels laden
        self.pose_labels_all = load_all_pose_labels(scene)
        # Einzelkamera für Kamerabild
        self.pose_labels_cam = self.pose_labels_all[camera]
        self.labels_3d = load_labels_3d(scene)
        self.track_hist.clear()
        self.frame_idx = 0
        if hasattr(self, "slider"):
            self.slider.valmax = max(len(self.image_paths)-1, 1)
            self.slider.set_val(0)
            self._update_title()

    def _get_persons(self):
        if not self.image_paths: return []
        fname = self.image_paths[self.frame_idx].name
        stem  = Path(fname).stem
        l3d_f = self.labels_3d.get(stem, {})
        result   = []
        seen_ids: set[int] = set()

        # Alle 5 Kameras auswerten — Deduplizierung via track_id
        for cam in CAMERAS:
            K, D, cam2ego = self.all_calibrations[cam]
            anns = self.pose_labels_all[cam].get(fname, [])
            for ann in anns:
                track_id = ann.get("track_id", 0)
                if track_id in seen_ids:
                    continue   # bereits von anderer Kamera erfasst

                kps   = np.array(ann["keypoints"], dtype=float).reshape(17,3)
                pts3d = project_keypoints_3d(kps, K, D, cam2ego)
                if pts3d is None: continue
                seen_ids.add(track_id)

                # Track-History aktualisieren
                feet = pts3d[[15,16]]
                vf   = feet[~np.isnan(feet[:,0])]
                if len(vf):
                    self.track_hist[track_id].append(vf.mean(axis=0)[:2].tolist())

                # Alle Quellen berechnen
                y_shldr, y_shldr_m, c_shldr = yaw_shoulder(kps)
                y_eye,   y_eye_m,   c_eye   = yaw_eye(kps)
                y_hip,               c_hip   = yaw_hip(pts3d)
                y_vel,               c_vel   = yaw_velocity(self.track_hist[track_id])
                y_gt,    _                   = yaw_gt(l3d_f.get(track_id))

                # Ob Person still steht (keine verlässliche Velocity)
                standing = (y_vel is None)

                # Combined
                if not standing:
                    w_plan = W_WITH_VEL
                    sources = [
                        (y_vel,   w_plan["vel"],   c_vel),
                        (y_shldr, w_plan["shldr"], c_shldr),
                        (y_eye,   w_plan["eye"],   c_eye),
                        (y_hip,   w_plan["hip"],   c_hip),
                    ]
                else:
                    w_plan = W_WITHOUT_VEL
                    sources = [
                        (y_shldr, w_plan["shldr"], c_shldr),
                        (y_eye,   w_plan["eye"],   c_eye),
                        (y_hip,   w_plan["hip"],   c_hip),
                    ]
                y_comb = combined_yaw(sources)

                result.append({
                    "track_id":       track_id,
                    "pts3d":          pts3d,
                    "yaw_shoulder":   y_shldr,
                    "yaw_shoulder_m": y_shldr_m,   # gespiegelte Alternative
                    "yaw_eye":        y_eye,
                    "yaw_eye_m":      y_eye_m,      # gespiegelte Alternative
                    "yaw_hip":        y_hip,
                    "yaw_velocity":   y_vel,
                    "yaw_gt":         y_gt,
                    "yaw_combined":   y_comb,
                    "standing":       standing,     # True = Velocity nicht verlässlich
                })
        return result

    # ── UI aufbauen ───────────────────────────────────────────────────────────

    def _build_ui(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9), facecolor="#07070e")
        self.fig.canvas.manager.set_window_title("JRDB 3D Combined Orientation Viewer")

        # 3D-Plot (links)
        self.ax3d = self.fig.add_subplot(121, projection="3d", computed_zorder=False)
        self.ax3d.set_position([0.01, 0.12, 0.58, 0.86])
        self.ax3d.set_facecolor("#07070e")

        # Kamerabild (rechts oben)
        self.ax_cam = self.fig.add_axes([0.62, 0.52, 0.36, 0.44])
        self.ax_cam.axis("off")

        # Info-Panel (rechts unten)
        self.ax_info = self.fig.add_axes([0.62, 0.12, 0.36, 0.37])
        self.ax_info.axis("off")
        self.ax_info.set_facecolor("#0d0d18")

        # Slider
        ax_sl = self.fig.add_axes([0.01, 0.105, 0.58, 0.025], facecolor="#0d0d18")
        n = len(self.image_paths)
        self.slider = Slider(ax_sl, "", 0, max(n-1,1), valinit=0, valstep=1, color="#4fc3f7")
        self.slider.label.set_color("#333355")
        self.slider.valtext.set_color("#666688")
        self.slider.on_changed(self._on_slider)

        bs = dict(color="#10101e", hovercolor="#1e1e38")

        # Frame-Buttons (über dem Slider, also y=0.065)
        self.btn_prev = Button(self.fig.add_axes([0.13, 0.065, 0.055, 0.036]), "◀◀", **bs)
        self.btn_play = Button(self.fig.add_axes([0.19, 0.065, 0.055, 0.036]), "▶ Play", **bs)
        self.btn_next = Button(self.fig.add_axes([0.25, 0.065, 0.055, 0.036]), "▶▶", **bs)
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            b.label.set_color("#aaaacc"); b.label.set_fontsize(9)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_play.on_clicked(self._on_play)
        self.btn_next.on_clicked(self._on_next)

        # Szene + Kamera
        self.btn_sp  = Button(self.fig.add_axes([0.62, 0.075, 0.08, 0.038]), "◀ Szene", **bs)
        self.btn_sn  = Button(self.fig.add_axes([0.71, 0.075, 0.08, 0.038]), "Szene ▶", **bs)
        self.btn_cam = Button(self.fig.add_axes([0.81, 0.075, 0.08, 0.038]), "Kamera ▶", **bs)
        for b in (self.btn_sp, self.btn_sn, self.btn_cam):
            b.label.set_color("#aaaacc"); b.label.set_fontsize(8)
        self.btn_sp.on_clicked(self._on_scene_p)
        self.btn_sn.on_clicked(self._on_scene_n)
        self.btn_cam.on_clicked(self._on_cam)

        # ── Perspektiv-Buttons (unter dem 3D-Plot) ──
        # 6 Presets als kleine Buttons in einer Reihe
        preset_labels = [p[3] for p in VIEW_PRESETS]
        self._view_btns = []
        btn_w = 0.58 / len(VIEW_PRESETS) - 0.005
        for i, label in enumerate(preset_labels):
            x = 0.01 + i * (btn_w + 0.005)
            b = Button(self.fig.add_axes([x, 0.01, btn_w, 0.038]), label,
                       color="#10101e", hovercolor="#1e1e38")
            b.label.set_fontsize(7.5)
            b.label.set_color("#aaaacc")
            b.on_clicked(lambda _, idx=i: self._on_view_preset(idx))
            self._view_btns.append(b)
        self._update_view_btn_colors()

        # Zoom via Scroll-Event
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        # GT Toggle
        self.btn_gt = Button(self.fig.add_axes([0.62, 0.025, 0.13, 0.038]),
                             "GT: AN", color="#10101e", hovercolor="#1e1e38")
        self.btn_gt.label.set_color(COL_GT); self.btn_gt.label.set_fontsize(8)
        self.btn_gt.on_clicked(self._on_gt_toggle)

        # Combined-only Toggle
        self.btn_combined = Button(self.fig.add_axes([0.76, 0.025, 0.22, 0.038]),
                                   "Pfeile: ALLE", color="#10101e", hovercolor="#1e1e38")
        self.btn_combined.label.set_color(COL_COMBINED); self.btn_combined.label.set_fontsize(8)
        self.btn_combined.on_clicked(self._on_combined_toggle)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._update_title()

    def _update_title(self):
        scene  = self.scenes[self.scene_idx]
        camera = CAMERAS[self.cam_idx]
        n      = len(self.image_paths)
        self.fig.canvas.manager.set_window_title(
            f"JRDB 3D Combined — {scene}  [{camera}]  {n} Frames")

    # ── Zeichnen ──────────────────────────────────────────────────────────────

    def _draw(self):
        persons = self._get_persons()

        # 3D-Ansicht: Preset hat Vorrang, sonst freie Maus-Position beibehalten
        elev = self.ax3d.elev
        azim = self.ax3d.azim
        draw_3d_scene(self.ax3d, persons, self.show_gt, self.ground_z, self.combined_only)
        self.ax3d.view_init(elev=elev, azim=azim)
        self.ax3d.dist = self.view_dist

        # Kamerabild
        camera = CAMERAS[self.cam_idx]
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
                    col = (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
                    for a, b in SKELETON_EDGES:
                        if kps[a,2]>0 and kps[b,2]>0:
                            cv2.line(img,(int(kps[a,0]),int(kps[a,1])),
                                     (int(kps[b,0]),int(kps[b,1])), col, 1)
                    for i in range(17):
                        if kps[i,2]>0:
                            cv2.circle(img,(int(kps[i,0]),int(kps[i,1])),3,col,-1)
                    vis = [(kps[i,0],kps[i,1]) for i in range(17) if kps[i,2]>0]
                    if vis:
                        tx = int(sum(p[0] for p in vis)/len(vis))
                        ty = int(min(p[1] for p in vis))-6
                        cv2.putText(img,f"#{tid}",(tx,max(ty,8)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1,cv2.LINE_AA)
                self.ax_cam.imshow(img, aspect="auto")
        self.ax_cam.set_title(f"Kamera: {camera}", color="#444466", fontsize=7, pad=2)

        # Info-Panel mit Legende
        self.ax_info.cla(); self.ax_info.axis("off")
        self.ax_info.set_facecolor("#0d0d18")
        scene = self.scenes[self.scene_idx]
        n     = len(self.image_paths)

        for i, (k, v) in enumerate([
            ("Szene",    scene[:26]),
            ("Kamera",   CAMERAS[self.cam_idx]),
            ("Frame",    f"{self.frame_idx+1} / {n}"),
            ("Personen", str(len(persons))),
        ]):
            y = 0.97 - i*0.13
            self.ax_info.text(0.03, y, k+":", color="#333355", fontsize=7,
                              va="top", transform=self.ax_info.transAxes)
            self.ax_info.text(0.03, y-0.07, v, color="#aaaacc", fontsize=8,
                              va="top", fontweight="bold",
                              transform=self.ax_info.transAxes)

        # Pfeil-Legende
        legend_items = [
            (COL_SHOULDER, "Schulter-Asymmetrie  (Pixel-Breite)"),
            (COL_EYE,      "Augen-Asymmetrie     (Pixel-Breite)"),
            (COL_HIP,      "Hüft-Richtung        (3D-Vorzeichen)"),
            (COL_VELOCITY, "Velocity             (Track-History)"),
            (COL_GT,       f"Ground-Truth         ({'AN' if self.show_gt else 'AUS'})"),
            (COL_COMBINED, "Combined             (gewichtet)"),
        ]
        self.ax_info.text(0.03, 0.44, "Pfeile:", color="#555566", fontsize=7,
                          va="top", transform=self.ax_info.transAxes)
        for j, (col, label) in enumerate(legend_items):
            y = 0.38 - j * 0.09
            self.ax_info.add_patch(mpatches.Rectangle(
                (0.03, y-0.025), 0.06, 0.04,
                facecolor=col, transform=self.ax_info.transAxes,
                clip_on=False, alpha=0.85))
            gt_dim = 0.5 if (col == COL_GT and not self.show_gt) else 1.0
            self.ax_info.text(0.12, y, label, color=col, fontsize=6.5,
                              va="center", transform=self.ax_info.transAxes,
                              alpha=gt_dim)

        # Gewichte anzeigen
        self.ax_info.text(0.03, -0.15, "Gewichte (Combined):", color="#333355", fontsize=6.5,
                          va="top", transform=self.ax_info.transAxes)
        self.ax_info.text(0.03, -0.22,
                          "Mit Vel: vel×0.60 · shldr×0.25 · eye×0.10 · hip×0.05",
                          color="#555566", fontsize=6, va="top",
                          transform=self.ax_info.transAxes)
        self.ax_info.text(0.03, -0.29,
                          "Ohne Vel: shldr×0.60 · eye×0.25 · hip×0.15",
                          color="#555566", fontsize=6, va="top",
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

    def _on_gt_toggle(self, _):
        self.show_gt = not self.show_gt
        self.btn_gt.label.set_text(f"GT: {'AN' if self.show_gt else 'AUS'}")
        self._draw()

    def _on_combined_toggle(self, _):
        self.combined_only = not self.combined_only
        label = "Pfeile: NUR Combined" if self.combined_only else "Pfeile: ALLE"
        self.btn_combined.label.set_text(label)
        self._draw()

    def _on_cam(self, _):
        self.cam_idx = (self.cam_idx+1) % len(CAMERAS)
        # Kamerabild-Labels neu setzen
        scene = self.scenes[self.scene_idx]
        camera = CAMERAS[self.cam_idx]
        self.image_paths = sorted((IMG_BASE / camera / scene).glob("*.jpg"))
        self.pose_labels_cam = self.pose_labels_all[camera]
        if hasattr(self, "slider"):
            self.slider.valmax = max(len(self.image_paths)-1, 1)
            self.slider.set_val(self.frame_idx)
        self._update_title()
        self._draw()

    def _on_view_preset(self, idx):
        """Wechselt zu einem Perspektiv-Preset."""
        self.view_idx  = idx
        elev, azim, dist, _ = VIEW_PRESETS[idx]
        self.view_dist = dist
        self.ax3d.view_init(elev=elev, azim=azim)
        self.ax3d.dist = dist
        self._update_view_btn_colors()
        self.fig.canvas.draw_idle()

    def _update_view_btn_colors(self):
        for i, b in enumerate(self._view_btns):
            b.label.set_color("#ffffff" if i == self.view_idx else "#666688")
            b.ax.set_facecolor("#1e2a3a" if i == self.view_idx else "#10101e")

    def _on_scroll(self, event):
        """Zoom via Mausrad — nur wenn Maus über dem 3D-Plot."""
        if event.inaxes != self.ax3d: return
        factor = 0.9 if event.button == "up" else 1.1
        self.view_dist = max(3.0, min(20.0, self.view_dist * factor))
        self.ax3d.dist = self.view_dist
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if   event.key == " ":     self.playing = not self.playing
        elif event.key == "right": self._on_next(None)
        elif event.key == "left":  self._on_prev(None)
        elif event.key == "n":     self._on_scene_n(None)
        elif event.key == "p":     self._on_scene_p(None)
        elif event.key == "c":     self._on_cam(None)
        elif event.key == "g":     self._on_gt_toggle(None)
        elif event.key == "v":     self._on_combined_toggle(None)
        # Perspektiven per Nummertasten 1–6
        elif event.key in [str(i+1) for i in range(len(VIEW_PRESETS))]:
            self._on_view_preset(int(event.key) - 1)

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
    parser.add_argument("--scene",  default="")
    parser.add_argument("--camera", default="image_0", choices=CAMERAS)
    args = parser.parse_args()
    CombinedViewer3D(init_scene=args.scene, init_camera=args.camera).run()


if __name__ == "__main__":
    main()
