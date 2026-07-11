"""
JRDB Bird's-Eye-View Viewer
Zeigt ALLE Personen einer Szene in der Vogelperspektive — aus allen 5 Kameras
gleichzeitig sowie aus dem stitched 360°-Panorama.

Modi:
  • "Alle Kameras" — alle 5 Einzelkameras werden gleichzeitig ausgewertet,
    Duplikate über track_id dedupliziert
  • "Stitched 360°" — zylindrische Projektion des Panoramabildes

Aufruf:
    python jrdb_viewer_bev.py
    python jrdb_viewer_bev.py --scene stlc-111-2019-04-19_0
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.widgets import Button, CheckButtons, Slider

# ── Pfade ─────────────────────────────────────────────────────────────────────
BASE            = Path(__file__).parent / "jrdb"
IMG_BASE        = BASE / "images"
POSE_BASE       = BASE / "labels" / "labels_2d_pose_coco"
POSE_STITCH_BASE= BASE / "labels" / "labels_2d_pose_stitched_coco"
CALIB_CAM       = BASE / "calibration" / "cameras.yaml"
CALIB_LID       = BASE / "calibration" / "lidars.yaml"

SINGLE_CAMS = ["image_0", "image_2", "image_4", "image_6", "image_8"]
CAM_SUFFIX  = {"image_0":"image0","image_2":"image2","image_4":"image4",
               "image_6":"image6","image_8":"image8"}
CAM_SENSOR  = {"image_0":"sensor_0","image_2":"sensor_2","image_4":"sensor_4",
               "image_6":"sensor_6","image_8":"sensor_8"}

CAM_HEIGHT   = 0.82   # Meter Kamerahöhe über Boden
BEV_RANGE    = 8.0   # ±13 m in X und Y

# Stitched-Parameter (cylindrische Projektion)
W_STITCH   = 3760
OFFSET_X   = 1880     # Pixel x=1880 = Vorwärtsrichtung
FY_STITCH  = 484.0    # gemittelte fy aller 5 Kameras
CY_STITCH  = 240.0    # Bildhöhe/2

# ── Skelett ───────────────────────────────────────────────────────────────────
SKELETON_EDGES = [
    (1,2),(0,4),(3,4),(8,10),(5,7),(10,13),(14,16),
    (4,5),(7,12),(4,8),(3,6),(13,15),(11,14),(6,9),(8,11),
]

PERSON_COLORS_HEX = [
    "#e57373","#f06292","#ba68c8","#9575cd",
    "#64b5f6","#4dd0e1","#4db6ac","#81c784",
    "#dce775","#ffb74d","#ff8a65","#a1887f",
]

def person_color(track_id: int) -> str:
    """Hex-Farbe für BEV-Plot."""
    return PERSON_COLORS_HEX[int(track_id) % len(PERSON_COLORS_HEX)]

def person_color_bgr(track_id: int) -> tuple[int, int, int]:
    """RGB-Farbe für OpenCV-Zeichenfunktionen auf einem bereits RGB-konvertierten Bild."""
    h = PERSON_COLORS_HEX[int(track_id) % len(PERSON_COLORS_HEX)].lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b)  # Bild ist nach cv2.cvtColor bereits RGB


# ── Kalibrierung ──────────────────────────────────────────────────────────────

def load_all_calibrations() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Gibt {camera_name: (K, D, cam2ego)} für alle 5 Einzelkameras zurück."""
    with open(CALIB_CAM) as f:
        cam_data = yaml.safe_load(f)
    with open(CALIB_LID) as f:
        lid_data = yaml.safe_load(f)
    result = {}
    for cam in SINGLE_CAMS:
        sensor = CAM_SENSOR[cam]
        s = cam_data["cameras"][sensor]
        K = np.array(list(map(float, s["K"].split()))).reshape(3, 3)
        D = np.array(list(map(float, s["D"].split())))
        cam2ego = np.array(lid_data[sensor]["cam2ego"])
        result[cam] = (K, D, cam2ego)
    return result


# ── Datenladen ────────────────────────────────────────────────────────────────

def list_scenes() -> list[str]:
    p = IMG_BASE / "image_0"
    return sorted(d.name for d in p.iterdir() if d.is_dir()) if p.exists() else []


def load_pose_single(scene: str, camera: str) -> dict[str, list]:
    """frame_filename → [annotations]"""
    path = POSE_BASE / f"{scene}_{CAM_SUFFIX[camera]}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    id2fn = {img["id"]: Path(img["file_name"]).name for img in data["images"]}
    out: dict[str, list] = {}
    for ann in data["annotations"]:
        fn = id2fn.get(ann["image_id"], "")
        if fn:
            out.setdefault(fn, []).append(ann)
    return out


def load_pose_stitched(scene: str) -> dict[str, list]:
    """frame_filename → [annotations]  (aus stitched-JSON)"""
    path = POSE_STITCH_BASE / f"{scene}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    id2fn = {img["id"]: Path(img["file_name"]).name for img in data["images"]}
    out: dict[str, list] = {}
    for ann in data["annotations"]:
        fn = id2fn.get(ann["image_id"], "")
        if fn:
            out.setdefault(fn, []).append(ann)
    return out


def load_image_paths(scene: str) -> dict[str, list[Path]]:
    """camera → sortierte Bildpfade"""
    result = {}
    for cam in SINGLE_CAMS:
        d = IMG_BASE / cam / scene
        result[cam] = sorted(d.glob("*.jpg")) if d.exists() else []
    d_stitch = IMG_BASE / "image_stitched" / scene
    result["image_stitched"] = sorted(d_stitch.glob("*.jpg")) if d_stitch.exists() else []
    return result


# ── Projektion: Einzelkamera ──────────────────────────────────────────────────

def pixel_ray_ego(u, v, K, D, cam2ego):
    pts = np.array([[[u, v]]], dtype=np.float32)
    pts_r = cv2.undistortPoints(pts, K, D, P=K)
    u_r, v_r = pts_r[0, 0]
    ray_cam = np.array([(u_r-K[0,2])/K[0,0], (v_r-K[1,2])/K[1,1], 1.0])
    return cam2ego[:3,3].copy(), cam2ego[:3,:3] @ ray_cam


def foot_depth_single(u, v, K, D, cam2ego) -> float | None:
    origin, ray = pixel_ray_ego(u, v, K, D, cam2ego)
    gz = origin[2] - CAM_HEIGHT
    if abs(ray[2]) < 1e-6: return None
    t = (gz - origin[2]) / ray[2]
    return t if t > 0.1 else None


def project_kps_single(kps: np.ndarray, K, D, cam2ego) -> np.ndarray | None:
    """Projiziert 17 Keypoints einer Einzelkamera in Ego-XY. Gibt (17,2) zurück."""
    depths = [foot_depth_single(kps[fi,0], kps[fi,1], K, D, cam2ego)
              for fi in [15,16] if kps[fi,2] > 0]
    depths = [d for d in depths if d is not None]
    if not depths:
        return None
    t = float(np.mean(depths))
    origin, _ = pixel_ray_ego(kps[0,0], kps[0,1], K, D, cam2ego)

    pts = np.full((17,2), np.nan)
    for i in range(17):
        if kps[i,2] > 0:
            _, ray = pixel_ray_ego(kps[i,0], kps[i,1], K, D, cam2ego)
            p = origin + t * ray
            pts[i] = [p[0], p[1]]

    _interpolate_missing(pts)
    return pts if not np.all(np.isnan(pts[:,0])) else None


# ── Projektion: Stitched 360° ─────────────────────────────────────────────────

def pixel_ray_ego_stitch(x_s: float, y_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Zylindrische Rückprojektion für das stitched Panoramabild.
    theta = Azimut im Ego-Frame (0 = vorwärts, positiv = links)
    """
    theta   = (x_s - OFFSET_X) * (2 * np.pi / W_STITCH)
    y_norm  = (y_s - CY_STITCH) / FY_STITCH
    # Strahl: Ego X=vorwärts, Y=links, Z=oben
    ray_ego = np.array([np.cos(theta), np.sin(theta), -y_norm])
    origin  = np.array([0.0, 0.0, CAM_HEIGHT])
    return origin, ray_ego


def foot_depth_stitch(x_s: float, y_s: float) -> float | None:
    origin, ray = pixel_ray_ego_stitch(x_s, y_s)
    if abs(ray[2]) < 1e-6: return None
    t = -origin[2] / ray[2]
    return t if t > 0.1 else None


def project_kps_stitch(kps: np.ndarray) -> np.ndarray | None:
    """Projiziert 17 Keypoints aus dem stitched-Bild in Ego-XY. Gibt (17,2) zurück."""
    depths = [foot_depth_stitch(kps[fi,0], kps[fi,1])
              for fi in [15,16] if kps[fi,2] > 0]
    depths = [d for d in depths if d is not None]
    if not depths:
        return None
    t = float(np.mean(depths))
    origin, _ = pixel_ray_ego_stitch(kps[0,0], kps[0,1])

    pts = np.full((17,2), np.nan)
    for i in range(17):
        if kps[i,2] > 0:
            _, ray = pixel_ray_ego_stitch(kps[i,0], kps[i,1])
            p = origin + t * ray
            pts[i] = [p[0], p[1]]

    _interpolate_missing(pts)
    return pts if not np.all(np.isnan(pts[:,0])) else None


# ── Hilfsfunktion ─────────────────────────────────────────────────────────────

def _interpolate_missing(pts: np.ndarray):
    """Fehlende Keypoints (NaN) aus Skelett-Nachbarn interpolieren. In-place."""
    for i in range(17):
        if np.isnan(pts[i,0]):
            nb = [j for (a,b) in SKELETON_EDGES
                  for j in ([b] if a==i else [a] if b==i else [])
                  if not np.isnan(pts[j,0])]
            if nb:
                pts[i] = np.nanmean([pts[j] for j in nb], axis=0)
    known = pts[~np.isnan(pts[:,0])]
    if len(known):
        c = known.mean(axis=0)
        for i in range(17):
            if np.isnan(pts[i,0]):
                pts[i] = c


# ── BEV zeichnen ──────────────────────────────────────────────────────────────

def draw_bev(ax, persons: list[tuple[np.ndarray, int]], show_labels: bool):
    ax.cla()
    ax.set_facecolor("#0a0a0f")

    # Grid 1m
    for v in np.arange(-BEV_RANGE, BEV_RANGE+1, 1.0):
        ax.axhline(v, color="#111122", lw=0.4, zorder=0)
        ax.axvline(v, color="#111122", lw=0.4, zorder=0)
    # Grid 5m
    for v in np.arange(-BEV_RANGE, BEV_RANGE+1, 5.0):
        ax.axhline(v, color="#1a1a33", lw=0.9, zorder=0)
        ax.axvline(v, color="#1a1a33", lw=0.9, zorder=0)

    # Abstandsringe
    for r in [2, 4, 6, 8, 10, 12]:
        ax.add_patch(plt.Circle((0,0), r, color="#151530", fill=False,
                                lw=0.7, ls="--", zorder=1))
        ax.text(0.08, r+0.12, f"{r}m", color="#2a2a55", fontsize=6,
                ha="left", va="bottom", zorder=1)

    # Roboter
    ax.scatter(0, 0, s=150, c="#ffffff", marker="^", zorder=10, lw=0)
    ax.text(0.2, 0.15, "Robot", color="#888888", fontsize=7, zorder=10)
    ax.annotate("", xy=(0, 1.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=1.8))

    # Personen
    for pts_xy, track_id in persons:
        color = person_color(track_id)

        # BEV-Mapping: plot_x = -Y_ego, plot_y = X_ego
        px = -pts_xy[:,1]
        py =  pts_xy[:,0]

        # Knochen
        for a, b in SKELETON_EDGES:
            if not (np.isnan(px[a]) or np.isnan(px[b])):
                ax.plot([px[a], px[b]], [py[a], py[b]],
                        color=color, lw=1.8, alpha=0.85, zorder=5)

        # Keypoints
        v = ~np.isnan(px)
        ax.scatter(px[v], py[v], s=16, c=color, zorder=6, lw=0)

        # Kopf
        if not np.isnan(px[0]):
            ax.scatter(px[0], py[0], s=50, c=color, zorder=7,
                       lw=0.8, edgecolors="#ffffff")

        # Blickrichtungspfeil aus Schultervektor
        r_s, l_s = pts_xy[3], pts_xy[5]
        if not (np.isnan(r_s[0]) or np.isnan(l_s[0])):
            mid_x = (-r_s[1] + -l_s[1]) / 2
            mid_y = ( r_s[0] +  l_s[0]) / 2
            dsx   = -(r_s[1] - l_s[1])
            dsy   =  (r_s[0] - l_s[0])
            fwd_x, fwd_y = -dsy, dsx
            n = np.hypot(fwd_x, fwd_y)
            if n > 0.01:
                fwd_x /= n; fwd_y /= n
                ax.annotate("", xy=(mid_x + fwd_x*0.45, mid_y + fwd_y*0.45),
                             xytext=(mid_x, mid_y),
                             arrowprops=dict(arrowstyle="->", color=color,
                                             lw=1.1, alpha=0.75))

        # ID-Label
        if show_labels:
            vi = np.where(~np.isnan(px))[0]
            if len(vi):
                ti = vi[np.argmax(py[vi])]
                ax.text(px[ti], py[ti]+0.2, f"#{track_id}",
                        color=color, fontsize=6.5, ha="center", va="bottom",
                        fontweight="bold", zorder=8)

    ax.set_xlim(-BEV_RANGE, BEV_RANGE)
    ax.set_ylim(-BEV_RANGE, BEV_RANGE)
    ax.set_aspect("equal")
    ax.set_xlabel("← links           rechts →", color="#444466", fontsize=7)
    ax.set_ylabel("vorwärts →", color="#444466", fontsize=7)
    ax.tick_params(colors="#222244", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1a33")


# ── Viewer ────────────────────────────────────────────────────────────────────

# Anzeigemodi
MODE_ALL    = "Alle Kameras"
MODE_STITCH = "Stitched 360°"
MODES = [MODE_ALL, MODE_STITCH]


class BEVViewer:
    def __init__(self, init_scene: str = ""):
        self.scenes    = list_scenes()
        if not self.scenes:
            raise RuntimeError(f"Keine Szenen unter {IMG_BASE}")

        self.scene_idx  = self.scenes.index(init_scene) if init_scene in self.scenes else 0
        self.mode_idx   = 0          # 0=alle Kameras, 1=stitched
        self.cam_idx    = 0          # aktive Kamera für das Kamerabild
        self.frame_idx  = 0
        self.playing    = False
        self.fps        = 8
        self.show_labels = True

        # Alle Kameras inkl. stitched für das Kamerabild
        self.cam_display = SINGLE_CAMS + ["image_stitched"]

        self.calibrations = load_all_calibrations()

        # Daten für aktuelle Szene
        self.image_paths: dict[str, list[Path]] = {}
        self.pose_single: dict[str, dict] = {}   # cam → {frame→[anns]}
        self.pose_stitch: dict[str, list] = {}    # frame → [anns]
        self.n_frames = 0

        self._load_scene()
        self._build_ui()
        self._draw()

    # ── Laden ─────────────────────────────────────────────────────────────────

    def _load_scene(self):
        scene = self.scenes[self.scene_idx]
        self.image_paths = load_image_paths(scene)

        self.pose_single = {cam: load_pose_single(scene, cam) for cam in SINGLE_CAMS}
        self.pose_stitch = load_pose_stitched(scene)

        # Anzahl Frames aus image_0
        self.n_frames = len(self.image_paths.get("image_0", []))
        self.frame_idx = 0

        if hasattr(self, "slider"):
            self.slider.valmax = max(self.n_frames - 1, 1)
            self.slider.set_val(0)
            self._update_title()

    def _get_persons(self, frame_idx: int) -> list[tuple[np.ndarray, int]]:
        """Gibt Liste (pts_xy, track_id) für aktuellen Frame zurück."""
        # Dateiname aus image_0 als Referenz
        paths_0 = self.image_paths.get("image_0", [])
        if frame_idx >= len(paths_0):
            return []
        fname = paths_0[frame_idx].name

        mode = MODES[self.mode_idx]
        persons: list[tuple[np.ndarray, int]] = []
        seen_ids: set[int] = set()

        if mode == MODE_ALL:
            for cam in SINGLE_CAMS:
                K, D, cam2ego = self.calibrations[cam]
                anns = self.pose_single[cam].get(fname, [])
                for ann in anns:
                    tid  = ann.get("track_id", 0)
                    kps  = np.array(ann["keypoints"], dtype=float).reshape(17, 3)
                    pts  = project_kps_single(kps, K, D, cam2ego)
                    if pts is not None and tid not in seen_ids:
                        persons.append((pts, tid))
                        seen_ids.add(tid)

        else:  # MODE_STITCH
            anns = self.pose_stitch.get(fname, [])
            for ann in anns:
                tid = ann.get("track_id", 0)
                kps = np.array(ann["keypoints"], dtype=float).reshape(17, 3)
                pts = project_kps_stitch(kps)
                if pts is not None and tid not in seen_ids:
                    persons.append((pts, tid))
                    seen_ids.add(tid)

        return persons

    # ── UI aufbauen ───────────────────────────────────────────────────────────

    def _build_ui(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(14, 9), facecolor="#07070e")
        self.fig.canvas.manager.set_window_title("JRDB Bird's-Eye-View")

        # BEV-Achse
        self.ax_bev = self.fig.add_axes([0.02, 0.13, 0.61, 0.84])

        # Kamerabild
        self.ax_cam = self.fig.add_axes([0.65, 0.52, 0.33, 0.44])
        self.ax_cam.axis("off")

        # Info-Panel
        self.ax_info = self.fig.add_axes([0.65, 0.13, 0.33, 0.36])
        self.ax_info.axis("off")
        self.ax_info.set_facecolor("#0d0d18")

        # Slider
        ax_sl = self.fig.add_axes([0.02, 0.07, 0.61, 0.025], facecolor="#0d0d18")
        self.slider = Slider(ax_sl, "", 0, max(self.n_frames-1, 1),
                             valinit=0, valstep=1, color="#4fc3f7")
        self.slider.label.set_color("#333355")
        self.slider.valtext.set_color("#666688")
        self.slider.on_changed(self._on_slider)

        bs = dict(color="#10101e", hovercolor="#1e1e38")

        # Frame-Buttons
        self.btn_prev = Button(self.fig.add_axes([0.17, 0.015, 0.055, 0.042]), "◀◀", **bs)
        self.btn_play = Button(self.fig.add_axes([0.23, 0.015, 0.055, 0.042]), "▶ Play", **bs)
        self.btn_next = Button(self.fig.add_axes([0.29, 0.015, 0.055, 0.042]), "▶▶", **bs)
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            b.label.set_color("#aaaacc"); b.label.set_fontsize(9)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_play.on_clicked(self._on_play)
        self.btn_next.on_clicked(self._on_next)

        # Szene-Buttons
        self.btn_sp = Button(self.fig.add_axes([0.65, 0.075, 0.075, 0.038]), "◀ Szene", **bs)
        self.btn_sn = Button(self.fig.add_axes([0.73, 0.075, 0.075, 0.038]), "Szene ▶", **bs)
        for b in (self.btn_sp, self.btn_sn):
            b.label.set_color("#aaaacc"); b.label.set_fontsize(8)
        self.btn_sp.on_clicked(self._on_scene_p)
        self.btn_sn.on_clicked(self._on_scene_n)

        # Modus-Button
        self.btn_mode = Button(self.fig.add_axes([0.82, 0.075, 0.14, 0.038]),
                               MODES[self.mode_idx], **bs)
        self.btn_mode.label.set_color("#4fc3f7"); self.btn_mode.label.set_fontsize(8)
        self.btn_mode.on_clicked(self._on_mode)

        # Kamera-Auswahl für das Kamerabild (◀ / Label / ▶)
        self.btn_cam_p = Button(self.fig.add_axes([0.65, 0.495, 0.04, 0.028]), "◀", **bs)
        self.btn_cam_n = Button(self.fig.add_axes([0.955, 0.495, 0.04, 0.028]), "▶", **bs)
        for b in (self.btn_cam_p, self.btn_cam_n):
            b.label.set_color("#aaaacc"); b.label.set_fontsize(9)
        self.btn_cam_p.on_clicked(self._on_cam_p)
        self.btn_cam_n.on_clicked(self._on_cam_n)

        # Label für aktive Kamera (wird in _update_title gesetzt)
        self.ax_cam_label = self.fig.add_axes([0.69, 0.495, 0.265, 0.028])
        self.ax_cam_label.axis("off")
        self.ax_cam_label.set_facecolor("#10101e")
        self.txt_cam_label = self.ax_cam_label.text(
            0.5, 0.5, "", color="#4fc3f7", fontsize=8,
            ha="center", va="center", transform=self.ax_cam_label.transAxes,
            fontweight="bold")

        # Toggle IDs
        ax_chk = self.fig.add_axes([0.65, 0.015, 0.14, 0.05], facecolor="#0d0d18")
        self.check = CheckButtons(ax_chk, ["IDs anzeigen"], [self.show_labels])
        self.check.labels[0].set_color("#888899"); self.check.labels[0].set_fontsize(8)
        self.check.on_clicked(self._on_toggle)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._update_title()

    def _update_title(self):
        scene = self.scenes[self.scene_idx]
        mode  = MODES[self.mode_idx]
        cam   = self.cam_display[self.cam_idx]
        self.fig.canvas.manager.set_window_title(
            f"JRDB BEV — {scene}  [{mode}]  {self.n_frames} Frames")
        if hasattr(self, "btn_mode"):
            self.btn_mode.label.set_text(mode)
        if hasattr(self, "txt_cam_label"):
            self.txt_cam_label.set_text(cam)

    # ── Zeichnen ──────────────────────────────────────────────────────────────

    def _draw(self):
        persons = self._get_persons(self.frame_idx)
        draw_bev(self.ax_bev, persons, self.show_labels)

        scene = self.scenes[self.scene_idx]
        mode  = MODES[self.mode_idx]
        self.ax_bev.set_title(
            f"{scene}   [{mode}]   Frame {self.frame_idx+1}/{self.n_frames}"
            f"   |   {len(persons)} Personen",
            color="#555577", fontsize=8, pad=5)

        # Kamerabild — gewählte Kamera
        active_cam = self.cam_display[self.cam_idx]
        self.ax_cam.cla(); self.ax_cam.axis("off")
        cam_paths = self.image_paths.get(active_cam, [])
        if self.frame_idx < len(cam_paths):
            img = cv2.imread(str(cam_paths[self.frame_idx]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fname = cam_paths[self.frame_idx].name
                # Pose-Annotations für diese Kamera laden
                if active_cam in SINGLE_CAMS:
                    anns = self.pose_single[active_cam].get(fname, [])
                else:  # stitched
                    anns = self.pose_stitch.get(fname, [])
                for ann in anns:
                    kps = np.array(ann["keypoints"], dtype=float).reshape(17,3)
                    tid = ann.get("track_id", 0)
                    col = person_color_bgr(tid)
                    for a, b in SKELETON_EDGES:
                        if kps[a,2] > 0 and kps[b,2] > 0:
                            cv2.line(img, (int(kps[a,0]),int(kps[a,1])),
                                     (int(kps[b,0]),int(kps[b,1])), col, 1)
                    for i in range(17):
                        if kps[i,2] > 0:
                            cv2.circle(img, (int(kps[i,0]),int(kps[i,1])), 3, col, -1)
                    # ID-Label über dem obersten sichtbaren Keypoint
                    visible = [(kps[i,0], kps[i,1]) for i in range(17) if kps[i,2] > 0]
                    if visible:
                        tx = int(sum(p[0] for p in visible) / len(visible))
                        ty = int(min(p[1] for p in visible)) - 6
                        cv2.putText(img, f"#{tid}", (tx, max(ty, 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1,
                                    cv2.LINE_AA)
                self.ax_cam.imshow(img, aspect="auto")
        self.ax_cam.set_title(f"Kamera: {active_cam}", color="#444466", fontsize=7, pad=2)

        # Info
        self.ax_info.cla(); self.ax_info.axis("off")
        self.ax_info.set_facecolor("#0d0d18")
        for i, (k, v) in enumerate([
            ("Szene",   scene[:30]),
            ("Modus",   mode),
            ("Frame",   f"{self.frame_idx+1} / {self.n_frames}"),
            ("Personen",f"{len(persons)}"),
        ]):
            y = 0.88 - i*0.20
            self.ax_info.text(0.04, y,   k+":", color="#333355", fontsize=7.5,
                              va="top", transform=self.ax_info.transAxes)
            self.ax_info.text(0.04, y-0.10, v, color="#aaaacc", fontsize=8,
                              va="top", fontweight="bold",
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
        self.frame_idx = min(self.n_frames-1, self.frame_idx+1)
        self.slider.set_val(self.frame_idx)

    def _on_play(self, _):
        self.playing = not self.playing

    def _on_scene_p(self, _):
        self.scene_idx = (self.scene_idx-1) % len(self.scenes)
        self._load_scene(); self._draw()

    def _on_scene_n(self, _):
        self.scene_idx = (self.scene_idx+1) % len(self.scenes)
        self._load_scene(); self._draw()

    def _on_mode(self, _):
        self.mode_idx = (self.mode_idx+1) % len(MODES)
        self._update_title(); self._draw()

    def _on_cam_p(self, _):
        self.cam_idx = (self.cam_idx - 1) % len(self.cam_display)
        self._update_title(); self._draw()

    def _on_cam_n(self, _):
        self.cam_idx = (self.cam_idx + 1) % len(self.cam_display)
        self._update_title(); self._draw()

    def _on_toggle(self, _):
        self.show_labels = not self.show_labels; self._draw()

    def _on_key(self, event):
        if   event.key == " ":      self.playing = not self.playing
        elif event.key == "right":  self._on_next(None)
        elif event.key == "left":   self._on_prev(None)
        elif event.key == "n":      self._on_scene_n(None)
        elif event.key == "p":      self._on_scene_p(None)
        elif event.key == "m":      self._on_mode(None)
        elif event.key == "k":      self._on_cam_n(None)

    # ── Hauptschleife ─────────────────────────────────────────────────────────

    def run(self):
        plt.show(block=False)
        while plt.fignum_exists(self.fig.number):
            if self.playing and self.n_frames > 0:
                self.frame_idx = (self.frame_idx+1) % self.n_frames
                self.slider.set_val(self.frame_idx)
                self._draw()
            self.fig.canvas.flush_events()
            time.sleep(1.0 / self.fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="", help="Szenenname")
    args = parser.parse_args()
    BEVViewer(init_scene=args.scene).run()


if __name__ == "__main__":
    main()