"""
JRDB Annotation Viewer
Interaktiver Viewer für den JRDB-Datensatz: zeigt Bilder mit überlagerten
2D-BBoxen, Pose-Skeletons und Aktivitäts-Labels.

Aufruf:
    python jrdb_viewer.py
    python jrdb_viewer.py --scene stlc-111-2019-04-19_0 --camera image_0
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider

# ── Pfade ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "jrdb"
IMG_BASE   = BASE / "images"
L2D_BASE   = BASE / "labels" / "labels_2d"
POSE_BASE  = BASE / "labels" / "labels_2d_pose_coco"
L3D_BASE   = BASE / "labels" / "labels_3d"

CAMERAS = ["image_0", "image_2", "image_4", "image_6", "image_8", "image_stitched"]
# Kamera-Index für Label-Datei-Suffix (image_stitched → _stitched hat eigene Ordner)
CAM_LABEL_SUFFIX = {
    "image_0": "image0", "image_2": "image2", "image_4": "image4",
    "image_6": "image6", "image_8": "image8",
}

# ── JRDB-Skelett ────────────────────────────────────────────────────────────
KP_NAMES = [
    "head", "right eye", "left eye",
    "right shoulder", "center shoulder", "left shoulder",
    "right elbow", "left elbow",
    "center hip",
    "right wrist", "right hip", "left hip", "left wrist",
    "right knee", "left knee",
    "right foot", "left foot",
]
SKELETON = [
    (1, 2), (0, 4), (3, 4), (8, 10), (5, 7),
    (10, 13), (14, 16), (4, 5), (7, 12), (4, 8),
    (3, 6), (13, 15), (11, 14), (6, 9), (8, 11),
]

# Keypoint-Farben nach Körpergruppe
KP_COLORS = [
    "#4fc3f7",  # 0  head          — blau
    "#4fc3f7",  # 1  right eye
    "#4fc3f7",  # 2  left eye
    "#66bb6a",  # 3  right shoulder — grün
    "#66bb6a",  # 4  center shoulder
    "#66bb6a",  # 5  left shoulder
    "#ffa726",  # 6  right elbow    — orange
    "#ffa726",  # 7  left elbow
    "#ab47bc",  # 8  center hip     — lila
    "#ffa726",  # 9  right wrist
    "#ab47bc",  # 10 right hip
    "#ab47bc",  # 11 left hip
    "#ffa726",  # 12 left wrist
    "#ef5350",  # 13 right knee     — rot
    "#ef5350",  # 14 left knee
    "#ef9a9a",  # 15 right foot     — hellrot
    "#ef9a9a",  # 16 left foot
]

OCCLUSION_COLORS = {
    "Fully_visible":       "#66bb6a",   # grün
    "Partially_occluded":  "#ffd54f",   # gelb
    "Severely_occluded":   "#ef5350",   # rot
    "":                    "#90a4ae",   # grau (unbekannt)
}

# Jeder Person eine konsistente Farbe aus dieser Palette
PERSON_PALETTE = [
    "#e57373", "#f06292", "#ba68c8", "#9575cd", "#64b5f6",
    "#4dd0e1", "#4db6ac", "#81c784", "#dce775", "#ffb74d",
    "#ff8a65", "#a1887f", "#90a4ae", "#fff176", "#80cbc4",
]

def person_color(label_id: str) -> str:
    try:
        idx = int(label_id.split(":")[-1])
    except (ValueError, IndexError):
        idx = hash(label_id)
    return PERSON_PALETTE[idx % len(PERSON_PALETTE)]


# ── Datenladen ──────────────────────────────────────────────────────────────

def list_scenes() -> list[str]:
    p = IMG_BASE / "image_0"
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def load_labels_2d(scene: str, camera: str) -> dict:
    """Gibt Dict {filename → [annotations]} zurück (leeres Dict wenn nicht vorhanden)."""
    if camera == "image_stitched":
        path = BASE / "labels" / "labels_2d_stitched" / f"{scene}.json"
    else:
        suffix = CAM_LABEL_SUFFIX.get(camera, "image0")
        path = L2D_BASE / f"{scene}_{suffix}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("labels", {})


def load_labels_pose(scene: str, camera: str) -> tuple[dict, dict]:
    """
    Gibt zurück:
      frame_annots: {frame_filename → [annotation_dicts]}
      image_meta:   {frame_filename → image_dict}
    """
    if camera == "image_stitched":
        path = BASE / "labels" / "labels_2d_pose_stitched_coco" / f"{scene}.json"
    else:
        suffix = CAM_LABEL_SUFFIX.get(camera, "image0")
        path = POSE_BASE / f"{scene}_{suffix}.json"
    if not path.exists():
        return {}, {}
    with open(path) as f:
        data = json.load(f)

    # image_id → filename-Basisteil (z.B. "000000.jpg")
    id_to_fname: dict[int, str] = {}
    image_meta: dict[str, dict] = {}
    for img in data.get("images", []):
        fname = Path(img["file_name"]).name
        id_to_fname[img["id"]] = fname
        image_meta[fname] = img

    frame_annots: dict[str, list] = {}
    for ann in data.get("annotations", []):
        fname = id_to_fname.get(ann["image_id"], "")
        if fname:
            frame_annots.setdefault(fname, []).append(ann)
    return frame_annots, image_meta


def load_labels_3d(scene: str) -> dict:
    """Gibt Dict {pcd_filename → [annotations]} zurück."""
    path = L3D_BASE / f"{scene}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("labels", {})


# ── Zeichnen ─────────────────────────────────────────────────────────────────

def draw_bbox(ax, bbox, label_id, attributes, social_activity, show_activity):
    x, y, w, h = bbox
    occ = attributes.get("occlusion", "")
    color = OCCLUSION_COLORS.get(occ, OCCLUSION_COLORS[""])
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)

    pid = label_id.split(":")[-1] if ":" in label_id else label_id
    label_text = f"#{pid}"
    if show_activity:
        walk = social_activity.get("walking", 0)
        stand = social_activity.get("standing", 0)
        sit = social_activity.get("sitting", 0)
        top_act = max([("walk", walk), ("stand", stand), ("sit", sit)], key=lambda t: t[1])
        label_text += f" {top_act[0]}"

    ax.text(
        x, y - 3, label_text,
        color=color, fontsize=6.5, fontweight="bold",
        va="bottom", ha="left",
        bbox=dict(boxstyle="square,pad=0.1", facecolor="#0d0d0d", alpha=0.6, edgecolor="none"),
    )


def draw_pose(ax, keypoints_flat):
    """keypoints_flat: Liste von 51 Werten [x,y,v, x,y,v, ...]"""
    kps = np.array(keypoints_flat, dtype=float).reshape(17, 3)

    # Knochen
    for a, b in SKELETON:
        if kps[a, 2] > 0 and kps[b, 2] > 0:
            ax.plot(
                [kps[a, 0], kps[b, 0]],
                [kps[a, 1], kps[b, 1]],
                color="#eeeeee", linewidth=1.0, alpha=0.7, zorder=3,
            )

    # Keypoints
    for i, (x, y, v) in enumerate(kps):
        if v > 0:
            size = 14 if v == 2 else 8
            ax.scatter(x, y, s=size, c=KP_COLORS[i], zorder=4, linewidths=0)


# ── Viewer ────────────────────────────────────────────────────────────────────

class JRDBViewer:
    def __init__(self, init_scene: str = "", init_camera: str = "image_0"):
        self.scenes = list_scenes()
        if not self.scenes:
            raise RuntimeError(f"Keine Szenen gefunden unter {IMG_BASE}/image_0/")

        self.scene_idx  = self.scenes.index(init_scene) if init_scene in self.scenes else 0
        self.camera_idx = CAMERAS.index(init_camera) if init_camera in CAMERAS else 0

        self.playing    = False
        self.frame_idx  = 0
        self.fps        = 15

        # Daten für aktuelle Szene+Kamera
        self.image_paths: list[Path] = []
        self.labels_2d:   dict = {}
        self.labels_pose: dict = {}
        self.labels_3d:   dict = {}

        # Toggle-Zustand
        self.show_bbox     = True
        self.show_pose     = True
        self.show_activity = False

        self._build_ui()
        self._load_scene()
        self._draw_frame()

    # ── Daten laden ──────────────────────────────────────────────────────────

    def _load_scene(self):
        scene  = self.scenes[self.scene_idx]
        camera = CAMERAS[self.camera_idx]
        scene_dir = IMG_BASE / camera / scene
        if scene_dir.exists():
            self.image_paths = sorted(scene_dir.glob("*.jpg"))
        else:
            self.image_paths = []

        self.labels_2d   = load_labels_2d(scene, camera)
        self.labels_pose, _ = load_labels_pose(scene, camera)
        self.labels_3d   = load_labels_3d(scene)

        self.frame_idx = 0
        n = len(self.image_paths)
        self.slider.valmin = 0
        self.slider.valmax = max(n - 1, 0)
        self.slider.set_val(0)
        self._update_title()

    def _get_frame_data(self, idx: int):
        if not self.image_paths or idx >= len(self.image_paths):
            return None, [], []
        path = self.image_paths[idx]
        img_bgr = cv2.imread(str(path))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None

        fname = path.name
        bboxes = self.labels_2d.get(fname, [])
        poses  = self.labels_pose.get(fname, [])
        return img, bboxes, poses

    # ── UI aufbauen ──────────────────────────────────────────────────────────

    def _build_ui(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(15, 7), facecolor="#0d0d0d")
        self.fig.canvas.manager.set_window_title("JRDB Annotation Viewer")

        # Haupt-Image-Axis
        self.ax_img = self.fig.add_axes([0.01, 0.15, 0.72, 0.80])
        self.ax_img.set_facecolor("#111111")
        self.ax_img.axis("off")

        # Info-Panel rechts oben
        self.ax_info = self.fig.add_axes([0.75, 0.72, 0.23, 0.23])
        self.ax_info.set_facecolor("#111111")
        self.ax_info.axis("off")

        # ── Slider ──
        ax_slider = self.fig.add_axes([0.01, 0.07, 0.72, 0.03], facecolor="#222222")
        self.slider = Slider(ax_slider, "", 0, 1, valinit=0, valstep=1, color="#4fc3f7")
        self.slider.label.set_color("#888888")
        self.slider.valtext.set_color("#cccccc")
        self.slider.on_changed(self._on_slider)

        # ── Playback-Buttons ──
        btn_w, btn_h, btn_y = 0.05, 0.04, 0.01
        ax_prev = self.fig.add_axes([0.20, btn_y, btn_w, btn_h])
        ax_play = self.fig.add_axes([0.26, btn_y, btn_w, btn_h])
        ax_next = self.fig.add_axes([0.32, btn_y, btn_w, btn_h])

        style = dict(color="#222222", hovercolor="#333333")
        self.btn_prev = Button(ax_prev, "◀◀", **style)
        self.btn_play = Button(ax_play, "▶", **style)
        self.btn_next = Button(ax_next, "▶▶", **style)
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            b.label.set_color("#eeeeee")

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_play.on_clicked(self._on_play)
        self.btn_next.on_clicked(self._on_next)

        # ── Toggle-Checkboxen ──
        ax_check = self.fig.add_axes([0.75, 0.55, 0.23, 0.16], facecolor="#111111")
        self.check = CheckButtons(
            ax_check,
            ["2D BBoxes", "Pose Skeleton", "Activity Text"],
            [self.show_bbox, self.show_pose, self.show_activity],
        )
        for text in self.check.labels:
            text.set_color("#cccccc")
            text.set_fontsize(9)
        self.check.on_clicked(self._on_toggle)

        # ── Szenen-Buttons ──
        ax_slabel = self.fig.add_axes([0.75, 0.50, 0.23, 0.03])
        ax_slabel.axis("off")
        ax_slabel.text(0.0, 0.5, "Scene:", color="#888888", fontsize=8, va="center")

        ax_sprev = self.fig.add_axes([0.75, 0.44, 0.10, 0.04])
        ax_snext = self.fig.add_axes([0.88, 0.44, 0.10, 0.04])
        self.btn_sprev = Button(ax_sprev, "◀ Prev", color="#222222", hovercolor="#333333")
        self.btn_snext = Button(ax_snext, "Next ▶", color="#222222", hovercolor="#333333")
        for b in (self.btn_sprev, self.btn_snext):
            b.label.set_color("#eeeeee")
            b.label.set_fontsize(8)
        self.btn_sprev.on_clicked(self._on_scene_prev)
        self.btn_snext.on_clicked(self._on_scene_next)

        self.ax_scene_label = self.fig.add_axes([0.75, 0.39, 0.23, 0.04])
        self.ax_scene_label.axis("off")
        self.txt_scene = self.ax_scene_label.text(
            0.5, 0.5, "", color="#4fc3f7", fontsize=7,
            ha="center", va="center", wrap=True,
        )

        # ── Kamera-Buttons ──
        ax_clabel = self.fig.add_axes([0.75, 0.33, 0.23, 0.03])
        ax_clabel.axis("off")
        ax_clabel.text(0.0, 0.5, "Camera:", color="#888888", fontsize=8, va="center")

        ax_cprev = self.fig.add_axes([0.75, 0.27, 0.10, 0.04])
        ax_cnext = self.fig.add_axes([0.88, 0.27, 0.10, 0.04])
        self.btn_cprev = Button(ax_cprev, "◀", color="#222222", hovercolor="#333333")
        self.btn_cnext = Button(ax_cnext, "▶", color="#222222", hovercolor="#333333")
        for b in (self.btn_cprev, self.btn_cnext):
            b.label.set_color("#eeeeee")
            b.label.set_fontsize(8)
        self.btn_cprev.on_clicked(self._on_cam_prev)
        self.btn_cnext.on_clicked(self._on_cam_next)

        self.ax_cam_label = self.fig.add_axes([0.75, 0.22, 0.23, 0.04])
        self.ax_cam_label.axis("off")
        self.txt_cam = self.ax_cam_label.text(
            0.5, 0.5, "", color="#4fc3f7", fontsize=8,
            ha="center", va="center",
        )

        # ── Legende ──
        ax_legend = self.fig.add_axes([0.75, 0.02, 0.23, 0.18], facecolor="#111111")
        ax_legend.axis("off")
        ax_legend.text(0.02, 0.97, "Occlusion:", color="#888888", fontsize=7.5, va="top")
        for i, (label, color) in enumerate(OCCLUSION_COLORS.items()):
            if label == "":
                continue
            ax_legend.add_patch(mpatches.Rectangle(
                (0.02, 0.78 - i * 0.22), 0.12, 0.14,
                facecolor=color, edgecolor="none",
            ))
            ax_legend.text(0.18, 0.85 - i * 0.22, label.replace("_", " "),
                           color="#cccccc", fontsize=6.5, va="center")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── Zeichnen ─────────────────────────────────────────────────────────────

    def _draw_frame(self):
        img, bboxes, poses = self._get_frame_data(self.frame_idx)

        self.ax_img.cla()
        self.ax_img.axis("off")

        if img is None:
            self.ax_img.text(0.5, 0.5, "Kein Bild vorhanden",
                             color="#666666", ha="center", va="center",
                             transform=self.ax_img.transAxes, fontsize=12)
        else:
            self.ax_img.imshow(img, aspect="auto")

            if self.show_bbox:
                for ann in bboxes:
                    draw_bbox(
                        self.ax_img,
                        ann["box"],
                        ann.get("label_id", "?"),
                        ann.get("attributes", {}),
                        ann.get("social_activity", {}),
                        self.show_activity,
                    )

            if self.show_pose:
                for ann in poses:
                    kps = ann.get("keypoints", [])
                    if len(kps) == 51:
                        draw_pose(self.ax_img, kps)

        # Info-Panel
        self.ax_info.cla()
        self.ax_info.axis("off")
        self.ax_info.set_facecolor("#111111")
        scene  = self.scenes[self.scene_idx]
        camera = CAMERAS[self.camera_idx]
        n_frames = len(self.image_paths)
        n_persons = len(bboxes) if bboxes else 0
        n_poses   = len(poses)  if poses  else 0
        lines = [
            ("Szene",    scene[:28]),
            ("Kamera",   camera),
            ("Frame",    f"{self.frame_idx + 1} / {n_frames}"),
            ("Personen", f"{n_persons} Boxen · {n_poses} Posen"),
        ]
        for i, (key, val) in enumerate(lines):
            y = 0.82 - i * 0.22
            self.ax_info.text(0.03, y, f"{key}:", color="#888888", fontsize=8, va="top")
            self.ax_info.text(0.03, y - 0.10, val, color="#eeeeee", fontsize=8, va="top",
                              fontweight="bold")

        self.txt_scene.set_text(scene)
        self.txt_cam.set_text(camera)

        self.fig.canvas.draw_idle()

    def _update_title(self):
        scene  = self.scenes[self.scene_idx]
        camera = CAMERAS[self.camera_idx]
        n = len(self.image_paths)
        self.fig.canvas.manager.set_window_title(
            f"JRDB Viewer — {scene} [{camera}] — {n} Frames"
        )
        self.txt_scene.set_text(scene)
        self.txt_cam.set_text(camera)

    # ── Event-Handler ────────────────────────────────────────────────────────

    def _on_slider(self, val):
        self.frame_idx = int(val)
        self._draw_frame()

    def _on_prev(self, _):
        self.playing = False
        self.btn_play.label.set_text("▶")
        self.frame_idx = max(0, self.frame_idx - 1)
        self.slider.set_val(self.frame_idx)

    def _on_next(self, _):
        self.playing = False
        self.btn_play.label.set_text("▶")
        self.frame_idx = min(len(self.image_paths) - 1, self.frame_idx + 1)
        self.slider.set_val(self.frame_idx)

    def _on_play(self, _):
        self.playing = not self.playing
        self.btn_play.label.set_text("⏸" if self.playing else "▶")
        self.fig.canvas.draw_idle()

    def _on_toggle(self, label):
        if label == "2D BBoxes":
            self.show_bbox = not self.show_bbox
        elif label == "Pose Skeleton":
            self.show_pose = not self.show_pose
        elif label == "Activity Text":
            self.show_activity = not self.show_activity
        self._draw_frame()

    def _on_scene_prev(self, _):
        self.scene_idx = (self.scene_idx - 1) % len(self.scenes)
        self._load_scene()
        self._draw_frame()

    def _on_scene_next(self, _):
        self.scene_idx = (self.scene_idx + 1) % len(self.scenes)
        self._load_scene()
        self._draw_frame()

    def _on_cam_prev(self, _):
        self.camera_idx = (self.camera_idx - 1) % len(CAMERAS)
        self._load_scene()
        self._draw_frame()

    def _on_cam_next(self, _):
        self.camera_idx = (self.camera_idx + 1) % len(CAMERAS)
        self._load_scene()
        self._draw_frame()

    def _on_key(self, event):
        if event.key == " ":
            self._on_play(None)
        elif event.key == "right":
            self.frame_idx = min(len(self.image_paths) - 1, self.frame_idx + 1)
            self.slider.set_val(self.frame_idx)
        elif event.key == "left":
            self.frame_idx = max(0, self.frame_idx - 1)
            self.slider.set_val(self.frame_idx)

    # ── Hauptschleife ────────────────────────────────────────────────────────

    def run(self):
        plt.show(block=False)
        import time
        while plt.fignum_exists(self.fig.number):
            if self.playing and self.image_paths:
                self.frame_idx = (self.frame_idx + 1) % len(self.image_paths)
                self.slider.set_val(self.frame_idx)
                self._draw_frame()
            self.fig.canvas.flush_events()
            time.sleep(1.0 / self.fps)


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JRDB Annotation Viewer")
    parser.add_argument("--scene",  default="", help="Szenenname (z.B. stlc-111-2019-04-19_0)")
    parser.add_argument("--camera", default="image_0",
                        choices=CAMERAS, help="Kamera-Ordner")
    args = parser.parse_args()

    viewer = JRDBViewer(init_scene=args.scene, init_camera=args.camera)
    viewer.run()


if __name__ == "__main__":
    main()
