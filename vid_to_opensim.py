"""
Pipeline 3D → OpenSim
=====================
Input  : video vue de face + video vue de côté
Output : pose3d.trc (prêt pour OpenSim / Pose2Sim)

Usage :
    python pipeline_3d_opensim.py --front face.mp4 --side side.mp4 --output pose3d.trc --height 1750
"""

import os, sys, argparse, tempfile
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Indices RTMPose WholeBody (133 pts) → nos 27 marqueurs
INDICES_27 = [
    0,   # Nose
    1,   # LEye
    2,   # REye
    5,   # LShoulder
    6,   # RShoulder
    7,   # LElbow
    8,   # RElbow
    9,   # LWrist
    10,  # RWrist
    11,  # LHip
    12,  # RHip
    13,  # LKnee
    14,  # RKnee
    15,  # LAnkle
    16,  # RAnkle
    17,  # LBigToe
    18,  # LSmallToe
    19,  # LHeel
    20,  # RBigToe
    21,  # RSmallToe
    22,  # RHeel
    95,  # LThumb
    99,  # LIndex
    111, # LPinky
    116, # RThumb
    120, # RIndex
    132, # RPinky
]

MARKER_NAMES = [
    "Nose", "LEye", "REye",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LWrist", "RWrist",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
    "LThumb", "LIndex", "LPinky",
    "RThumb", "RIndex", "RPinky",
    "Neck", "Head",
]

assert len(MARKER_NAMES) == 29


# ══════════════════════════════════════════════════════════════
# 1. EXTRACTION KEYPOINTS RTMPOSE
# ══════════════════════════════════════════════════════════════

def extraire_keypoints(video_path, label):
    """
    Extrait les 29 keypoints depuis une vidéo avec RTMPose WholeBody.
    Retourne : (N, 29, 3) → [x_pixels, y_pixels, score]
    """
    from mmpose.apis import MMPoseInferencer

    print(f"\n[RTMPose] Chargement modèle ({label})...")
    inferencer = MMPoseInferencer(
        pose2d="rtmpose-l_8xb32-270e_coco-wholebody-384x288",
        device="cpu"
    )

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[RTMPose] {total} frames · {W}×{H} · {fps:.1f} fps")

    all_kps = []
    tmp = os.path.join(tempfile.gettempdir(), "_rtmpose_frame.jpg")

    for n in range(total):
        ret, frame = cap.read()
        if not ret: break

        cv2.imwrite(tmp, frame)
        res     = next(inferencer(tmp, show=False))
        persons = res["predictions"][0]

        if persons:
            p = max(persons, key=lambda x: x.get("bbox_score", 0))
            kps_all    = np.array(p["keypoints"])        # (133, 2)
            scores_all = np.array(p["keypoint_scores"])  # (133,)

            # Sélectionne les 27 joints
            kps_27    = kps_all[INDICES_27]
            scores_27 = scores_all[INDICES_27]

            # Calcule Neck = milieu LShoulder + RShoulder
            neck       = (kps_all[5] + kps_all[6]) / 2
            neck_score = (scores_all[5] + scores_all[6]) / 2

            # Calcule Head = Neck + 1.33 × (Neck → Nose)
            neck_to_nose = kps_all[0] - neck
            head         = neck + 1.33 * neck_to_nose
            head_score   = scores_all[0]

            kps_29    = np.vstack([kps_27, neck.reshape(1,2), head.reshape(1,2)])
            scores_29 = np.append(scores_27, [neck_score, head_score])

        else:
            kps_29    = np.zeros((29, 2))
            scores_29 = np.zeros(29)

        kps_with_score = np.hstack([kps_29, scores_29.reshape(-1,1)])  # (29, 3)
        all_kps.append(kps_with_score)

        if (n+1) % 10 == 0:
            print(f"\r[RTMPose] {label} : {n+1}/{total}", end="", flush=True)

    cap.release()
    print(f"\n[RTMPose] {label} terminé : {len(all_kps)} frames")
    return np.array(all_kps), fps, W, H


# ══════════════════════════════════════════════════════════════
# 2. RECONSTRUCTION 3D
# ══════════════════════════════════════════════════════════════

def reconstruire_3d(kps_front, kps_side, height_mm, H_front, H_side):
    """
    Reconstruit les coordonnées 3D depuis 2 vues orthogonales.

    Vue FACE  → X (gauche/droite) + Y (haut/bas)
    Vue CÔTÉ  → Z (avant/arrière) + Y (haut/bas)
    Y final   = moyenne pondérée par les scores

    Retourne : (N, 29, 3) en mètres
    """
    N = min(len(kps_front), len(kps_side))
    kps_front = kps_front[:N]
    kps_side  = kps_side[:N]

    # Calcul de l'échelle pixels → mètres
    # Utilise la hauteur de la personne détectée automatiquement
    # On prend la distance entre Head et LAnkle/RAnkle en pixels
    head_idx   = MARKER_NAMES.index("Head")
    lankle_idx = MARKER_NAMES.index("LAnkle")
    rankle_idx = MARKER_NAMES.index("RAnkle")

    head_y   = kps_front[:, head_idx,   1].mean()
    ankle_y  = (kps_front[:, lankle_idx, 1] + kps_front[:, rankle_idx, 1]).mean() / 2
    height_px = abs(ankle_y - head_y)

    if height_px > 10:
        scale = (height_mm / 1000.0) / height_px  # mm → mètres
        print(f"[3D] Hauteur détectée : {height_px:.0f} px → échelle {scale*1000:.3f} mm/px")
    else:
        scale = (height_mm / 1000.0) / H_front
        print(f"[3D] Hauteur non détectée, échelle par défaut")

    coords_3d = np.zeros((N, 29, 3))

    for f in range(N):
        for m in range(29):
            x_front  = kps_front[f, m, 0]
            y_front  = kps_front[f, m, 1]
            x_side   = kps_side[f, m, 0]
            y_side   = kps_side[f, m, 1]
            s_front  = kps_front[f, m, 2]
            s_side   = kps_side[f, m, 2]

            # X → vue de face (gauche/droite)
            X = x_front * scale

            # Z → vue de côté (avant/arrière)
            Z = x_side * scale

            # Y → moyenne pondérée des deux vues (haut/bas)
            if s_front + s_side > 0:
                Y = (y_front * s_front + y_side * s_side) / (s_front + s_side) * scale
            else:
                Y = y_front * scale

            # OpenSim : Y vers le haut → inverser (pixels vers le bas)
            coords_3d[f, m] = [X, -Y, Z]

    print(f"[3D] Reconstruction terminée : {N} frames · 29 marqueurs")
    return coords_3d, N


# ══════════════════════════════════════════════════════════════
# 3. CENTRAGE DU PELVIS
# ══════════════════════════════════════════════════════════════

def centrer_pelvis(coords_3d):
    """
    Centre les coordonnées sur le pelvis (milieu des hanches).
    Et remonte Y pour que les pieds soient au niveau du sol.
    """
    lhip_idx = MARKER_NAMES.index("LHip")
    rhip_idx = MARKER_NAMES.index("RHip")

    # Centre X et Z sur le pelvis
    pelvis_x = (coords_3d[:, lhip_idx, 0] + coords_3d[:, rhip_idx, 0]) / 2
    pelvis_z = (coords_3d[:, lhip_idx, 2] + coords_3d[:, rhip_idx, 2]) / 2

    coords_3d[:, :, 0] -= pelvis_x[:, np.newaxis]
    coords_3d[:, :, 2] -= pelvis_z[:, np.newaxis]

    # Remonte Y pour que les chevilles soient à Y=0
    lankle_idx = MARKER_NAMES.index("LAnkle")
    rankle_idx = MARKER_NAMES.index("RAnkle")
    ankle_y    = (coords_3d[:, lankle_idx, 1] + coords_3d[:, rankle_idx, 1]).mean() / 2
    coords_3d[:, :, 1] -= ankle_y

    print(f"[3D] Centrage pelvis effectué")
    return coords_3d
    
from scipy.signal import savgol_filter

def lisser_coordonnees(coords_3d, fps, cut_off=3):
    """
    Lisse les coordonnées 3D avec un filtre Savitzky-Golay.
    Réduit le bruit de détection RTMPose frame par frame.
    
    cut_off : fréquence de coupure en Hz (plus bas = plus lisse)
    """
    # Calcule la fenêtre selon le fps et la fréquence de coupure
    window = int(fps / cut_off)
    if window % 2 == 0:
        window += 1  # doit être impair
    window = max(window, 5)  # minimum 5 frames
    
    print(f"[Lissage] Fenêtre={window} frames · cut_off={cut_off} Hz")
    
    coords_smooth = coords_3d.copy()
    for m in range(coords_3d.shape[1]):
        for ax in range(3):
            coords_smooth[:, m, ax] = savgol_filter(
                coords_3d[:, m, ax], window, polyorder=3
            )
    
    print(f"[Lissage] Terminé")
    return coords_smooth


# ══════════════════════════════════════════════════════════════
# 4. EXPORT .TRC
# ══════════════════════════════════════════════════════════════

def export_trc(coords_3d, fps, output_path):
    """
    Exporte les coordonnées 3D au format .trc OpenSim.
    coords_3d : (N, 29, 3) en mètres
    """
    N_frames  = coords_3d.shape[0]
    N_markers = coords_3d.shape[1]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_path}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.2f}\t{fps:.2f}\t{N_frames}\t{N_markers}\t"
                f"m\t{fps:.2f}\t1\t{N_frames}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(MARKER_NAMES) + "\n")
        f.write("\t\t" + "\t".join(
            [f"X{j+1}\tY{j+1}\tZ{j+1}" for j in range(N_markers)]
        ) + "\n\n")

        for i in range(N_frames):
            t   = i / fps
            row = [str(i+1), f"{t:.6f}"]
            for m in range(N_markers):
                x, y, z = coords_3d[i, m]
                row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
            f.write("\t".join(row) + "\n")

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n[TRC] Exporté : {output_path}")
    print(f"[TRC] {N_frames} frames · {N_markers} marqueurs · mètres · {size_kb:.1f} KB")


# ══════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def run(front_video, side_video, output_trc, height_mm=1750):

    print("=" * 60)
    print("Pipeline 3D → OpenSim")
    print(f"  Vue face : {front_video}")
    print(f"  Vue côté : {side_video}")
    print(f"  Hauteur  : {height_mm} mm")
    print(f"  Sortie   : {output_trc}")
    print("=" * 60)

    # Étape 1 — Extraction keypoints
    print("\n── Étape 1/3 : Extraction RTMPose ──────────────────────")
    kps_front, fps_f, W_f, H_f = extraire_keypoints(front_video, "face")
    kps_front = np.array(kps_front)
    print(kps_front.shape)
    kps_side,  fps_s, W_s, H_s = extraire_keypoints(side_video,  "cote")
    kps_side = np.array(kps_side)
    fps = (fps_f + fps_s) / 2

    # Étape 2 — Reconstruction 3D
    print("\n── Étape 2/3 : Reconstruction 3D ───────────────────────")
    coords_3d, N = reconstruire_3d(kps_front, kps_side, height_mm, H_f, H_s)
    coords_3d    = centrer_pelvis(coords_3d)
    coords_3d = lisser_coordonnees(coords_3d, fps, cut_off=2)

    # Étape 3 — Export TRC
    print("\n── Étape 3/3 : Export TRC ───────────────────────────────")
    export_trc(coords_3d, fps, output_trc)

    print("\n✅ Terminé !")
    print(f"   {output_trc} → prêt pour OpenSim / Pose2Sim")
    print(f"\nPour lancer l'IK dans OpenSim :")
    print(f"   conda activate opensim_env")
    print(f"   copy {output_trc} pose-3d\\")
    print(f"   python -c \"import toml; from Pose2Sim.kinematics import kinematics_all; c=toml.load('Config.toml'); c['kinematics']['use_augmentation']=False; c['kinematics']['use_simple_model']=True; kinematics_all(c)\"")
    

# ══════════════════════════════════════════════════════════════
# 6. POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline 3D → OpenSim depuis 2 vidéos"
    )
    parser.add_argument("--front",  required=True, help="Vidéo vue de face")
    parser.add_argument("--side",   required=True, help="Vidéo vue de côté")
    parser.add_argument("--output", default="pose3d.trc", help="Fichier TRC de sortie")
    parser.add_argument("--height", type=int, default=1750,
                        help="Hauteur de la personne en mm (défaut: 1750)")
    args = parser.parse_args()

    run(args.front, args.side, args.output, args.height)