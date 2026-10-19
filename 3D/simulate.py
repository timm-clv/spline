"""
Simulation 3D du drone : génère les trajectoires avec ton VelocityController,
les exporte en JSON puis ouvre le viewer (viewer.html) dans le navigateur.

Dossier attendu (tout dans le même dossier) :
    application.py, modele_temps.pth, statistique_de_normalisation.npz,
    drone.glb, viewer.html, simulate.py
Lancer :  python simulate.py
"""
import functools
import http.server
import json
import threading
import webbrowser
from pathlib import Path

import numpy as np

from application import VelocityController

HERE = Path(__file__).resolve().parent
DT = 0.02          # pas d'échantillonnage (50 Hz)
PORT = 8000

# Repère de la simulation : X, Y horizontaux, Z vers le haut (comme ton code).
# État initial : [x, y, z, dir_x, dir_y, dir_z, vitesse]
START = [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]   # au sol, à l'arrêt

WAYPOINTS = [
    # Décollage vertical
    [0.0,   0.0, 5.0,   0.0, 0.0,  1.0, 2.0],
    # Avance en altitude
    [10.0,  0.0, 6.0,   1.0, 0.0,  0.0, 4.0],
    # Piqué en 3 petites splines
    [13.0,  0.0, 4.5,   1.0, 0.0, -0.7, 3.5],
    [16.0,  0.0, 2.8,   1.0, 0.0, -0.5, 3.5],
    [19.0,  0.0, 1.5,   1.0, 0.0,  0.0, 3.0],
    # Slalom autour des arbres en (15,6), (10,6), (5,6)
    [15.0,  3.8, 1.5,  -1.0, 0.0,  0.0, 3.0],
    [10.0,  8.2, 1.5,  -1.0, 0.0,  0.0, 3.0],
    [5.0,   3.8, 1.5,  -1.0, 0.0,  0.0, 3.0],
    # Retour au-dessus du point de départ, puis atterrissage vertical
    [-3.0,  3.0, 2.0,   0.6, -0.8, -0.1, 2.0],
    [0.0,   0.0, 1.5,   0.0, 0.0, -1.0, 0.8],
    [0.0,   0.0, 0.05,  0.0, 0.0, -1.0, 0.0],
]

SLALOM_TREES = [[15.0, 6.0], [10.0, 6.0], [5.0, 6.0]]   # [x, y] au sol


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-9)


def simulate():
    ctrl = VelocityController()
    state = list(START)
    segments = []

    for wp in WAYPOINTS:
        target = [*wp[:3], *unit(wp[3:6]), wp[6]]
        plan = ctrl.compute_plan(state, target)
        T = plan["T_opt"]
        ts = np.linspace(0.0, T, int(np.ceil(T / DT)) + 1)

        pos, vel = [], []
        for t in ts:
            p, v, _ = ctrl.sample_plan(plan, t)
            pos.append(p.tolist())
            vel.append(v.tolist())

        segments.append({"T": T, "pos": pos, "vel": vel,
                         "ctrl": [plan[k].tolist() for k in ("P0", "P1", "P2", "P3")]})
        print(f"Segment {len(segments)} : T = {T:.2f} s")

        # L'état de fin de segment devient l'état de départ du suivant
        v_end = np.array(vel[-1])
        speed = float(np.linalg.norm(v_end))
        d = unit(v_end) if speed > 1e-6 else np.array([1.0, 0.0, 0.0])
        state = [*pos[-1], *d, speed]

    out = {"dt": DT, "start": START,"trees": SLALOM_TREES,
           "waypoints": [[*w[:3], *unit(w[3:6]).tolist()] for w in WAYPOINTS],
           "segments": segments}
    (HERE / "trajectoires.json").write_text(json.dumps(out))
    print("trajectoires.json écrit.")


def serve():
    # Un petit serveur local est nécessaire : le navigateur refuse de charger
    # un .glb ou un .json depuis file://
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    httpd = http.server.ThreadingHTTPServer(("localhost", PORT), handler)
    url = f"http://localhost:{PORT}/viewer.html"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"Viewer : {url}   (Ctrl+C pour arrêter)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    simulate()
    serve()
