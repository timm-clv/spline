# Splines Bézier — Trajectographie hybride IA + Physique

Contrôleur de trajectoire pour robots mobiles combinant **courbes de Bézier cubiques**, **apprentissage supervisé** et **raffinement cinématique**. Pipeline validé en **2D** (robot mobile) et étendu en **3D** (drone), avec exécution temps réel compatible **50 Hz** (budget 20 ms/pas).




---

## 📁 Structure du dépôt

├── 2D/ # Pipeline pour robot mobile (plan)
│ ├── generateur.py # Génère le dataset (20000 trajectoires, 360°)
│ ├── entrainement.py # Entraîne TimePredictorNet (6→18→5→1)
│ ├── application.py # Contrôleur + TrajectoryExecutor 50 Hz
│ ├── benchmark.py # Mesure p50/p95/p99 par brique
│ └── demo_visuelle.py # Animation matplotlib
│
└── 3D/ # Pipeline pour drone (volume)
├── generateur.py # Dataset 3D (sphère complète, repère local)
├── entrainement.py # Entraîne TimePredictorNet (8→20→10→1)
├── application.py # Contrôleur 3D (matrice de rotation locale)
├── benchmark.py # Benchmark pipeline 3D
├── demo_visuelle.py # Animation 3D matplotlib
└── simulate.py # Génère un JSON + viewer web (three.js) avec drone en forêt

---

## ⚙️ Principe

1. **Génération hors-ligne** : pour chaque paire (robot, cible), on calcule par dichotomie le temps minimal \(T_{opt}\) tel que la courbe de Bézier cubique respecte \(V_{\max} = 6.5\) m/s et \(A_{\max} = 6.0\) m/s². Les points de contrôle sont déduits des vitesses de départ/arrivée imposées :

   \[
   P_1 = P_0 + \frac{v_{start} \cdot T}{3} \, u_{start}, \qquad
   P_2 = P_3 - \frac{v_{end} \cdot T}{3} \, u_{end}
   \]

2. **Apprentissage supervisé** : un MLP ultra-léger (~250 poids en 2D, ~400 en 3D) prédit \(T_{opt}\) à partir de la cible exprimée en **repère local robot** (invariance par rotation/translation).

3. **Inférence hybride** : la prédiction neuronale sert d'initialisation, puis `refine_time_with_physics` garantit mathématiquement qu'aucune contrainte n'est violée (marge de sécurité 1 %).

4. **Exécution 50 Hz** : `TrajectoryExecutor` échantillonne la courbe à chaque pas (mode `cached`) ou replanifie à chaud (mode `replan`).

---

## 📊 Performances (CPU standard)

| Étape | 2D | 3D |
|---|---|---|
| Inférence NN | 0.07 ms | ~0.07 ms |
| `compute_plan` complet | ~3.9 ms (p99 4.1 ms) | ~6.8 ms (p99 10.9 ms) |
| `sample_plan` | 0.018 ms | 0.018 ms |

Les deux modes (`cached` et `replan`) tiennent largement le budget de 20 ms par pas.

---

## 🚀 Utilisation rapide

### Pipeline 2D
```bash
cd 2D
python generateur.py      # génère dataset_generer.npz
python entrainement.py    # entraîne modele_temps.pth
python benchmark.py       # vérifie la tenue à 50 Hz
python demo_visuelle.py   # animation
```

### Pipeline 3D
```bash
cd 3D
python generateur.py      # dataset 3D
python entrainement.py
python benchmark.py
python simulate.py        # génère trajectoires.json + ouvre le viewer web
```

Projet académique (SeaTech, Toulon). Voir le rapport associé pour la description complète du système. [ici](report/Visual%20Inertial%20Data%20Collection%20Platform%20and%20Sensor%20Fusion%20Benchmarking.pdf).
