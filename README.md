# Splines Bézier — Trajectographie par NN

Contrôleur de trajectoire pour robots mobiles combinant **courbes de Bézier cubiques**, **apprentissage supervisé** et **raffinement cinématique**. Pipeline validé en **2D** (robot mobile) et étendu en **3D** (drone), avec exécution temps réel compatible **50 Hz** (budget 20 ms/pas).




---


## Utilisation

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

Projet académique (SeaTech, Toulon). Voir le rapport associé pour la description complète du système. [ici](rapport/Splines.pdf).













