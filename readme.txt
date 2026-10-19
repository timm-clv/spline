Objectif : Système de trajectoire 2D B-spline par réseau neuronal léger pouvant recalculer une position 50 fois par seconde.


Notes :
L'objectif dit « B-spline », le code fait du « Bézier cubique contraint ». On assume Bézier ⊂ B-spline.



Battu par : BEAST (NeurIPS 2026) : Cet article présente un "tokenizer" qui encode les séquences d'actions en utilisant des B-splines. L'avantage est de garantir la génération de trajectoires lisses, sans discontinuités, ce qui est crucial pour le contrôle d'un robot.



Commande pour scanner les dossiers :
Get-ChildItem -Path "." -Filter "*.py" -Recurse | Sort-Object FullName | ForEach-Object {
    "`n===== $($_.FullName) =====`n"
    Get-Content $_.FullName
} > contexte.txt



Actions :
# 1. Régénérer le dataset (360°, ~3-5 min sur CPU)
python generateur.py

# 2. Réentraîner le NN (~1-2 min)
python entrainement.py

# 3. Vérifier latence + valider 50 Hz
python benchmark.py

# 4. Démontrer visuellement (avec GIF si --save)
python demo_visuelle.py --save