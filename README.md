# Application Web LSF

Ce README decrit uniquement le lancement et l'utilisation de l'application web Flask dans [scripts/web_app.py](scripts/web_app.py).

Le modele est suppose deja pret.

## Prerequis

- Python 3.10+
- Webcam fonctionnelle
- Fichiers modeles presents:
	- `models/model.h5`
	- `models/hand_landmarker.task`

Dependances (voir [requirements.txt](requirements.txt)):

- mediapipe
- tensorflow
- opencv-python
- numpy
- flask

## Installation

Depuis la racine du projet:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Lancer l'application web

Important: [scripts/web_app.py](scripts/web_app.py) utilise des chemins relatifs (`../models/...`), donc il faut le lancer depuis le dossier scripts.

```bash
cd scripts
python web_app.py
```

Ensuite, ouvrir dans le navigateur:

- http://localhost:5000

## Endpoints exposes

- `GET /` : page web principale (template [frontend/templates/index.html](frontend/templates/index.html))
- `GET /video_feed` : flux webcam MJPEG en direct
- `GET /api/state` : etat JSON du jeu (target, score, prediction, feedback, etc.)

## Fonctionnement rapide

- Le backend lit la webcam et detecte les landmarks de mains avec MediaPipe.
- Le modele classe les gestes sur une fenetre temporelle.
- Le frontend met a jour l'etat en temps reel via polling sur `/api/state`.

## Depannage

- Si la page charge mais la video est noire:
	- verifier que la webcam est disponible
	- fermer les autres applis qui utilisent la camera
- Si erreur fichier manquant:
	- verifier la presence de `models/model.h5` et `models/hand_landmarker.task`
	- verifier que la commande est lancee depuis `scripts/`
- Si `localhost:5000` ne repond pas:
	- verifier que le process Flask tourne bien dans le terminal


