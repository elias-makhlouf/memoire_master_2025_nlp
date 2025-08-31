# Mémoire de Master 2025 – Traitement Automatique du Langage (NLP)

Ce répertoire contient le code que j'ai écrit dans le cadre de mon mémoire de double **Master Mondes Médiévaux/Humanités Numériques**. Les données associées à ce code sont également rendues disponibles.

---

## Pré-requis

* **Python 3.8 ou plus récent**
* Gestion d’environnement virtuel recommandée (`venv`, `conda` ou équivalent)
* Jupyter Notebook ou JupyterLab

---

## Installation

1. **Cloner le dépôt** :

   ```bash
   git clone https://github.com/elias-makhlouf/memoire_master_2025_nlp.git
   cd memoire_master_2025_nlp
   ```

2. **Créer un environnement virtuel** :

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances** :

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration du fichier `.env`

La seconde partie du projet mobilise les API de HuggingFace et de MistralAI. Il est donc nécessaire de passer des clés API dans le code pour autoriser les opérations. 
Ces clés sont privées et ne sont donc pas partagées dans ce répertoire. Pour faire fonctionner ce projet, il est necessaire de créer un fichier ``.env`` à la racine du projet. **Attention :** le fichier doit être exactement appelé de cette façon, sans extension supplémentaire.

### Exemple de contenu du fichier `.env` :

```dotenv
MISTRAL_API_KEY="votre_clé_api_ici"
HF_API_KEY="votre_clé_huggingface_ici"
```

### Étapes :

1. Créer un fichier `.env` à la racine du projet.
2. Y inscrire les clés nécessaires selon l’exemple ci-dessus.
3. Dans le notebook, ces valeurs sont automatiquement lues grâce à la librairie `python-dotenv` :

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

   if not MISTRAL_API_KEY:
       raise ValueError("La clé MISTRAL_API_KEY n'a pas été trouvée. Vérifiez votre fichier .env.")
   ```

⚠️ **Important** : ne pas partager le fichier `.env` ni vos clés API.

---

## Structure du projet

```
memoire_master_2025_nlp/
│
├── partie1
├── partie2
├── partie3
├────── data              # Répertoire contenant l'ensemble des données mobilisées
├────── figures           # Répertoire contenant les différentes figures générées
├────── utils.py          # Fonctions utlisées dans le notebook
├────── partie3.ipynb     # Notebook de la partie
├── partie4
├── requirements.txt      # Dépendances
├── .env 
├── .gitignore
└── README.md
```

- Chaque partie est structurée de la même façon que la partie 3 présentée ci-dessus.
