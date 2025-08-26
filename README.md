# Mémoire Master 2025 - NLP

Ce répertoire contient le code que j'ai écrit dans le cadre de mon mémoire de double **Master Mondes Médiévaux/Humanités Numériques**. Les données associées à ce code sont également rendues disponibles.

## Comment faire tourner le notebook ?

Avant de lancer le notebook, veuiller installez les librairies necessaires, à l'aide du fichier ```requirements.txt``` fournit. Exécutez la commande suivante dans le terminal : 

```
pip install -r requirements.txt
```

Vous pouvez faire tourner ce projet dans un environnement virtuel afin de prévenir le risque de conflits de dépendances.

Le notebook est censé fonctionner par défaut sans ajustements nécessaires. Il faut simplement renseigner une clé privée pour la partie 2.3. (Mistral AI) dans la première bulle de cette partie :

```
MISTRAL_API_KEY = "" #Veuillez insérer votre clé privée entre les guillemets
client = Mistral(api_key=MISTRAL_API_KEY)
```