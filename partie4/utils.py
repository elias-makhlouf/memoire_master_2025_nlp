import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.auto import tqdm
import umap
import pandas as pd

# Etape 1 : Moyenne des embeddings par (AUTEUR, VILLE) pour un thème donné
def mean_embedding_by_author_city(df, theme):
    filtered = df[df[theme] == True]
    grouped = filtered.groupby(["AUTEUR", "VILLE"])["EMBEDDING"]\
                .apply(lambda x: np.mean(np.vstack(x), axis=0))
    return grouped

# Etape 2 : Matrice de similarité cosinus
def cosine_similarity_matrix(embeddings_series):
    emb_array = np.vstack(embeddings_series.values)
    sim_matrix = cosine_similarity(emb_array)
    df_sim = pd.DataFrame(sim_matrix,
                          index=embeddings_series.index,
                          columns=embeddings_series.index)
    return df_sim

# Etape 3 : Heatmap
def plot_similarity_heatmap(df_sim, theme, base_path, show):
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_sim, annot=True, fmt=".1f", cmap="RdBu_r",vmin=0, vmax=1, square=True,
                cbar=True, cbar_kws={"shrink": 0.5, "pad": 0.02, "aspect": 10})
    plt.title(f"Similarité cosinus des textes \n Thème : {theme}")
    plt.xlabel(None)
    plt.ylabel(None)
    plt.savefig(f"{base_path}/similarity-heatmap/{theme}_matrice_sim.png", bbox_inches='tight')
    if show == True: plt.show()
    plt.close()

# Etape 4 : Réduction de dimension UMAP
def run_umap(embedding_series, n_neighbors=20, min_dist=0.3):
    X = np.vstack(embedding_series.values)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42, metric='cosine')
    X_umap = reducer.fit_transform(X)
    return X_umap

def compute_contributions(X_umap):
    contributions = (X_umap ** 2) / np.sum(X_umap ** 2, axis=0)
    cos2 = (X_umap ** 2) / np.sum(X_umap ** 2, axis=1)[:, np.newaxis]
    return contributions, cos2

def plot_umap_projection(X, labels, contribs, theme, show, base_path, title="UMAP",color_labels=None):
    plt.figure(figsize=(10, 10))

    # Couleurs
    if color_labels is not None:
        unique_vals = pd.Series(color_labels).unique()
        tab10 = plt.get_cmap("tab10")
        color_map = {val: tab10(i % 10) for i, val in enumerate(unique_vals)}
        colors = pd.Series(color_labels).map(color_map)
    else:
        colors = "skyblue"

    plt.scatter(X[:, 0], X[:, 1], s=contribs[:, 0]*5000, alpha=0.7,c=colors)
    for i, lbl in enumerate(labels):
        plt.text(X[i, 0], X[i, 1], str(lbl), fontsize=8, ha='left')

    # Marges
    x0, x1 = X[:, 0].min(), X[:, 0].max()
    y0, y1 = X[:, 1].min(), X[:, 1].max()
    m = 0.05
    xlim = (x0 - m*(x1 - x0), x1 + m*(x1 - x0))
    ylim = (y0 - m*(y1 - y0), y1 + m*(y1 - y0))
    plt.xlim(*xlim), plt.ylim(*ylim)

    # Ticks entiers uniquement
    xticks = np.arange(np.floor(xlim[0]), np.ceil(xlim[1]) + 1)
    yticks = np.arange(np.floor(ylim[0]), np.ceil(ylim[1]) + 1)
    plt.xticks(xticks), plt.yticks(yticks)

    plt.title(title)
    plt.xlabel("Dimension 1"), plt.ylabel("Dimension 2")
    plt.gca().set_aspect('equal')
    plt.grid(True, which='both', color='grey', linestyle='-', linewidth=0.2)

    # Légende
    if color_labels is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                       markerfacecolor=color_map[cat],
                       markersize=10)
        for cat in color_map]
        plt.legend(handles=handles, title=color_labels.name, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(f"{base_path}/UMAP-projection/{theme}_UMAP.png", bbox_inches='tight')
    if show == True: plt.show()
    plt.close()
    
def analyse_theme_embeddings_moy(df_chunks, theme, base_path, show=False):
  # Étape 1 : Embeddings moyens
  mean_embs = mean_embedding_by_author_city(df_chunks, theme)

  # Étape 2 : Matrice de similarité
  df_similarity = cosine_similarity_matrix(mean_embs)
  plot_similarity_heatmap(df_similarity, theme, base_path, show)

  # Étape 3 : UMAP
  X_umap_embedding = run_umap(mean_embs)

  # Étape 4 : Contributions / cos2
  contribs, cos2 = compute_contributions(X_umap_embedding)

  # Résultat dans un DataFrame
  resUMAP_embedding = pd.DataFrame({
      "AUTEUR": [idx[0] for idx in mean_embs.index],
      "VILLE": [idx[1] for idx in mean_embs.index],
      "Coord1": X_umap_embedding[:, 0],
      "Contrib1": contribs[:, 0],
      "Cos1": cos2[:, 0],
      "Coord2": X_umap_embedding[:, 1],
      "Contrib2": contribs[:, 1],
      "Cos2": cos2[:, 1],
  })
  resUMAP_embedding["GROUPE"] = resUMAP_embedding["AUTEUR"] + " \n " + resUMAP_embedding["VILLE"]

  # Affichage UMAP
  plot_umap_projection(X=X_umap_embedding,
                      labels=resUMAP_embedding["AUTEUR"].values,
                      contribs=contribs,
                      theme=theme,
                      show=show,
                      base_path=base_path,
                      color_labels=resUMAP_embedding["VILLE"],
                      title=f"Projection UMAP sur les embeddings moyennées \n Thème : {theme}")
  
  
  
  
  
  




# Étape 1 — Sélectionner les embeddings des chunks pour un thème donné
def select_chunks_for_theme(df, theme):
    return df[df[theme] == True].copy()

# Étape 2 — Similarité cosinus entre tous les chunks sélectionnés
def compute_pairwise_chunk_similarities(df_theme):
    embeddings = np.vstack(df_theme["EMBEDDING"].tolist())
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix, df_theme

# Étape 3 — Heatmap des similarités entre chunks
def plot_chunk_similarity_heatmap(sim_matrix, df_theme, theme, base_path, show):
    labels = [f"{auteur}-{ville}" for auteur, ville in zip(df_theme["AUTEUR"], df_theme["VILLE"])]
    plt.figure(figsize=(10, 10))
    sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels,
                cmap="RdBu_r", center=0.5, square=True, cbar_kws={"shrink": 0.5})
    plt.title(f"Similarité cosinus entre chunks — thème : {theme}")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{base_path}/similarity-heatmap-chunks/{theme}_matrice-sim-chunks.png", bbox_inches='tight')
    if show == True: plt.show()
    plt.close()

# Étape 4 — Moyenne des similarités cosinus entre textes, à partir de la matrice chunks×chunks
def compute_mean_text_similarities_from_chunk_matrix(sim_matrix_chunks, df_theme):

    # On s'assure que df_theme est dans le même ordre que sim_matrix_chunks
    df_theme_filtered = df_theme.loc[sim_matrix_chunks.index]

    # Grouper les indices des chunks par texte (tuple AUTEUR, VILLE)
    text_to_chunks = df_theme_filtered.groupby(["AUTEUR", "VILLE"]).apply(lambda g: g.index.tolist(), include_groups=False)

    texts = text_to_chunks.index.tolist()
    n = len(texts)
    sim_matrix_texts = np.zeros((n, n))

    for i in range(n):
        chunks_i = text_to_chunks.iloc[i]
        for j in range(i, n):
            chunks_j = text_to_chunks.iloc[j]

            # Extraire la sous-matrice des similarités entre les chunks des textes
            submatrix = sim_matrix_chunks.loc[chunks_i, chunks_j]
            mean_sim = submatrix.values.mean()
            sim_matrix_texts[i, j] = mean_sim
            sim_matrix_texts[j, i] = mean_sim
    return pd.DataFrame(sim_matrix_texts, index=texts, columns=texts)

# Étape 5 — Heatmap des similarités moyennes entre textes
def plot_text_similarity_heatmap(df_sim, theme, base_path, show):
    plt.figure(figsize=(10, 8))
    labels = [f"{auteur}-{ville}" for auteur, ville in df_sim.index]
    sns.heatmap(df_sim, annot=True, fmt=".1f", cmap="RdBu_r", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, square=True,
                cbar_kws={"shrink": 0.5})
    plt.title(f" Matrice des similarités cosinus moyennées par texte \n Thème : {theme}")
    plt.xlabel("Groupes (AUTEUR, VILLE)")
    plt.ylabel("Groupes (AUTEUR, VILLE)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{base_path}/similarity-heatmap-texts/{theme}_matrice-sim-textes.png", bbox_inches='tight')
    if show == True: plt.show()
    plt.close()
    

# Étape 6 — UMAP sur la matrice de similarité entre textes
def run_umap_on_text_similarities(df_sim, n_neighbors=10, min_dist=0.3):
    dist_matrix = 1 - df_sim.values           # convertir similarité en distance (valeurs entre 0 et 2)
    dist_matrix = np.clip(dist_matrix, 0, 2)  # éviter valeurs négatives

    reducer = umap.UMAP(metric="precomputed", n_neighbors=n_neighbors,
                        min_dist=min_dist, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(dist_matrix)
    return X_umap

def analyse_theme_cosinus_moy(df_chunks, theme, base_path, show=False):
    # Étape 1 : sélectionner les chunks pour le thème donné
    df_theme = select_chunks_for_theme(df_chunks, theme)

    # Étape 2 : similarité entre tous les chunks
    sim_chunk_matrix, df_theme_filtered = compute_pairwise_chunk_similarities(df_theme)
    sim_chunk_matrix = pd.DataFrame(sim_chunk_matrix, index=df_theme_filtered.index, columns=df_theme_filtered.index)
    plot_chunk_similarity_heatmap(sim_chunk_matrix, df_theme_filtered, theme, base_path, show)

    # Étape 4 : moyenne des similarités cosinus entre textes
    df_text_sim = compute_mean_text_similarities_from_chunk_matrix(sim_chunk_matrix, df_theme_filtered)
    plot_text_similarity_heatmap(df_text_sim, theme, base_path, show)

    # Étape 6 : projection UMAP
    X_umap = run_umap_on_text_similarities(df_text_sim)

    # Étape 7 : Contributions
    contributions, cos2 = compute_contributions(X_umap)

    # Construction du DataFrame résultat
    auteurs = [a for a, v in df_text_sim.index]
    villes = [v for a, v in df_text_sim.index]
    groupes = [f"{a} \n {v}" for a, v in df_text_sim.index]
    resUMAP_sim = pd.DataFrame({
        "AUTEUR": auteurs,
        "VILLE": villes,
        "GROUPE": groupes,
        "Coord1": X_umap[:, 0],
        "Coord2": X_umap[:, 1],
        "Contrib1": contributions[:, 0],
        "Contrib2": contributions[:, 1],
        "Cos2_1": cos2[:, 0],
        "Cos2_2": cos2[:, 1],
    })

    # UMAP avec projection harmonisée
    plot_umap_projection(
        X_umap,
        labels=resUMAP_sim["GROUPE"].values,
        contribs=contributions,
        theme=theme,
        show=show,
        base_path=base_path,
        title=f"UMAP — Similarités cosinus moyennées par texte \n Thème : {theme}",
        color_labels=resUMAP_sim["VILLE"]
    )