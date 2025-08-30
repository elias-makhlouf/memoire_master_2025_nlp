import matplotlib.pyplot as plt

# Extraire une séquence binaire multilabel par texte (auteur + ville)
def get_sequence(df, auteur, ville, theme_cols):
    df_text = df[(df['AUTEUR'] == auteur) & (df['VILLE'] == ville)].copy()
    df_text = df_text.sort_values('CHUNK_ID')
    return df_text[theme_cols].astype(int).values.tolist()

def distance_hamming(vec1, vec2):
    return sum(abs(a-b) for a,b in zip(vec1, vec2)) / len(vec1)

# Fonction pour tracer l'alignement DTW
def plot_alignment(path, titre="Alignement DTW entre deux textes"):
    x, y = zip(*path)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.xlabel("Chunks texte 1")
    plt.ylabel("Chunks texte 2")
    plt.title(titre)
    plt.grid(True)
    plt.show()

def plot_cartographie_texte(df_chunks, themes, auteur, ville, theme_cols, show):
    df_text = df_chunks[(df_chunks['AUTEUR'] == auteur) & (df_chunks['VILLE'] == ville)].copy()
    if df_text.empty:
        print("Aucun chunk trouvé pour cet auteur/ville.")
        return

    df_text.sort_values('CHUNK_ID', inplace=True)

    cmap1 = plt.get_cmap('tab20')
    cmap2 = plt.get_cmap('tab20b')
    cmap3 = plt.get_cmap('tab20c')
    colors1 = [cmap1(i) for i in range(20)]
    colors2 = [cmap2(i) for i in range(20)]
    colors3 = [cmap3(i) for i in range(20)]
    all_colors = colors1 + colors2 + colors3
    colors = all_colors[:len(themes)]
    theme_colors = {theme: colors[i] for i, theme in enumerate(themes)}

    fig, ax = plt.subplots(figsize=(len(df_text), 4))

    bar_width = 1

    for _, row in df_text.iterrows():
        x = row['CHUNK_ID'] - 1

        themes_true = [theme for theme in theme_cols if row[theme]]
        n_themes = len(themes_true)

        if n_themes == 0:
            ax.bar(x, 1, color='lightgrey', width=bar_width, edgecolor='none')
        else:
            height = 1 / n_themes
            for j, theme in enumerate(themes_true):
                bottom = j * height
                ax.bar(x, height, bottom=bottom, color=theme_colors[theme], width=bar_width, edgecolor='none')

    ax.set_xlim(-0.5, len(df_text) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("\n----------------------------------------------------------------------------->\nProgession dans le texte")
    ax.set_title(f"Cartographie thématique : {auteur} - {ville}")

    handles = [plt.Rectangle((0,0),1,1,color=color) for color in theme_colors.values()]
    ax.legend(handles, theme_colors.keys(), bbox_to_anchor=(0.5, -0.4), loc='upper center',ncols=3)

    plt.tight_layout()
    plt.savefig(f"figures/3-2/3-2_{auteur}_{ville}.png", bbox_inches='tight')
    
    if show == True: plt.show()
    
    plt.close()
    
def plot_cartographie_texte_regroupe(df, auteur, ville, groupes_cols, show):
    df_text = df[(df['AUTEUR'] == auteur) & (df['VILLE'] == ville)].copy()
    if df_text.empty:
        print("Aucun chunk trouvé pour cet auteur/ville.")
        return

    df_text.sort_values('CHUNK_ID', inplace=True)

    cmap1 = plt.get_cmap('Set3')
    cmap2 = plt.get_cmap('tab20b')
    colors1 = [cmap1(i) for i in range(20)]
    colors2 = [cmap2(i) for i in range(20)]
    all_colors = colors1 + colors2
    colors = all_colors[:len(groupes_cols)]
    theme_colors = {theme: colors[i] for i, theme in enumerate(groupes_cols)}

    fig, ax = plt.subplots(figsize=(len(df_text)*0.6, 6))

    bar_width = 1

    for _, row in df_text.iterrows():
        x = row['CHUNK_ID'] - 1

        groupes_true = [groupe for groupe in groupes_cols if row[groupe]]
        n_groupes = len(groupes_true)

        if n_groupes == 0:
            ax.bar(x, 1, color='lightgrey', width=bar_width, edgecolor='none')
        else:
            height = 1 / n_groupes
            for j, groupe in enumerate(groupes_true):
                bottom = j * height
                ax.bar(x, height, bottom=bottom, color=theme_colors[groupe], width=bar_width, edgecolor='none')

    ax.set_xlim(-0.5, len(df_text) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("\n----------------------------------------------------------------------------->\nProgression dans le texte")
    ax.set_title(f"Cartographie thématique regroupée : {auteur} - {ville}")

    handles = [plt.Rectangle((0,0),1,1,color=color) for color in theme_colors.values()]
    ax.legend(handles, theme_colors.keys(), bbox_to_anchor=(0.5, -0.4), loc='upper center', ncol=3)

    plt.tight_layout()
    plt.savefig(f"figures/3-3/3-3_{auteur}_{ville}.png", bbox_inches='tight')
    
    if show == True: plt.show()
    
    plt.close()