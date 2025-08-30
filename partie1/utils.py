import re

# --- Fonctions utilitaires ---
# Découpe en phrases avec ponctuation arabe et latine
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?؟؛])\s+', text) if s.strip()]

# Découpe par virgule ou point-virgule arabe/latin
def split_by_commas(text):
    return [p.strip() for p in re.split(r'[;,،]', text) if p.strip()]

# Compte les mots en séparant par espace
def count_words(text):
    return len(text.split())

# --- Fonction principale ---
def chunk_text(text):
    sentences = split_sentences(text)  # Découpe le texte en phrases
    chunks = []                        # Liste pour stocker les blocs finaux
    current_chunk = ""                # Bloc en cours de construction
    current_len = 0                   # Longueur (en mots) du bloc en cours

    for i, sentence in enumerate(sentences):  # Parcourt chaque phrase
        wcount = count_words(sentence)        # Compte les mots de la phrase

        if wcount == 50:                      # Si la phrase fait exactement 50 mots
            if current_len > 0:               # Si un bloc est en cours, on l’ajoute
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_len = 0
            chunks.append(sentence)           # Ajoute directement la phrase comme chunk

        elif wcount < 50:                     # Si la phrase est trop courte
            combined = sentence               # Commence à construire un bloc combiné
            combined_len = wcount
            j = i + 1
            while combined_len < 50 and j < len(sentences):  # Ajoute les phrases suivantes
                next_len = count_words(sentences[j])
                if combined_len + next_len > 50:
                    break                     # Arrête si on dépasse 50 mots
                combined += " " + sentences[j]
                combined_len += next_len
                j += 1
            if current_len > 0:               # Termine le bloc précédent si nécessaire
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_len = 0
            chunks.append(combined)           # Ajoute le bloc combiné
            for _ in range(i + 1, j):         # Marque les phrases combinées comme vides
                sentences[_] = ""

        else:  # wcount > 50                  # Si la phrase est trop longue
            parts = split_by_commas(sentence)  # Coupe la phrase en parties par virgules
            temp_chunk = ""
            temp_len = 0
            for part in parts:                 # Parcourt chaque sous-partie
                part_len = count_words(part)
                if temp_len + part_len > 50:   # Si on dépasse 50 mots avec ce morceau
                    if temp_len >= 50:         # Si la partie actuelle est suffisante
                        chunks.append(temp_chunk.strip())
                        temp_chunk = part
                        temp_len = part_len
                    else:
                        # Découpe manuellement en blocs de 50 mots
                        words = (temp_chunk + " " + part).split()
                        idx = 0
                        while idx < len(words):
                            end_idx = min(idx + 50, len(words))
                            chunk_words = words[idx:end_idx]
                            chunks.append(" ".join(chunk_words))
                            idx = end_idx
                        temp_chunk = ""
                        temp_len = 0
                else:
                    if temp_chunk:             # Ajoute avec une virgule si nécessaire
                        temp_chunk += ", " + part
                    else:
                        temp_chunk = part
                    temp_len += part_len
            if temp_chunk:                     # N’oublie pas d’ajouter le dernier fragment
                chunks.append(temp_chunk.strip())

    # Fusion finale des petits chunks < 10 mots
    final_chunks = []
    for chunk in chunks:
        if count_words(chunk) < 10 and final_chunks:    # Fusionne avec le précédent si trop court
            final_chunks[-1] += ' ' + chunk
        else:
            final_chunks.append(chunk)

    return final_chunks  # Retourne les chunks finaux