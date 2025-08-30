import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modèle 1 : DAMO-NLP XLM-R
def add_prefix(text, label_list, tokenizer):
    list_label_cleaned = [x + '.' if x[-1] not in ['.', '!'] else x for x in label_list]
    s_option = ' '.join(list_label_cleaned)
    return f'{s_option} {tokenizer.sep_token} {text}', list_label_cleaned

def run_damo_xlm(text, labels, model_name="DAMO-NLP-SG/zero-shot-classify-SSTuning-XLM-R", threshold=0.5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    labels_with_dots = [x + '.' if x[-1] not in ['.', '!'] else x for x in labels] # Version avec points pour le modèle
    prepared_text = f"{' '.join(labels_with_dots)} {tokenizer.sep_token} {text}"

    encoding = tokenizer([prepared_text], truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = model(**encoding).logits[:, :len(labels)]
    probs = torch.sigmoid(logits).squeeze().tolist()
    return {label: round(prob * 100, 2) for label, prob in zip(labels, probs)}# On associe les scores aux labels d’origine sans point

# Modèle 2 : Facebook BART Large MNLI
def run_facebook_bart(text, labels, model_name="facebook/bart-large-mnli"):
    classifier = pipeline("zero-shot-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
    result = classifier(text, candidate_labels=labels, multi_label=True)
    return {label: round(score * 100, 2) for label, score in zip(result["labels"], result["scores"])}

# Modèle 3 : XLM-RoBERTa Large XNLI
def run_xlm_roberta_xnli(text, labels, model_name="joeddav/xlm-roberta-large-xnli"):
    classifier = pipeline("zero-shot-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
    result = classifier(text, candidate_labels=labels, multi_label=True)
    return {label: round(score * 100, 2) for label, score in zip(result["labels"], result["scores"])}

# Modèle 4 : mDeBERTa (Multilingual DeBERTa)
def run_mdeberta_xnli(text, labels, model_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
    classifier = pipeline("zero-shot-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
    result = classifier(text, candidate_labels=labels, multi_label=True)
    return {label: round(score * 100, 2) for label, score in zip(result["labels"], result["scores"])}

# Comparaison harmonisée
def compare_models(text, labels, model_funcs):
    all_results = {}
    for name, func in model_funcs.items():
        print(f"Running model: {name}")
        scores = func(text, labels)
        all_results[name] = scores
    return pd.DataFrame(all_results).fillna(0)

# Retourne les mots-clés des seed topics correspondants
def get_topic_keywords(probabilities, seed_topics, threshold=0.15):
    multi_topic_results = []

    for chunk_idx, probas in enumerate(probabilities):
        detected_topics = []

        for topic_idx, prob in enumerate(probas):
            if prob > threshold and topic_idx != -1 and topic_idx < len(seed_topics):
                detected_topics.append({
                    'topic_index': topic_idx,
                    'keywords': seed_topics[topic_idx],
                    'probability': prob
                })

        multi_topic_results.append({
            'chunk_id': chunk_idx,
            'detected_topics': detected_topics
        })

    return multi_topic_results

def classify(text_chunk, themes, client):
    prompt = (
        f"Voici l'extrait d'un texte historique : \"{text_chunk}\"\n"
        f"Quels sont les thèmes abordés dans ce extrait parmi cette liste ? Donne uniquement les thèmes de la liste, en français :\n"
        f"{', '.join(themes)}\n"
        f"Ne donne ni phrase ni justification. Juste une liste séparée par des virgules."
        f"Ne surinterprète pas l'extrait."
    )
    max_retries = 5
    retry_delay = 1  # délai initial en secondes

    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": "Tu es un expert en analyse de textes historiques du Moyen-Age."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:  # Utilisez une exception plus générale pour capturer toutes les erreurs
            if hasattr(e, 'status_code') and e.status_code == 429 and attempt < max_retries - 1:
                print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # double le délai pour la prochaine tentative
            else:
                raise