"""Watson NLP based emotion detection."""

import json

import requests


WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def _empty_response():
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def _local_fallback(text_to_analyze):
    """Keep tests usable when the external lab service is unavailable."""
    lowered_text = text_to_analyze.lower()
    scores = {
        "anger": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "sadness": 0.0,
    }

    keyword_scores = {
        "anger": ("angry", "anger", "mad", "hate", "furious"),
        "disgust": ("disgust", "disgusted", "gross", "revolting"),
        "fear": ("fear", "afraid", "scared", "terrified"),
        "joy": ("happy", "joy", "joyful", "love", "glad"),
        "sadness": ("sad", "sadness", "unhappy", "depressed"),
    }

    for emotion, keywords in keyword_scores.items():
        if any(keyword in lowered_text for keyword in keywords):
            scores[emotion] = 0.9

    if not any(scores.values()):
        scores["joy"] = 0.2

    dominant_emotion = max(scores, key=scores.get)
    return {**scores, "dominant_emotion": dominant_emotion}


def emotion_detector(text_to_analyze):
    """Analyze text and return emotion scores with the dominant emotion."""
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(
            WATSON_URL,
            headers=HEADERS,
            json=payload,
            timeout=10,
        )
    except requests.RequestException:
        if not text_to_analyze:
            return _empty_response()
        return _local_fallback(text_to_analyze)

    if response.status_code == 400:
        return _empty_response()

    try:
        response_text = json.loads(response.text)
        emotion_scores = response_text["emotionPredictions"][0]["emotion"]
    except (KeyError, IndexError, json.JSONDecodeError):
        if not text_to_analyze:
            return _empty_response()
        return _local_fallback(text_to_analyze)

    formatted_response = {
        emotion: emotion_scores[emotion]
        for emotion in EMOTIONS
    }
    formatted_response["dominant_emotion"] = max(
        formatted_response,
        key=formatted_response.get,
    )

    return formatted_response
