"""Flask server for the emotion detection web application."""

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector


app = Flask(__name__)


@app.route("/")
def index():
    """Render the web interface."""
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detector_route():
    """Run emotion detection for text supplied by the web page."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    response = emotion_detector(text_to_analyze)

    if response["dominant_emotion"] is None:
        return "Invalid text! Please try again!"

    return (
        "For the given statement, the system response is "
        f"'anger': {response['anger']}, "
        f"'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, "
        f"'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
