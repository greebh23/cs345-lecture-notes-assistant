import os
import json
import tempfile
import threading
import wave
import subprocess
import requests
import re

from flask import Flask, request, jsonify
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk


# --------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)

# Maximum upload size: 25 MB
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")

ALLOWED_EXTENSIONS = {
    "wav", "mp3", "webm", "ogg",
    "opus", "m4a", "aac"
}


# --------------------------------------------------
# 2. FFmpeg AVAILABILITY CHECK
# --------------------------------------------------

def check_ffmpeg():

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10
        )

    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "FFmpeg is not installed or configured correctly."
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg availability check timed out.")


# --------------------------------------------------
# 3. AUDIO CONVERSION
# --------------------------------------------------

def convert_to_wav(input_path, output_path):
    """
    Convert audio into 16 kHz, 16-bit, mono PCM WAV.
    """

    check_ffmpeg()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                output_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )

    except subprocess.TimeoutExpired:
        raise TimeoutError("Audio conversion timed out.")

    except subprocess.CalledProcessError as error:
        raise ValueError(
            "Invalid or unsupported audio file."
        ) from error

    # Validate converted WAV
    try:
        with wave.open(output_path, "rb") as audio:

            if audio.getnchannels() != 1:
                raise ValueError("Audio must be mono.")

            if audio.getframerate() != 16000:
                raise ValueError("Audio must be 16 kHz.")

            if audio.getsampwidth() != 2:
                raise ValueError("Audio must be 16-bit PCM.")

            if audio.getnframes() == 0:
                raise ValueError("Audio recording is empty.")

    except (wave.Error, EOFError) as error:
        raise ValueError(
            "Failed to validate converted WAV audio."
        ) from error

    return output_path


# --------------------------------------------------
# 4. AUDIO DURATION
# --------------------------------------------------

def get_audio_duration(audio_path):

    try:
        with wave.open(audio_path, "rb") as audio:

            if audio.getframerate() == 0:
                return 0.0

            duration = (
                audio.getnframes() / audio.getframerate()
            )

            return round(duration, 2)

    except (wave.Error, EOFError):
        return 0.0


# --------------------------------------------------
# 5. AZURE SPEECH-TO-TEXT
# --------------------------------------------------

def transcribe_audio(audio_path):
    """
    Transcribe audio using Azure Speech.
    """

    if not SPEECH_KEY or not SPEECH_REGION:
        raise RuntimeError(
            "Azure Speech credentials are missing."
        )

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    # Request detailed results
    speech_config.output_format = (
        speechsdk.OutputFormat.Detailed
    )

    # English recognition
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(
        filename=audio_path
    )

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    transcript_parts = []
    words = []
    confidences = []
    errors = []

    done = threading.Event()

    # Recognition callback
    def recognized_callback(evt):

        result = evt.result

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:

            if result.text:
                transcript_parts.append(result.text)

            raw_json = result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )

            if raw_json:

                try:
                    data = json.loads(raw_json)
                    nbest = data.get("NBest", [])

                    if nbest:

                        best = nbest[0]

                        confidence = best.get("Confidence")

                        if isinstance(confidence, (int, float)):
                            confidences.append(float(confidence))

                        for item in best.get("Words", []):

                            offset = item.get("Offset", 0)
                            duration = item.get("Duration", 0)

                            words.append({
                                "word": item.get("Word", ""),
                                "offset": round(
                                    offset / 10000000, 3
                                ),
                                "duration": round(
                                    duration / 10000000, 3
                                ),
                                "confidence": item.get("Confidence")
                            })

                except (json.JSONDecodeError, TypeError, ValueError):
                    print(
                        "Could not parse detailed recognition JSON."
                    )

        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech recognized in this segment.")

    # Cancellation callback
    def canceled_callback(evt):

        if evt.reason == speechsdk.CancellationReason.Error:

            errors.append(
                evt.error_details or "Unknown Azure Speech error."
            )

            print(
                "Speech recognition error:",
                evt.error_details
            )

        done.set()

    # Session completion callback
    def session_stopped_callback(evt):

        print("Speech recognition session completed.")
        done.set()

    recognizer.recognized.connect(recognized_callback)
    recognizer.canceled.connect(canceled_callback)
    recognizer.session_stopped.connect(session_stopped_callback)

    recognition_started = False

    try:

        recognizer.start_continuous_recognition_async().get()

        recognition_started = True

        if not done.wait(timeout=600):
            raise TimeoutError("Speech recognition timed out.")

    finally:

        if recognition_started:

            try:
                recognizer.stop_continuous_recognition_async().get()

            except Exception as error:
                print("Recognition cleanup warning:", str(error))

        recognizer = None
        audio_config = None

    if errors:
        raise RuntimeError(errors[0])

    transcript = " ".join(transcript_parts).strip()

    if not transcript:
        raise ValueError(
            "No speech could be recognized in this audio file."
        )

    confidence = (
        round(sum(confidences) / len(confidences), 3)
        if confidences else None
    )

    duration_seconds = get_audio_duration(audio_path)

    return {
        "transcript": transcript,
        "language": "en-US",
        "duration_seconds": duration_seconds,
        "confidence": confidence,
        "words": words
    }


# --------------------------------------------------
# 6. AZURE AI LANGUAGE ANALYSIS
# --------------------------------------------------

def analyze_text_feature(text, feature):
    """
    Send text to Azure AI Language for analysis.
    """

    if not LANGUAGE_KEY or not LANGUAGE_ENDPOINT:
        raise RuntimeError(
            "Azure AI Language credentials are missing."
        )

    endpoint = LANGUAGE_ENDPOINT.rstrip("/")

    url = (
        f"{endpoint}/language/:analyze-text"
        "?api-version=2024-11-01"
    )

    payload = {
        "kind": feature,
        "analysisInput": {
            "documents": [
                {
                    "id": "1",
                    "language": "en",
                    "text": text
                }
            ]
        },
        "parameters": {
            "modelVersion": "latest"
        }
    }

    headers = {
        "Ocp-Apim-Subscription-Key": LANGUAGE_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    if not response.ok:
        raise RuntimeError(
            f"Azure Language {feature} failed: "
            f"{response.status_code} {response.text}"
        )

    data = response.json()

    documents = data.get(
        "results", {}
    ).get("documents", [])

    if not documents:
        raise RuntimeError(
            f"No results returned for {feature}."
        )

    return documents[0]


# --------------------------------------------------
# 7. ANALYSE TRANSCRIPT
# --------------------------------------------------

def analyze_transcript(text):
    """
    Extract key phrases, entities and sentiment.
    Return simplified, formatted JSON.
    """

    # Key phrase extraction
    key_phrases_result = analyze_text_feature(
        text,
        "KeyPhraseExtraction"
    )

    # Named entity recognition
    entities_result = analyze_text_feature(
        text,
        "EntityRecognition"
    )

    # Sentiment analysis
    sentiment_result = analyze_text_feature(
        text,
        "SentimentAnalysis"
    )

    # Format entities
    formatted_entities = []

    for entity in entities_result.get("entities", []):

        formatted_entities.append({
            "category": entity.get("category"),
            "confidence": entity.get("confidenceScore"),
            "length": entity.get("length"),
            "offset": entity.get("offset"),
            "subcategory": entity.get("subcategory"),
            "text": entity.get("text")
        })

    # Format sentiment
    sentiment = {
        "label": sentiment_result.get("sentiment"),
        "confidence_scores": sentiment_result.get(
            "confidenceScores", {}
        )
    }

    # Return structured analysis
    return {
        "entities": formatted_entities,

        "entity_linking": {
            "enabled": False,
            "message": (
                "Entity linking was not implemented "
                "because it is optional."
            )
        },

        "key_phrases": key_phrases_result.get(
            "keyPhrases", []
        ),

        "language": "en",

        "sentiment": sentiment,

        "text": text
    }



# --------------------------------------------------
# 8. LECTURE STUDY MATERIAL GENERATION
# --------------------------------------------------

def get_sentences(text):
    """Split transcript into individual sentences."""

    return [
        sentence.strip()
        for sentence in re.split(r'(?<=[.!?])\s+', text)
        if sentence.strip()
    ]


def generate_structured_notes(text, key_phrases):
    """
    Organize extracted key phrases into study notes
    using relevant sentences from the transcript.
    """

    sentences = get_sentences(text)
    notes = []

    for phrase in key_phrases:

        matching_sentences = [
            sentence for sentence in sentences
            if phrase.lower() in sentence.lower()
        ]

        if matching_sentences:

            notes.append({
                "topic": phrase,
                "details": matching_sentences[:3]
            })

    return notes


def generate_glossary(text, key_phrases):
    """
    Create a glossary of unique key terms.
    Include relevant transcript context.
    """

    sentences = get_sentences(text)
    glossary = []
    seen_terms = set()

    for phrase in key_phrases:

        normalized = phrase.strip().lower()

        if not normalized or normalized in seen_terms:
            continue

        seen_terms.add(normalized)

        matching_sentences = [
            sentence for sentence in sentences
            if normalized in sentence.lower()
        ]

        glossary.append({
            "term": phrase,
            "context": (
                matching_sentences[0]
                if matching_sentences else None
            ),
            "occurrences": len(matching_sentences)
        })

    return glossary


def generate_quiz_questions(text, key_phrases):
    """
    Generate up to three self-check questions
    using important terms and lecture sentences.
    """

    sentences = get_sentences(text)
    questions = []
    used_terms = set()

    for phrase in key_phrases:

        normalized = phrase.strip().lower()

        if normalized in used_terms:
            continue

        for sentence in sentences:

            if normalized not in sentence.lower():
                continue

            if len(sentence.split()) < 5:
                continue

            question_text = re.sub(
                re.escape(phrase),
                "________",
                sentence,
                count=1,
                flags=re.IGNORECASE
            )

            if question_text == sentence:
                continue

            questions.append({
                "question": (
                    "Complete the missing term: "
                    + question_text
                ),
                "answer": phrase
            })

            used_terms.add(normalized)
            break

        if len(questions) == 3:
            break

    return questions


def generate_study_materials(text, analysis):
    """
    Generate all lecture study materials.
    """

    key_phrases = analysis.get("key_phrases", [])

    structured_notes = generate_structured_notes(
        text, key_phrases
    )

    glossary = generate_glossary(
        text, key_phrases
    )

    quiz_questions = generate_quiz_questions(
        text, key_phrases
    )

    return {
        "structured_notes": structured_notes,
        "glossary": glossary,
        "quiz_questions": quiz_questions
    }


# --------------------------------------------------
# 8. POST /ANALYZE ENDPOINT
# --------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A JSON request body is required."
        }), 400

    transcript = data.get("text")

    if not isinstance(transcript, str) or not transcript.strip():
        return jsonify({
            "error": "Please provide transcript text."
        }), 400

    try:

        results = analyze_transcript(transcript.strip())

        study_materials = generate_study_materials(
            transcript.strip(),
            results
        )

        results.update(study_materials)
        # Return formatted JSON directly
        return jsonify(results), 200

    except requests.RequestException as error:

        print("Language connection error:", str(error))

        return jsonify({
            "error": "Could not connect to Azure AI Language."
        }), 502

    except RuntimeError as error:

        print("Language analysis error:", str(error))

        return jsonify({
            "error": "Azure AI Language analysis failed. "
                     "Check your Language resource settings."
        }), 500


# --------------------------------------------------
# 9. POST /TRANSCRIBE ENDPOINT
# --------------------------------------------------

@app.route("/transcribe", methods=["POST"])
def transcribe():

    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file uploaded. "
                     "Use the field name 'audio'."
        }), 400

    audio_file = request.files["audio"]

    if not audio_file.filename:
        return jsonify({
            "error": "Please select an audio file."
        }), 400

    extension = audio_file.filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Unsupported audio format."
        }), 415

    temp_path = None
    converted_path = None

    try:

        # Save uploaded file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix="." + extension
        ) as temp_file:

            temp_path = temp_file.name
            audio_file.save(temp_path)

        if os.path.getsize(temp_path) == 0:
            raise ValueError("Uploaded audio is empty.")

        # Create converted WAV file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as wav_file:

            converted_path = wav_file.name

        convert_to_wav(temp_path, converted_path)

        # Transcribe
        result = transcribe_audio(converted_path)

        return jsonify(result), 200

    except TimeoutError as error:

        return jsonify({
            "error": str(error)
        }), 504

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 422

    except RuntimeError as error:

        print("Transcription error:", str(error))

        return jsonify({
            "error": "Speech transcription failed. "
                     "Check FFmpeg, Azure settings and audio file."
        }), 500

    except Exception as error:

        print("Unexpected transcription error:", str(error))

        return jsonify({
            "error": "An unexpected transcription error occurred."
        }), 500

    finally:

        for path in [temp_path, converted_path]:

            if path and os.path.exists(path):

                try:
                    os.remove(path)

                except OSError as error:
                    print("Temporary file cleanup warning:", str(error))


# --------------------------------------------------
# 10. POST /PROCESS COMBINED PIPELINE
# --------------------------------------------------

@app.route("/process", methods=["POST"])
def process():

    if "audio" not in request.files:
        return jsonify({
            "error": "Please upload an audio file."
        }), 400

    audio_file = request.files["audio"]

    if not audio_file.filename:
        return jsonify({
            "error": "No audio file selected."
        }), 400

    extension = audio_file.filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Unsupported audio format."
        }), 415

    temp_path = None
    converted_path = None

    try:

        # Stage 1: Save uploaded audio
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix="." + extension
        ) as temp_file:

            temp_path = temp_file.name
            audio_file.save(temp_path)

        if os.path.getsize(temp_path) == 0:
            raise ValueError("Uploaded audio is empty.")

        # Stage 2: Convert audio
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as wav_file:

            converted_path = wav_file.name

        convert_to_wav(temp_path, converted_path)

        # Stage 3: Speech transcription
        transcription = transcribe_audio(converted_path)

        transcript = transcription["transcript"]

        # Stage 4: Language analysis
        analysis = analyze_transcript(transcript)

        # Stage 5: Generate lecture study materials
        study_materials = generate_study_materials(
           transcript,
           analysis
        )

        # Stage 6: Calculate domain metric
        duration_minutes = (
            transcription["duration_seconds"] / 60
        )

        unique_terms = len(study_materials["glossary"])

        key_terms_per_minute = (
            round(unique_terms / duration_minutes, 2)
            if duration_minutes > 0 else 0
        )

        # Stage 7: Return complete pipeline
        return jsonify({
            "transcription": transcription,

            "analysis": analysis,

            "study_materials": study_materials,

            "metrics": {
                "unique_key_terms": unique_terms,
                "audio_duration_minutes": round(
                    duration_minutes, 2
                ),
                "key_terms_per_minute": key_terms_per_minute
            }

        }), 200

    except TimeoutError as error:

        return jsonify({
            "error": str(error)
        }), 504

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 422

    except requests.RequestException as error:

        print("Azure connection error:", str(error))

        return jsonify({
            "error": "Could not connect to Azure AI Language."
        }), 502

    except RuntimeError as error:

        print("Pipeline error:", str(error))

        return jsonify({
            "error": "Audio processing pipeline failed. "
                     "Check your Azure credentials and services."
        }), 500

    except Exception as error:

        print("Unexpected pipeline error:", str(error))

        return jsonify({
            "error": "An unexpected processing error occurred."
        }), 500

    finally:

        # Remove temporary files
        for path in [temp_path, converted_path]:

            if path and os.path.exists(path):

                try:
                    os.remove(path)

                except OSError as error:
                    print("Temporary file cleanup warning:", str(error))


# --------------------------------------------------
# 11. UPLOAD SIZE ERROR HANDLER
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "File too large. Maximum upload size is 25 MB."
    }), 413


# --------------------------------------------------
# 12. START FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get(
            "FLASK_DEBUG", "false"
        ).lower() == "true"
    )