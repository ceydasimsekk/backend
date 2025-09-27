from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import os
import tempfile
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)  # CORS'u etkinleştir

# Spotify API için gerekli bilgiler
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

# API bilgilerini kontrol et
if not client_id or not client_secret:
    print("UYARI: Spotify API bilgileri bulunamadı veya eksik!")
    print(f"SPOTIFY_CLIENT_ID: {client_id}")
    print(f"SPOTIFY_CLIENT_SECRET: {'*****' if client_secret else 'Bulunamadı'}")

# Spotify API istemcisini oluştur
try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    ))
    print("Spotify API bağlantısı başarılı!")
except Exception as e:
    print(f"Spotify API bağlantı hatası: {e}")
    sp = None

# Modeli ve etiketleri yükle
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_emotion_model.keras")
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
le = LabelEncoder()
le.fit(emotions)

def get_random_track(emotion):
    """Spotify'dan belirli bir duyguya uygun şarkı önerisi al"""
    # Spotify API bağlantısı yoksa varsayılan şarkı öner
    if sp is None:
        default_tracks = {
            "happy": {"name": "Happy", "artist": "Pharrell Williams", "url": "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH"},
            "sad": {"name": "Someone Like You", "artist": "Adele", "url": "https://open.spotify.com/track/4kflIGfjdZJW4ot2ioixTB"},
            "angry": {"name": "Numb", "artist": "Linkin Park", "url": "https://open.spotify.com/track/2nLtzopw4rPReszdYBJU6h"},
            "fearful": {"name": "Thriller", "artist": "Michael Jackson", "url": "https://open.spotify.com/track/3S2R0EVwBSAVMd5UMgKTL0"},
            "neutral": {"name": "Imagine", "artist": "John Lennon", "url": "https://open.spotify.com/track/7pKfPomDEeI4TPT6EOYjn9"},
            "calm": {"name": "Weightless", "artist": "Marconi Union", "url": "https://open.spotify.com/track/5eSHgGiQfEB2m9ro9xqB9s"},
            "disgust": {"name": "Creep", "artist": "Radiohead", "url": "https://open.spotify.com/track/6b2oQwSGFkzsMtQruIWm2p"},
            "surprised": {"name": "Wow", "artist": "Post Malone", "url": "https://open.spotify.com/track/7fvUMiyapMsRRxr07cU8NC"}
        }
        # Eğer duygu varsayılan listede yoksa, rastgele bir şarkı seç
        return default_tracks.get(emotion.lower(), random.choice(list(default_tracks.values())))

    try:
        results = sp.search(q=f"{emotion} mood", type="track", limit=10)
        tracks = results["tracks"]["items"]
        if tracks:
            track = random.choice(tracks)
            name = track.get('name', 'Bilinmeyen Şarkı')
            artist = track['artists'][0].get('name', 'Bilinmeyen Sanatçı')
            url = track.get('external_urls', {}).get('spotify', None)

            if url:
                return {
                    "name": name,
                    "artist": artist,
                    "url": url
                }
        return None
    except Exception as e:
        print(f"Şarkı önerisi alınırken hata: {e}")
        # Hata durumunda varsayılan şarkı öner
        return {
            "name": "Imagine",
            "artist": "John Lennon",
            "url": "https://open.spotify.com/track/7pKfPomDEeI4TPT6EOYjn9"
        }

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """Ses dosyasını analiz et ve duygu tahmini yap"""
    if 'audio' not in request.files:
        return jsonify({"error": "Ses dosyası bulunamadı"}), 400

    audio_file = request.files['audio']

    # Geçici dosya oluştur
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        audio_file.save(temp_file.name)
        temp_filename = temp_file.name

    try:
        # Ses dosyasını yükle ve özellik çıkarımı yap
        audio, sr = librosa.load(temp_filename, duration=3, offset=0.0)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)

        # Model için girdi şeklini ayarla
        input_data = np.expand_dims(np.expand_dims(mfcc_scaled, axis=0), axis=2)

        # Tahmin yap
        predictions = model.predict(input_data)[0]

        # Duygu olasılıklarını hesapla
        emotion_probs = {
            str(le.inverse_transform([i])[0]): float(pred) * 100
            for i, pred in enumerate(predictions)
        }

        # En yüksek olasılıklı duyguyu bul
        dominant_emotion = max(emotion_probs, key=emotion_probs.get)

        # Spotify'dan şarkı önerisi al
        spotify_recommendation = get_random_track(dominant_emotion)

        # Sonuçları döndür
        return jsonify({
            "emotions": emotion_probs,
            "dominantEmotion": dominant_emotion,
            "spotifyRecommendation": spotify_recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Geçici dosyayı temizle
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """API sağlık kontrolü"""
    return jsonify({"status": "ok", "message": "Emotion Analysis API is running"})

if __name__ == '__main__':
    import os
    # Render/Railway gibi ortamlarda PORT env gelir; lokalde yoksa 5000 kullan
    port = int(os.getenv("PORT", 5000))
    # Dış dünyaya açılmak için 0.0.0.0 (lokalde de sorun olmaz)
    host = "0.0.0.0"
    # İstersen FLASK_DEBUG=1 ile lokalde debug açabilirsin
    debug = os.getenv("FLASK_DEBUG", "").lower() == "1"

    print(f"Emotion Analysis API başlatılıyor... http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)


