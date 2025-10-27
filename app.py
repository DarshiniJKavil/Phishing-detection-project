from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from database import db, User, Message
import os
from datetime import datetime
from collections import Counter
import json
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from moviepy.editor import VideoFileClip
from werkzeug.utils import secure_filename
import speech_recognition as sr
from pydub import AudioSegment
import os
import cv2
import pytesseract
from PIL import Image
from googletrans import Translator
translator = Translator()

from PIL import Image, ImageOps, ImageFilter
import pytesseract

def imgtotext(fname):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = Image.open(fname)
    gray_image = ImageOps.grayscale(image)
    # Resize the image to enhance details.
    scale_factor = 2
    resized_image = gray_image.resize((gray_image.width * scale_factor, gray_image.height * scale_factor),resample=Image.LANCZOS)
    thresholded_image = resized_image.filter(ImageFilter.FIND_EDGES)
    improved_text = pytesseract.image_to_string(thresholded_image)

    print("Extracted Text:")
    print(improved_text)
    return(improved_text)

# Load model
model = load_model("spam.h5")

# Load tokenizer and label encoder
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///messaging.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directories
for folder in ['images', 'videos', 'audio']:
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], folder), exist_ok=True)

db.init_app(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'images': {'png', 'jpg', 'jpeg', 'gif'},
    'videos': {'mp4', 'mov', 'avi'},
    'audio': {'mp3', 'wav', 'ogg'}
}

def allowed_file(filename, file_type):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())

def get_dashboard_stats():
    # Messages sent by current user
    sent_messages = Message.query.filter_by(sender_id=current_user.id).all()
    
    # Messages received by current user
    received_messages = Message.query.filter_by(receiver_id=current_user.id).all()
    
    # Category distribution for sent messages
    sent_categories = Counter([msg.category for msg in sent_messages])
    
    # Category distribution for received messages
    received_categories = Counter([msg.category for msg in received_messages])
    
    # Media type distribution
    media_types = Counter([msg.media_type for msg in sent_messages + received_messages])
    
    return {
        'total_sent': len(sent_messages),
        'total_received': len(received_messages),
        'sent_categories': dict(sent_categories),
        'received_categories': dict(received_categories),
        'media_types': dict(media_types)
    }



def audiototext(fname):
    r = sr.Recognizer()
    # Load audio file
    audio_file = fname#"output_audio.wav"  # Replace with your file
    text=""

    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)  # You can also use r.listen(source) for live mic input
        try:
            text = r.recognize_google(audio_data)
            print("\n📝 Transcribed Text:\n")
            print(text)
        except sr.UnknownValueError:
            print("❌ Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"❌ Could not request results; {e}")
    return text

def videoaudio(fname):
    video = VideoFileClip(fname)
    audio = video.audio
    audio.write_audiofile("static/output_audio.wav")  # You can also use .wav


def clean_text(text):
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [word for word in text if word not in stopwords.words("english")]
    return " ".join(text)

def predict_text(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)
    label = le.inverse_transform([pred.argmax(axis=1)[0]])
    return label[0]

def classify_message(content):
    res=predict_text(content)
    res=res.lower()
    print(res)
    return res

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    stats = get_dashboard_stats()
    return render_template('dashboard.html', stats=stats)

@app.route('/dashboard/stats')
@login_required
def dashboard_stats():
    stats = get_dashboard_stats()
    return jsonify(stats)

@app.route('/send_message', methods=['GET', 'POST'])
@login_required
def send_message():
    users = User.query.filter(User.id != current_user.id).all()
    
    if request.method == 'POST':
        receiver_id = request.form['receiver_id']
        content = request.form.get('content', '')
        detected = translator.detect(content)
        print("Detected Language:", detected.lang)
        # Step 2: Translate to English
        translated = translator.translate(content, dest='en')
        print("English Translation:", translated.text)
        content=translated.text
        category = classify_message(content)
        #category = ""#request.form.get('category', 'General')
        media_file = request.files.get('media_file')
        
        
        
        
        
        message = Message(
            sender_id=current_user.id,
            receiver_id=receiver_id,
            content=content,
            category=category,
            media_type='text'
        )
        
        if media_file and media_file.filename:
            file_type = request.form.get('file_type', 'images')
            
            if allowed_file(media_file.filename, file_type):
                filename = secure_filename(f"{datetime.utcnow().timestamp()}_{media_file.filename}")
                upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], file_type)
                media_file.save(os.path.join(upload_folder, filename))
                file_path=os.path.join(upload_folder, filename)
                if file_type=="images":
                    content=imgtotext(file_path)
                    detected = translator.detect(content)
                    print("Detected Language:", detected.lang)
                    # Step 2: Translate to English
                    translated = translator.translate(content, dest='en')
                    print("English Translation:", translated.text)
                    content=translated.text

                    category = classify_message(content)
                    
                if file_type=="videos":
                    videoaudio(file_path)
                    content=audiototext('static/output_audio.wav')
                    print(content)
                    detected = translator.detect(content)
                    print("Detected Language:", detected.lang)
                    # Step 2: Translate to English
                    translated = translator.translate(content, dest='en')
                    print("English Translation:", translated.text)
                    content=translated.text
                    category = classify_message(content)

                if file_type=="audio":
                    content=audiototext(file_path)
                    detected = translator.detect(content)
                    print("Detected Language:", detected.lang)
                    # Step 2: Translate to English
                    translated = translator.translate(content, dest='en')
                    print("English Translation:", translated.text)
                    content=translated.text
                    category = classify_message(content)
                message.media_type = file_type
                message.media_filename = filename
                message.category=category
        else:
            file_type=""
            filename=""
            
                
        
        
        
        db.session.add(message)
        db.session.commit()
        
        flash('Message sent successfully!')
        return redirect(url_for('sent_messages'))
    
    return render_template('send_message.html', users=users)

@app.route('/inbox')
@login_required
def inbox():
    messages = Message.query.filter_by(receiver_id=current_user.id).order_by(Message.created_at.desc()).all()
    
    # Group by category
    categorized_messages = {}
    for message in messages:
        if message.category not in categorized_messages:
            categorized_messages[message.category] = []
        categorized_messages[message.category].append(message)
    
    return render_template('inbox.html', categorized_messages=categorized_messages)

@app.route('/sent')
@login_required
def sent_messages():
    messages = Message.query.filter_by(sender_id=current_user.id).order_by(Message.created_at.desc()).all()
    
    # Group by receiver
    receiver_messages = {}
    for message in messages:
        receiver_name = message.receiver.username
        if receiver_name not in receiver_messages:
            receiver_messages[receiver_name] = []
        receiver_messages[receiver_name].append(message)
    
    return render_template('sent.html', receiver_messages=receiver_messages)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        if 'profile_pic' in request.files:
            profile_pic = request.files['profile_pic']
            if profile_pic and allowed_file(profile_pic.filename, 'images'):
                filename = secure_filename(f"profile_{current_user.id}_{profile_pic.filename}")
                profile_pic.save(os.path.join(app.config['UPLOAD_FOLDER'], 'images', filename))
                current_user.profile_pic = filename
                db.session.commit()
                flash('Profile picture updated successfully!')
        
        return redirect(url_for('profile'))
    
    return render_template('profile.html')

@app.route('/uploads/<file_type>/<filename>')
@login_required
def uploaded_file(file_type, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], file_type), filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
