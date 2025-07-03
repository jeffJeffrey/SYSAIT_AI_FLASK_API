from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime, timedelta
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from contextlib import contextmanager

# Configuration de l'application Flask
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = '456fsd4g56df45d1gb56df4g65df4g56esdfg456f'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Extensions
CORS(app)
jwt = JWTManager(app)

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales pour le modèle
model = None
scaler = None
feature_columns = [
    'primary_school', 'school_id', 'classroom_id', 'sex', 'age', 
    'enrollment_duration', 'mark', 'average_grade', 'evaluation_count',
    'absence_percentage', 'absence_justified_percentage', 'has_behavior_issues',
    'submission_rate', 'correctness_rate', 'teacher_comment_score',
    'is_available', 'extra_activity_count', 'activity_score',
    'parent_occupation_level', 'address_level', 'course_count', 'course_difficulty'
]

# Base de données SQLite pour les utilisateurs (simple pour la démo)
@contextmanager
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialise la base de données"""
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Création d'un utilisateur admin par défaut
        admin_hash = generate_password_hash('admin123')
        conn.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, email)
            VALUES (?, ?, ?)
        ''', ('admin', admin_hash, 'admin@school.com'))
        
        conn.commit()

def generate_synthetic_data(n_samples=1000):
    """Génère des données synthétiques pour l'entraînement"""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # Génération de données réalistes
        primary_school = np.random.randint(1, 6)
        school_id = np.random.randint(1, 4)
        classroom_id = np.random.randint(1, 21)
        sex = np.random.randint(0, 2)  # 0=F, 1=M
        age = np.random.randint(11, 17)
        enrollment_duration = np.random.randint(1, 48)  # en mois
        
        # Notes et évaluations (corrélées)
        base_performance = np.random.normal(0.6, 0.2)  # Performance de base
        mark = np.clip(base_performance * 20 + np.random.normal(0, 2), 0, 20)
        average_grade = mark + np.random.normal(0, 1)
        evaluation_count = np.random.randint(1, 15)
        
        # Absences (inversement corrélées à la performance)
        absence_percentage = np.clip(np.random.exponential(5) + (1-base_performance)*10, 0, 50)
        absence_justified_percentage = np.random.uniform(20, 80)
        
        # Comportement (corrélé à la performance)
        has_behavior_issues = 1 if (base_performance < 0.4 and np.random.random() < 0.7) else 0
        
        # Devoirs (corrélés à la performance)
        submission_rate = np.clip(base_performance * 100 + np.random.normal(0, 15), 0, 100)
        correctness_rate = np.clip(submission_rate * 0.8 + np.random.normal(0, 10), 0, 100)
        
        # Commentaires professeurs (corrélés à la performance)
        teacher_comment_score = np.clip(base_performance * 5 + np.random.normal(0, 0.5), 1, 5)
        
        # Activités extra-scolaires (légèrement corrélées à la performance)
        extra_activity_count = np.random.poisson(base_performance * 3)
        activity_score = extra_activity_count * 2 + np.random.randint(0, 3)
        
        # Autres variables
        is_available = np.random.choice([0, 1], p=[0.1, 0.9])
        parent_occupation_level = np.random.randint(1, 5)
        address_level = np.random.randint(1, 4)
        course_count = np.random.randint(4, 8)
        course_difficulty = np.random.randint(1, 5)
        
        # Catégorie cible basée sur la performance globale
        performance_score = (
            mark/20 * 0.3 +
            submission_rate/100 * 0.2 +
            (100-absence_percentage)/100 * 0.2 +
            teacher_comment_score/5 * 0.2 +
            (1-has_behavior_issues) * 0.1
        )
        
        if performance_score >= 0.75:
            category = 'high_potential'
        elif performance_score >= 0.5:
            category = 'medium_potential'
        else:
            category = 'at_risk'
        
        data.append([
            primary_school, school_id, classroom_id, sex, age, enrollment_duration,
            mark, average_grade, evaluation_count, absence_percentage,
            absence_justified_percentage, has_behavior_issues, submission_rate,
            correctness_rate, teacher_comment_score, is_available,
            extra_activity_count, activity_score, parent_occupation_level,
            address_level, course_count, course_difficulty, category
        ])
    
    columns = feature_columns + ['category']
    return pd.DataFrame(data, columns=columns)

def train_model():
    """Entraîne le modèle de machine learning"""
    global model, scaler
    
    logger.info("Génération des données d'entraînement...")
    df = generate_synthetic_data(1000)
    
    # Séparation des features et du target
    X = df[feature_columns]
    y = df['category']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement du modèle
    logger.info("Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Précision du modèle: {accuracy:.3f}")
    logger.info(f"Rapport de classification:\n{classification_report(y_test, y_pred)}")
    
    # Sauvegarde du modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/student_performance_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    logger.info("Modèle sauvegardé avec succès")

def load_model():
    """Charge le modèle pré-entraîné"""
    global model, scaler
    
    try:
        if os.path.exists('models/student_performance_model.pkl'):
            model = joblib.load('models/student_performance_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            logger.info("Modèle chargé avec succès")
        else:
            logger.info("Aucun modèle trouvé, entraînement d'un nouveau modèle...")
            train_model()
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        train_model()

# Routes d'authentification
@app.route('/register', methods=['POST'])
def register():
    """Enregistrement d'un nouvel utilisateur"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        password_hash = generate_password_hash(password)
        
        with get_db_connection() as conn:
            try:
                conn.execute('''
                    INSERT INTO users (username, password_hash, email)
                    VALUES (?, ?, ?)
                ''', (username, password_hash, email))
                conn.commit()
                
                return jsonify({'message': 'User registered successfully'}), 201
            except sqlite3.IntegrityError:
                return jsonify({'error': 'Username already exists'}), 409
                
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/login', methods=['POST'])
def login():
    """Connexion utilisateur"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        with get_db_connection() as conn:
            user = conn.execute('''
                SELECT * FROM users WHERE username = ?
            ''', (username,)).fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                access_token = create_access_token(identity=username)
                return jsonify({
                    'access_token': access_token,
                    'user': {
                        'id': user['id'],
                        'username': user['username'],
                        'email': user['email']
                    }
                }), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
                
    except Exception as e:
        logger.error(f"Erreur lors de la connexion: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Routes principales
@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    """Prédiction de performance d'un étudiant"""
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        # Validation des données d'entrée
        missing_fields = []
        for field in feature_columns:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Préparation des données pour la prédiction
        features = []
        for field in feature_columns:
            value = data[field]
            # Validation et conversion des types
            if isinstance(value, (int, float)):
                features.append(float(value))
            else:
                return jsonify({'error': f'Invalid type for field {field}'}), 400
        
        # Normalisation
        features_scaled = scaler.transform([features])
        
        # Prédiction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mapping des classes
        classes = model.classes_
        prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        
        # Calcul d'un score de confiance
        confidence = float(max(probabilities))
        
        # Recommandations basées sur la prédiction
        recommendations = generate_recommendations(prediction, data)
        
        response = {
            'predicted_category': prediction,
            'confidence': confidence,
            'probabilities': prob_dict,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'user': current_user
        }
        
        logger.info(f"Prédiction effectuée pour l'utilisateur {current_user}: {prediction}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

def generate_recommendations(prediction, student_data):
    """Génère des recommandations basées sur la prédiction"""
    recommendations = []
    
    if prediction == 'at_risk':
        if student_data.get('absence_percentage', 0) > 20:
            recommendations.append("Améliorer l'assiduité - le taux d'absence est élevé")
        
        if student_data.get('submission_rate', 0) < 70:
            recommendations.append("Mettre en place un suivi personnalisé pour les devoirs")
        
        if student_data.get('has_behavior_issues', 0) == 1:
            recommendations.append("Prévoir un accompagnement comportemental")
        
        if student_data.get('teacher_comment_score', 0) < 3:
            recommendations.append("Renforcer la motivation et l'engagement")
        
        if student_data.get('extra_activity_count', 0) == 0:
            recommendations.append("Encourager la participation à des activités extra-scolaires")
    
    elif prediction == 'medium_potential':
        if student_data.get('average_grade', 0) < 12:
            recommendations.append("Proposer du soutien scolaire pour améliorer les résultats")
        
        if student_data.get('extra_activity_count', 0) < 2:
            recommendations.append("Encourager davantage d'activités extra-scolaires")
        
        recommendations.append("Maintenir l'effort actuel et viser l'excellence")
    
    else:  # high_potential
        recommendations.append("Excellent profil ! Proposer des défis supplémentaires")
        recommendations.append("Considérer des programmes d'enrichissement")
        
        if student_data.get('extra_activity_count', 0) < 3:
            recommendations.append("Encourager le leadership dans les activités extra-scolaires")
    
    return recommendations

@app.route('/model/info', methods=['GET'])
@jwt_required()
def model_info():
    """Informations sur le modèle"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Importance des features
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                feature_importance[feature_columns[i]] = float(importance)
        
        info = {
            'model_type': type(model).__name__,
            'features': feature_columns,
            'feature_importance': feature_importance,
            'classes': model.classes_.tolist() if hasattr(model, 'classes_') else [],
            'n_features': len(feature_columns)
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos du modèle: {e}")
        return jsonify({'error': 'Failed to get model info'}), 500

@app.route('/model/retrain', methods=['POST'])
@jwt_required()
def retrain_model():
    """Re-entraîne le modèle (admin seulement)"""
    try:
        current_user = get_jwt_identity()
        
        # Vérification des droits admin (simplifié)
        if current_user != 'admin':
            return jsonify({'error': 'Admin rights required'}), 403
        
        logger.info(f"Re-entraînement du modèle initié par {current_user}")
        train_model()
        
        return jsonify({
            'message': 'Model retrained successfully',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur lors du re-entraînement: {e}")
        return jsonify({'error': 'Retraining failed'}), 500

@app.route('/students/calculate-metrics', methods=['POST'])
@jwt_required()
def calculate_student_metrics():
    """Calcule les métriques d'un étudiant à partir de ses données de base"""
    try:
        data = request.get_json()
        
        # Calculs similaires à ceux du frontend
        current_date = datetime.now()
        enrollment_date = datetime.strptime(data.get('first_enrollment_date'), '%Y-%m-%d')
        enrollment_duration = (current_date - enrollment_date).days // 30
        
        # Calcul de la moyenne
        grades = data.get('grades', [])
        mark_average = sum(grades) / len(grades) if grades else 0
        
        # Calcul des absences
        absences = data.get('absences', [])
        total_school_days = 180
        absence_percentage = (len(absences) / total_school_days) * 100
        justified_absences = sum(1 for abs in absences if abs.get('justified', False))
        absence_justified_percentage = (justified_absences / len(absences) * 100) if absences else 0
        attendance_rate = 100 - absence_percentage
        
        # Calcul comportement
        disciplinary_notes = data.get('disciplinary_notes', [])
        has_behavior_issues = 1 if disciplinary_notes else 0
        
        # Calcul devoirs
        homework_submissions = data.get('homework_submissions', [])
        submitted_homework = sum(1 for hw in homework_submissions if hw.get('submitted', False))
        submission_rate = (submitted_homework / len(homework_submissions) * 100) if homework_submissions else 0
        
        graded_homework = [hw for hw in homework_submissions if hw.get('submitted') and hw.get('grade')]
        correctness_rate = sum(hw['grade'] for hw in graded_homework) / len(graded_homework) if graded_homework else 0
        
        # Calcul commentaires professeurs
        teacher_comments = data.get('teacher_comments', [])
        teacher_comment_score = sum(comment['score'] for comment in teacher_comments) / len(teacher_comments) if teacher_comments else 3
        
        # Activités extra-scolaires
        extra_activities = data.get('extra_activities', [])
        extra_activity_count = len(extra_activities)
        activity_score = extra_activity_count * 2
        
        # Autres métriques
        evaluation_count = len(grades)
        course_subjects = data.get('course_subjects', [])
        course_count = len(course_subjects)
        course_difficulty = min(5, max(1, course_count // 2))
        
        metrics = {
            'enrollment_duration': enrollment_duration,
            'mark': mark_average,
            'average_grade': mark_average,
            'evaluation_count': evaluation_count,
            'absence_percentage': absence_percentage,
            'absence_justified_percentage': absence_justified_percentage,
            'attendance_rate': attendance_rate,
            'has_behavior_issues': has_behavior_issues,
            'submission_rate': submission_rate,
            'correctness_rate': correctness_rate,
            'teacher_comment_score': teacher_comment_score,
            'is_available': 1,
            'extra_activity_count': extra_activity_count,
            'activity_score': activity_score,
            'course_count': course_count,
            'course_difficulty': course_difficulty
        }
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques: {e}")
        return jsonify({'error': 'Calculation failed'}), 500

# Gestion des erreurs
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token'}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({'error': 'Authorization token is required'}), 401

# Initialisation
if __name__ == '__main__':
    # Initialisation de la base de données
    init_db()
    
    # Chargement du modèle
    load_model()
    
    # Démarrage de l'application
    logger.info("Démarrage de l'API de prédiction de performance des étudiants")
    app.run(debug=True, host='0.0.0.0', port=8000)