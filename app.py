from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
scaler = None

columns_order = [
    'sex', 'enrollment_duration', 'mark', 'average_grade', 'evaluation_count',
    'absence_percentage', 'absence_justified_percentage', 'has_behavior_issues',
    'submission_rate', 'correctness_rate', 'teacher_comment_score', 'is_available',
    'extra_activity_count', 'activity_score', 'parent_occupation_level',
    'address_level', 'course_count', 'course_difficulty', 'age'
]

def normalize_input(data):
    """Normalise et valide les données d'entrée selon le code de référence"""
    for col in columns_order:
        if col not in data:
            raise ValueError(f"Champ manquant : {col}")
        try:
            data[col] = float(data[col])
        except ValueError:
            raise ValueError(f"Valeur non numérique pour : {col}")
    
    return pd.DataFrame([data])[columns_order]

def load_model():
    """Charge le modèle pré-entraîné et le scaler"""
    global model, scaler
    
    try:
        if os.path.exists('models/student_performance_model.pkl'):
            model = joblib.load('models/student_performance_model.pkl')
            logger.info("Modèle chargé avec succès")
        else:
            logger.error("Aucun modèle trouvé")
            
        if os.path.exists('models/student_scaler.pkl'):
            scaler = joblib.load('models/student_scaler.pkl')
            logger.info("Scaler chargé avec succès")
        else:
            logger.error("Aucun scaler trouvé")
            
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle ou du scaler: {e}")

@app.route('/')
def index():
    """Page d'accueil"""
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'scaler_type': str(type(scaler)) if scaler is not None else None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prédiction de performance d'un étudiant - version optimisée"""
    try:
        data = request.get_json()
        
        input_df = normalize_input(data)
        print("Données d'entrée normalisées :")
        print(input_df)
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        if scaler is None:
            return jsonify({'error': 'Scaler not loaded'}), 500
        
        input_scaled = scaler.transform(input_df)
        
        print("Input final pour le modèle :")
        print(input_df)
        
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        classes = model.classes_
        confidence = proba[list(classes).index(pred)]
        
        recommendations = generate_recommendations(pred, data)
        
        response = {
            'prediction': round(float(pred), 4),
            'predicted_category': round(float(pred), 4),  
            'confidence': round(float(confidence), 4),
            'probabilities': {
                str(classes[i]): round(float(proba[i]), 4) 
                for i in range(len(classes))
            },
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'model_type': str(type(model).__name__)
        }
        
        logger.info(f"Prédiction effectuée: {pred} (confiance: {confidence:.4f})")
        return jsonify(response), 200
        
    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({'error': str(e)}), 400

def generate_recommendations(prediction, student_data):
    """Génère des recommandations basées sur la prédiction"""
    recommendations = []
    
    if prediction == 0 :
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
    
    elif prediction == 2:
        if student_data.get('average_grade', 0) < 12:
            recommendations.append("Proposer du soutien scolaire pour améliorer les résultats")
        
        if student_data.get('extra_activity_count', 0) < 2:
            recommendations.append("Encourager davantage d'activités extra-scolaires")
        
        recommendations.append("Maintenir l'effort actuel et viser l'excellence")
    
    else:  
        recommendations.append("Excellent profil ! Proposer des défis supplémentaires")
        recommendations.append("Considérer des programmes d'enrichissement")
        
        if student_data.get('extra_activity_count', 0) < 3:
            recommendations.append("Encourager le leadership dans les activités extra-scolaires")
    
    return recommendations

@app.route('/model/info', methods=['GET'])
def model_info():
    """Informations sur le modèle"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                feature_importance[columns_order[i]] = float(importance)
        
        info = {
            'model_type': type(model).__name__,
            'features': columns_order,
            'feature_importance': feature_importance,
            'classes': model.classes_.tolist() if hasattr(model, 'classes_') else [],
            'n_features': len(columns_order),
            'scaler_available': scaler is not None,
            'scaler_type': str(type(scaler)) if scaler is not None else None
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos du modèle: {e}")
        return jsonify({'error': 'Failed to get model info'}), 500

@app.route('/students/calculate-metrics', methods=['POST'])
def calculate_student_metrics():
    """Calcule les métriques d'un étudiant à partir de ses données de base"""
    try:
        data = request.get_json()
        
        current_date = datetime.now()
        enrollment_date = datetime.strptime(data.get('first_enrollment_date'), '%Y-%m-%d')
        enrollment_duration = (current_date - enrollment_date).days // 30
        
        grades = data.get('grades', [])
        mark_average = sum(grades) / len(grades) if grades else 0
        
        absences = data.get('absences', [])
        total_school_days = 180
        absence_percentage = (len(absences) / total_school_days) * 100
        justified_absences = sum(1 for abs in absences if abs.get('justified', False))
        absence_justified_percentage = (justified_absences / len(absences) * 100) if absences else 0
        attendance_rate = 100 - absence_percentage
        
        disciplinary_notes = data.get('disciplinary_notes', [])
        has_behavior_issues = 1 if disciplinary_notes else 0
        
        homework_submissions = data.get('homework_submissions', [])
        submitted_homework = sum(1 for hw in homework_submissions if hw.get('submitted', False))
        submission_rate = (submitted_homework / len(homework_submissions) * 100) if homework_submissions else 0
        
        graded_homework = [hw for hw in homework_submissions if hw.get('submitted') and hw.get('grade')]
        correctness_rate = sum(hw['grade'] for hw in graded_homework) / len(graded_homework) if graded_homework else 0
        
        teacher_comments = data.get('teacher_comments', [])
        teacher_comment_score = sum(comment['score'] for comment in teacher_comments) / len(teacher_comments) if teacher_comments else 3
        
        extra_activities = data.get('extra_activities', [])
        extra_activity_count = len(extra_activities)
        activity_score = extra_activity_count * 2
        
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    load_model()
    
    logger.info("Démarrage de l'API de prédiction de performance des étudiants")
    app.run(debug=True, host='0.0.0.0', port=8000)