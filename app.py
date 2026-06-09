"""
Student Dropout Prediction System - Flask Web Application
FYP: An Explainable ML Model for Early Prediction of Student Dropout
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import plotly.io as pio

def fig_to_json(fig):
    """Serialize Plotly figure to JSON without binary arrays."""
    import numpy as np
    d = json.loads(fig.to_json())
    _purge_bdata(d)
    return json.dumps(d, default=str)

def _purge_bdata(obj):
    """Recursively replace {dtype,bdata} dicts with actual decoded arrays."""
    import base64, struct
    dtype_map = {'f8': ('d', 8), 'f4': ('f', 4), 'i4': ('i', 4), 'i2': ('h', 2), 'i1': ('b', 1), 'u1': ('B', 1), 'u2': ('H', 2), 'u4': ('I', 4)}
    
    if isinstance(obj, dict):
        if 'bdata' in obj and 'dtype' in obj:
            dt = obj['dtype']
            if dt in dtype_map:
                fmt, sz = dtype_map[dt]
                raw = base64.b64decode(obj['bdata'])
                count = len(raw) // sz
                vals = list(struct.unpack(f'<{count}{fmt}', raw))
                shape = obj.get('shape')
                if shape and len(shape) == 2:
                    rows, cols = shape
                    return [vals[i*cols:(i+1)*cols] for i in range(rows)]
                return vals
            return obj
        else:
            for k, v in list(obj.items()):
                result = _purge_bdata(v)
                if result is not v:
                    obj[k] = result
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            result = _purge_bdata(v)
            if result is not v:
                obj[i] = result
    return obj

from flask import (Flask, render_template, request, redirect, url_for, 
                   flash, session, jsonify, send_file)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fyp-dropout-prediction-2026-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dropout_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('models/shap_explainer.pkl', 'rb') as f:
    shap_explainer = pickle.load(f)
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
with open('models/numerical_cols.pkl', 'rb') as f:
    numerical_cols = pickle.load(f)
with open('models/metrics.json', 'r') as f:
    model_metrics = json.load(f)

df_raw = pd.read_csv('data/student_dropout_3000.csv')
df_processed = pd.read_csv('data/student_dropout_processed.csv')

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(50), default='teacher')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class PasswordResetToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    used = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

SMTP_HOST = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USER = 'exemplary42@gmail.com'
SMTP_PASS = 'mnepobkbjcqmubsu'
SMTP_FROM_NAME = 'EduPredict System'

def send_email(to_email, subject, html_body):
    """Send email via Gmail SMTP."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    if SMTP_USER == 'your.gmail@gmail.com':
        print("[SMTP] Email not configured - skipping send")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = f'{SMTP_FROM_NAME} <{SMTP_USER}>'
        msg['To'] = to_email
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS.replace(' ', ''))
            server.sendmail(SMTP_USER, to_email, msg.as_string())
        
        print(f"[SMTP] Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"[SMTP] Error: {e}")
        return False

def preprocess_student(data_dict):
    """Preprocess a single student record for prediction."""
    df_single = pd.DataFrame([data_dict])
    
    binary_cols = ['School', 'Gender', 'Address', 'Family_Size', 'Parental_Status',
                   'School_Support', 'Family_Support', 'Extra_Paid_Class',
                   'Extra_Curricular_Activities', 'Attended_Nursery',
                   'Wants_Higher_Education', 'Internet_Access', 'In_Relationship']
    
    for col in binary_cols:
        if col in label_encoders and col in df_single.columns:
            try:
                df_single[col] = label_encoders[col].transform(df_single[col].astype(str))
            except ValueError:
                df_single[col] = 0
    
    nominal_cols = ['Mother_Job', 'Father_Job', 'Reason_for_Choosing_School', 'Guardian']
    df_single = pd.get_dummies(df_single, columns=nominal_cols, drop_first=True)
    
    for col in feature_names:
        if col not in df_single.columns:
            df_single[col] = 0
    
    df_single = df_single[feature_names]
    
    for col in df_single.columns:
        if df_single[col].dtype == bool:
            df_single[col] = df_single[col].astype(int)
    
    num_cols_present = [c for c in numerical_cols if c in df_single.columns]
    df_single[num_cols_present] = scaler.transform(df_single[num_cols_present])
    
    return df_single

def get_shap_explanation(features_df):
    """Get SHAP values for a single prediction."""
    shap_vals = shap_explainer.shap_values(features_df)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1]
    else:
        sv = shap_vals[0]
    
    explanation = []
    for feat, val in sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:10]:
        explanation.append({
            'feature': feat,
            'shap_value': float(val),
            'direction': 'increases' if val > 0 else 'decreases',
            'impact': abs(float(val))
        })
    return explanation

def generate_dashboard_plots():
    """Generate all Plotly dashboard charts."""
    plots = {}
    
    def neutral(fig, h=400):
        fig.update_layout(
            template='none',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='DM Sans, sans-serif'),
            height=h,
            margin=dict(l=40, r=20, t=50, b=40)
        )
        fig.update_xaxes(gridcolor='rgba(148,163,184,.1)', zerolinecolor='rgba(148,163,184,.15)')
        fig.update_yaxes(gridcolor='rgba(148,163,184,.1)', zerolinecolor='rgba(148,163,184,.15)')
        return fig
    
    dropout_counts = df_raw['Dropped_Out'].value_counts()
    fig_pie = px.pie(values=dropout_counts.values, names=['Not At-Risk', 'At-Risk'],
                     color_discrete_sequence=['#34d399', '#f43f5e'], title='Student Dropout Risk Distribution')
    neutral(fig_pie, 380)
    plots['pie_chart'] = fig_to_json(fig_pie)
    
    fig_box = make_subplots(rows=1, cols=3, subplot_titles=['Grade 1', 'Grade 2', 'Final Grade'])
    for i, grade in enumerate(['Grade_1', 'Grade_2', 'Final_Grade'], 1):
        for status, color in [(False, '#34d399'), (True, '#f43f5e')]:
            data = df_raw[df_raw['Dropped_Out'] == status][grade]
            fig_box.add_trace(go.Box(y=data, name=f"{'At-Risk' if status else 'Safe'}", marker_color=color, showlegend=(i==1)), row=1, col=i)
    neutral(fig_box, 400)
    fig_box.update_layout(title='Grade Distribution by Dropout Status')
    plots['box_chart'] = fig_to_json(fig_box)
    
    fi = model_metrics['feature_importance'][:10]
    fig_bar = go.Figure(go.Bar(
        x=[f['shap_importance'] for f in fi],
        y=[f['feature'].replace('_',' ') for f in fi],
        orientation='h', marker_color='#7c3aed'
    ))
    neutral(fig_bar, 400)
    fig_bar.update_layout(title='Top 10 Feature Importance (SHAP)', yaxis=dict(autorange='reversed'))
    plots['bar_chart'] = fig_to_json(fig_bar)
    
    df_sample = df_raw.sample(min(500, len(df_raw)), random_state=42)
    fig_bubble = px.scatter(df_sample, x='Number_of_Absences', y='Final_Grade',
        color=df_sample['Dropped_Out'].map({True:'At-Risk', False:'Safe'}),
        size='Number_of_Failures', size_max=20,
        color_discrete_map={'At-Risk':'#f43f5e','Safe':'#34d399'},
        title='Absences vs Final Grade')
    neutral(fig_bubble, 400)
    plots['bubble_chart'] = fig_to_json(fig_bubble)
    
    num_cols = ['Grade_1','Grade_2','Final_Grade','Number_of_Failures',
                'Study_Time','Number_of_Absences','Free_Time','Going_Out','Health_Status','Age']
    df_corr = df_raw[num_cols].copy().apply(pd.to_numeric, errors='coerce')
    df_corr['Dropped_Out'] = df_raw['Dropped_Out'].map({True:1, False:0, 'True':1, 'False':0}).astype(float)
    corr_matrix = df_corr.corr(numeric_only=True)
    
    z_vals = corr_matrix.values.tolist()
    labels_list = corr_matrix.columns.tolist()
    text_vals = [[f'{v:.2f}' for v in row] for row in z_vals]
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=z_vals, x=labels_list, y=labels_list,
        text=text_vals, texttemplate='%{text}', textfont=dict(size=11, color='#e0e0e0'),
        colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1
    ))
    neutral(fig_heatmap, 520)
    fig_heatmap.update_layout(title='Correlation Heatmap of Key Features')
    plots['heatmap'] = fig_to_json(fig_heatmap)
    
    fig_gauge = make_subplots(rows=1, cols=4, specs=[[{'type':'indicator'}]*4],
                              subplot_titles=['Accuracy','Precision','Recall','F1-Score'])
    for i, (m, v) in enumerate([('Accuracy',model_metrics['accuracy']),('Precision',model_metrics['precision']),
                                 ('Recall',model_metrics['recall']),('F1',model_metrics['f1_score'])]):
        fig_gauge.add_trace(go.Indicator(mode="gauge+number", value=v*100,
            number={'suffix':'%','font':{'size':22,'color':'#94a3b8'}},
            gauge={'axis':{'range':[0,100],'tickfont':{'color':'#94a3b8'}},'bar':{'color':'#7c3aed'},
                   'steps':[{'range':[0,60],'color':'rgba(244,63,94,.2)'},{'range':[60,80],'color':'rgba(251,191,36,.2)'},
                            {'range':[80,100],'color':'rgba(52,211,153,.2)'}]}), row=1, col=i+1)
    neutral(fig_gauge, 260)
    plots['gauge_chart'] = fig_to_json(fig_gauge)
    
    gd = df_raw.groupby(['Gender','Dropped_Out']).size().reset_index(name='Count')
    gd['Status'] = gd['Dropped_Out'].map({True:'At-Risk', False:'Safe'})
    fig_gender = px.bar(gd, x='Gender', y='Count', color='Status', barmode='group',
                        color_discrete_map={'At-Risk':'#f43f5e','Safe':'#34d399'}, title='Dropout by Gender')
    neutral(fig_gender, 350)
    plots['gender_chart'] = fig_to_json(fig_gender)
    
    fig_age = px.histogram(df_raw, x='Age', color=df_raw['Dropped_Out'].map({True:'At-Risk',False:'Safe'}),
                           barmode='overlay', color_discrete_map={'At-Risk':'#f43f5e','Safe':'#34d399'},
                           title='Age Distribution', opacity=0.7)
    neutral(fig_age, 350)
    plots['age_chart'] = fig_to_json(fig_age)
    
    return plots

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name')
        role = request.form.get('role', 'teacher')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('register.html')
        
        user = User(username=username, email=email, full_name=full_name, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        send_email(email, 'Welcome to EduPredict - Your Account is Ready!',
            f'<div style="max-width:560px;margin:0 auto;font-family:Arial,sans-serif">'
            f'<div style="background:linear-gradient(135deg,#7c3aed,#ec4899);padding:40px 32px;border-radius:16px 16px 0 0;text-align:center">'
            f'<h1 style="color:#fff;font-size:28px;margin:0 0 8px">Welcome to EduPredict</h1>'
            f'<p style="color:rgba(255,255,255,.8);font-size:14px;margin:0">AI-Powered Student Dropout Prediction System</p></div>'
            f'<div style="background:#ffffff;padding:32px;border:1px solid #e5e7eb;border-top:none">'
            f'<p style="font-size:16px;color:#111827;margin:0 0 16px">Hi <strong>{full_name}</strong>,</p>'
            f'<p style="font-size:15px;color:#374151;line-height:1.7;margin:0 0 24px">Your account has been created successfully. You now have access to our explainable AI system that uses <strong>Random Forest + SHAP</strong> to predict student dropout risk and explain exactly why.</p>'
            f'<div style="background:#f8f9fc;border-radius:12px;padding:20px;margin:0 0 24px">'
            f'<p style="margin:0 0 8px;font-size:13px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;font-weight:700">Your Account Details</p>'
            f'<p style="margin:0 0 4px;font-size:15px;color:#111827"><strong>Username:</strong> {username}</p>'
            f'<p style="margin:0;font-size:15px;color:#111827"><strong>Role:</strong> {role.capitalize()}</p></div>'
            f'<a href="{request.host_url}" style="display:block;text-align:center;background:#7c3aed;color:#fff;padding:14px 28px;border-radius:12px;text-decoration:none;font-weight:600;font-size:15px;margin:0 0 24px">Log In to EduPredict</a>'
            f'<div style="border-top:1px solid #e5e7eb;padding-top:20px;text-align:center">'
            f'<p style="font-size:12px;color:#9ca3af;margin:0">EduPredict - UiTM Faculty of Computer and Mathematical Sciences</p></div></div></div>')
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            import secrets
            token = secrets.token_urlsafe(32)
            reset_token = PasswordResetToken(user_id=user.id, token=token)
            db.session.add(reset_token)
            db.session.commit()
            reset_url = request.host_url.rstrip('/') + url_for('reset_password', token=token)
            send_email(email, 'EduPredict - Password Reset',
                f'<h2>Password Reset Request</h2>'
                f'<p>Hi {user.full_name},</p>'
                f'<p>Click the link below to reset your password:</p>'
                f'<p><a href="{reset_url}" style="background:#7c3aed;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:bold">Reset Password</a></p>'
                f'<p>Or copy this link: {reset_url}</p>'
                f'<p>This link expires in 1 hour.</p>'
                f'<p>If you did not request this, ignore this email.</p>')
        flash('If that email exists in our system, a reset link has been sent.', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    reset_token = PasswordResetToken.query.filter_by(token=token, used=False).first()
    if not reset_token:
        flash('Invalid or expired reset link.', 'danger')
        return redirect(url_for('login'))
    age = (datetime.utcnow() - reset_token.created_at).total_seconds()
    if age > 3600:
        flash('Reset link has expired. Please request a new one.', 'danger')
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('reset_password.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('reset_password.html')
        user = db.session.get(User, reset_token.user_id)
        user.set_password(password)
        reset_token.used = True
        db.session.commit()
        flash('Password reset successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html')

@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash('Access denied. Admin only.', 'danger')
        return redirect(url_for('dashboard'))
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/add-user', methods=['POST'])
@login_required
def admin_add_user():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    full_name = request.form.get('full_name')
    role = request.form.get('role', 'teacher')
    if User.query.filter_by(username=username).first():
        flash('Username already exists.', 'danger')
        return redirect(url_for('admin_users'))
    if User.query.filter_by(email=email).first():
        flash('Email already registered.', 'danger')
        return redirect(url_for('admin_users'))
    user = User(username=username, email=email, full_name=full_name, role=role)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    send_email(email, 'EduPredict - Your Account Has Been Created',
        f'<h2>Welcome to EduPredict, {full_name}!</h2>'
        f'<p>An administrator has created an account for you.</p>'
        f'<p><strong>Username:</strong> {username}</p>'
        f'<p><strong>Temporary Password:</strong> {password}</p>'
        f'<p><strong>Role:</strong> {role.capitalize()}</p>'
        f'<p>Please log in and change your password.</p>'
        f'<p>Login at: {request.host_url}</p>')
    flash(f'User {full_name} added successfully. Email notification sent.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/delete-user/<int:user_id>')
@login_required
def admin_delete_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    user = db.session.get(User, user_id)
    if not user or user.username == 'admin':
        flash('Cannot delete this user.', 'danger')
        return redirect(url_for('admin_users'))
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.full_name} deleted.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/dashboard')
@login_required
def dashboard():
    plots = generate_dashboard_plots()
    
    total_students = len(df_raw)
    at_risk = int(df_raw['Dropped_Out'].sum())
    not_at_risk = total_students - at_risk
    risk_pct = (at_risk / total_students) * 100
    
    stats = {
        'total': total_students,
        'at_risk': at_risk,
        'not_at_risk': not_at_risk,
        'risk_pct': f"{risk_pct:.1f}",
        'accuracy': f"{model_metrics['accuracy']*100:.2f}",
        'f1': f"{model_metrics['f1_score']*100:.2f}"
    }
    
    return render_template('dashboard.html', plots=plots, stats=stats, metrics=model_metrics)

@app.route('/students')
@login_required
def students():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    search = request.args.get('search', '')
    filter_status = request.args.get('status', 'all')
    
    df_display = df_raw.copy()
    df_display['id'] = df_display.index
    
    if filter_status == 'at_risk':
        df_display = df_display[df_display['Dropped_Out'] == True]
    elif filter_status == 'safe':
        df_display = df_display[df_display['Dropped_Out'] == False]
    
    if search:
        mask = df_display.apply(lambda x: search.lower() in str(x).lower(), axis=1)
        df_display = df_display[mask]
    
    total = len(df_display)
    total_pages = (total + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    
    students_list = df_display.iloc[start:end].to_dict('records')
    for s in students_list:
        s['status'] = 'At-Risk' if s['Dropped_Out'] in [True, 'True'] else 'Not At-Risk'
    
    return render_template('students.html', students=students_list, 
                          page=page, total_pages=total_pages, total=total,
                          search=search, filter_status=filter_status)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        student_data = {}
        
        fields = ['School', 'Gender', 'Age', 'Address', 'Family_Size', 'Parental_Status',
                  'Mother_Education', 'Father_Education', 'Mother_Job', 'Father_Job',
                  'Reason_for_Choosing_School', 'Guardian', 'Travel_Time', 'Study_Time',
                  'Number_of_Failures', 'School_Support', 'Family_Support', 'Extra_Paid_Class',
                  'Extra_Curricular_Activities', 'Attended_Nursery', 'Wants_Higher_Education',
                  'Internet_Access', 'In_Relationship', 'Family_Relationship', 'Free_Time',
                  'Going_Out', 'Weekend_Alcohol_Consumption', 'Weekday_Alcohol_Consumption',
                  'Health_Status', 'Number_of_Absences', 'Grade_1', 'Grade_2', 'Final_Grade']
        
        for field in fields:
            val = request.form.get(field)
            if val is not None:
                try:
                    student_data[field] = int(val) if val.isdigit() else val
                except:
                    student_data[field] = val
        
        features = preprocess_student(student_data)
        prediction = rf_model.predict(features)[0]
        probability = rf_model.predict_proba(features)[0]
        
        risk_level = 'At-Risk' if prediction == 1 else 'Not At-Risk'
        risk_prob = float(probability[1]) * 100
        safe_prob = float(probability[0]) * 100
        
        explanation = get_shap_explanation(features)
        
        shap_chart_data = []
        for exp in explanation[:8]:
            shap_chart_data.append({
                'feature': exp['feature'].replace('_', ' '),
                'value': exp['shap_value'],
                'color': '#e74c3c' if exp['shap_value'] > 0 else '#2ecc71'
            })
        
        fig_shap = go.Figure(go.Bar(
            x=[d['value'] for d in shap_chart_data],
            y=[d['feature'] for d in shap_chart_data],
            orientation='h',
            marker_color=[d['color'] for d in shap_chart_data]
        ))
        fig_shap.update_layout(
            title='SHAP Feature Impact on Prediction',
            xaxis_title='SHAP Value (Impact on Prediction)',
            template='none', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'),
            plot_bgcolor='rgba(0,0,0,0)', height=400,
            yaxis=dict(autorange='reversed')
        )
        shap_plot = fig_to_json(fig_shap)
        
        fig_risk = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob,
            number={'suffix':'%','font':{'size':48,'color':'#94a3b8'}},
            title={'text':risk_level,'font':{'size':22,'color':'#94a3b8'}},
            gauge={
                'axis':{'range':[0,100],'tickfont':{'size':14,'color':'#94a3b8'},'dtick':25},
                'bar':{'color':'#f43f5e' if prediction == 1 else '#34d399','thickness':0.8},
                'bgcolor':'rgba(0,0,0,0)',
                'steps':[
                    {'range':[0,30],'color':'rgba(52,211,153,.25)'},
                    {'range':[30,60],'color':'rgba(251,191,36,.25)'},
                    {'range':[60,100],'color':'rgba(244,63,94,.25)'}
                ],
                'threshold':{'line':{'color':'#fff','width':3},'thickness':0.8,'value':risk_prob}
            }
        ))
        fig_risk.update_layout(template='none',paper_bgcolor='rgba(0,0,0,0)',
                               font=dict(color='#94a3b8'),height=280,margin=dict(t=60,b=20,l=30,r=30))
        risk_gauge = fig_to_json(fig_risk)
        
        interventions = generate_interventions(explanation, student_data)
        
        return render_template('prediction_result.html',
                             student_data=student_data,
                             risk_level=risk_level,
                             risk_prob=risk_prob,
                             safe_prob=safe_prob,
                             explanation=explanation,
                             shap_plot=shap_plot,
                             risk_gauge=risk_gauge,
                             interventions=interventions)
    
    return render_template('predict.html')

@app.route('/student/<int:student_id>')
@login_required
def student_detail(student_id):
    if student_id >= len(df_raw):
        flash('Student not found.', 'danger')
        return redirect(url_for('students'))
    
    student = df_raw.iloc[student_id].to_dict()
    student['id'] = student_id
    
    student_input = {k: v for k, v in student.items() if k != 'Dropped_Out'}
    features = preprocess_student(student_input)
    prediction = rf_model.predict(features)[0]
    probability = rf_model.predict_proba(features)[0]
    
    risk_level = 'At-Risk' if prediction == 1 else 'Not At-Risk'
    risk_prob = float(probability[1]) * 100
    
    explanation = get_shap_explanation(features)
    
    shap_data = explanation[:8]
    fig_shap = go.Figure(go.Bar(
        x=[d['shap_value'] for d in shap_data],
        y=[d['feature'].replace('_', ' ') for d in shap_data],
        orientation='h',
        marker_color=['#e74c3c' if d['shap_value'] > 0 else '#2ecc71' for d in shap_data]
    ))
    fig_shap.update_layout(
        title='SHAP Explanation for This Student',
        template='none', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'),
        plot_bgcolor='rgba(0,0,0,0)', height=350,
        yaxis=dict(autorange='reversed')
    )
    shap_plot = fig_to_json(fig_shap)
    
    radar_features = ['Grade_1', 'Grade_2', 'Final_Grade', 'Study_Time', 'Health_Status']
    radar_vals = [student.get(f, 0) for f in radar_features]
    radar_max = [20, 20, 20, 4, 5]
    radar_pct = [v/m*100 for v, m in zip(radar_vals, radar_max)]
    
    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_pct + [radar_pct[0]],
        theta=[f.replace('_', ' ') for f in radar_features] + [radar_features[0].replace('_', ' ')],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.3)',
        line_color='#3498db'
    ))
    fig_radar.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(range=[0, 100])),
        template='none', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'),
        title='Student Profile Radar', height=350, showlegend=False
    )
    radar_plot = fig_to_json(fig_radar)
    
    interventions = generate_interventions(explanation, student)
    
    return render_template('student_detail.html',
                          student=student, risk_level=risk_level, risk_prob=risk_prob,
                          explanation=explanation, shap_plot=shap_plot,
                          radar_plot=radar_plot, interventions=interventions)

@app.route('/explanation')
@login_required
def explanation():
    fi = model_metrics['feature_importance'][:15]
    
    fig_global = px.bar(
        x=[f['shap_importance'] for f in fi],
        y=[f['feature'] for f in fi],
        orientation='h',
        title='Global Feature Importance (SHAP)',
        labels={'x': 'Mean |SHAP Value|', 'y': 'Feature'},
        color=[f['shap_importance'] for f in fi],
        color_continuous_scale='Viridis'
    )
    fig_global.update_layout(template='none', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'),
                             plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(autorange='reversed'),
                             height=500, showlegend=False)
    global_shap = fig_to_json(fig_global)
    
    academic = ['Grade_1', 'Grade_2', 'Final_Grade', 'Number_of_Failures', 'Study_Time']
    behavioral = ['Free_Time', 'Going_Out', 'Weekend_Alcohol_Consumption', 
                  'Weekday_Alcohol_Consumption', 'Health_Status']
    demographic = ['Age', 'Mother_Education', 'Father_Education', 'School']
    
    cat_importance = {}
    for f in fi:
        name = f['feature']
        val = f['shap_importance']
        if name in academic:
            cat_importance['Academic'] = cat_importance.get('Academic', 0) + val
        elif name in behavioral:
            cat_importance['Behavioral'] = cat_importance.get('Behavioral', 0) + val
        elif name in demographic:
            cat_importance['Demographic'] = cat_importance.get('Demographic', 0) + val
        else:
            cat_importance['Other'] = cat_importance.get('Other', 0) + val
    
    fig_cat = px.pie(
        values=list(cat_importance.values()),
        names=list(cat_importance.keys()),
        title='Feature Category Contribution to Predictions',
        color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    )
    fig_cat.update_layout(template='none', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
    category_chart = fig_to_json(fig_cat)
    
    return render_template('explanation.html', global_shap=global_shap,
                          category_chart=category_chart, metrics=model_metrics)

@app.route('/report')
@login_required
def report():
    total = len(df_raw)
    at_risk = int(df_raw['Dropped_Out'].sum())
    
    stats = {
        'total': total,
        'at_risk': at_risk,
        'not_at_risk': total - at_risk,
        'risk_pct': f"{(at_risk/total)*100:.1f}",
        'avg_grade': f"{df_raw['Final_Grade'].mean():.2f}",
        'avg_absences': f"{df_raw['Number_of_Absences'].mean():.2f}",
        'avg_failures': f"{df_raw['Number_of_Failures'].mean():.2f}"
    }
    
    cm = model_metrics['confusion_matrix']
    labels = ['Not At-Risk', 'At-Risk']
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        text=[[str(v) for v in row] for row in cm],
        texttemplate='%{text}', textfont=dict(size=22, color='#fff'),
        colorscale='Purples', showscale=False
    ))
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted', yaxis_title='Actual',
        template='none', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='DM Sans'),
        height=400, yaxis=dict(autorange='reversed')
    )
    cm_chart = fig_to_json(fig_cm)
    
    return render_template('report.html', stats=stats, metrics=model_metrics,
                          cm_chart=cm_chart, now=datetime.now().strftime('%d %B %Y, %I:%M %p'))

def generate_interventions(explanation, student_data):
    """Generate suggested interventions based on SHAP explanation."""
    interventions = []
    
    for exp in explanation[:5]:
        feat = exp['feature']
        direction = exp['direction']
        
        if feat == 'Final_Grade' and direction == 'increases':
            interventions.append({
                'factor': 'Low Academic Performance',
                'description': 'Student shows low final grade which strongly indicates dropout risk.',
                'action': 'Assign additional tutoring sessions and academic support programs. Monitor weekly progress.'
            })
        elif feat == 'Number_of_Absences' and direction == 'increases':
            interventions.append({
                'factor': 'High Absenteeism',
                'description': 'Frequent absences are contributing to dropout risk.',
                'action': 'Schedule parent-teacher meeting. Implement attendance monitoring with weekly check-ins.'
            })
        elif feat == 'Number_of_Failures' and direction == 'increases':
            interventions.append({
                'factor': 'Past Academic Failures',
                'description': 'History of course failures increases dropout risk.',
                'action': 'Provide remedial classes and peer tutoring. Consider course load adjustment.'
            })
        elif feat == 'Grade_1' and direction == 'increases':
            interventions.append({
                'factor': 'Low First Period Grade',
                'description': 'Poor performance in Grade 1 assessment detected.',
                'action': 'Early academic intervention. Assign study groups and monitor assignment completion.'
            })
        elif feat == 'Grade_2' and direction == 'increases':
            interventions.append({
                'factor': 'Low Second Period Grade',
                'description': 'Declining grades in the second period.',
                'action': 'Intensive tutoring and counselor check-in. Review study habits and time management.'
            })
        elif feat == 'Study_Time' and direction == 'increases':
            interventions.append({
                'factor': 'Insufficient Study Time',
                'description': 'Low weekly study hours contribute to risk.',
                'action': 'Provide structured study schedules. Create supervised study sessions after school.'
            })
        elif feat == 'Family_Relationship' and direction == 'increases':
            interventions.append({
                'factor': 'Family Relationship Issues',
                'description': 'Poor family relationships detected as a risk factor.',
                'action': 'Refer to school counselor. Consider family engagement programs.'
            })
        elif 'Alcohol' in feat and direction == 'increases':
            interventions.append({
                'factor': 'Alcohol Consumption',
                'description': 'Alcohol consumption patterns increase dropout risk.',
                'action': 'Refer to student wellness program. Provide awareness sessions and counseling.'
            })
        elif direction == 'increases':
            interventions.append({
                'factor': feat.replace('_', ' '),
                'description': f'This factor increases the student\'s dropout risk.',
                'action': 'Monitor this factor closely and consult with academic advisor for targeted intervention.'
            })
    
    if not interventions:
        interventions.append({
            'factor': 'General Monitoring',
            'description': 'No strong risk factors detected.',
            'action': 'Continue regular monitoring and maintain supportive environment.'
        })
    
    return interventions[:5]

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@school.edu.my',
                    full_name='System Administrator', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
