import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 FATIGUE DETECTION PIPELINE - CONSERVATIVE CLASSIFICATION\n")
print("="*70)

# ================== STEP 1: LOAD DATA ==================
print("\n📂 Loading dataset...")
# CORRECTED PATH - Update this to your actual file location
df = pd.read_csv('/content/drive/MyDrive/MASTER_EEG_FATIGUE_DATASET_IMPUTED.csv')
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ================== STEP 2: BINARY CLASSIFICATION (CONSERVATIVE) ==================
print("\n🔄 Converting to binary classification (CONSERVATIVE)...")
print(f"Original label distribution:\n{df['fatigue_label'].value_counts().sort_index()}\n")

# CONSERVATIVE Binary mapping: 
# 0, 1, 2 → 0 (Not Fatigued / Low Fatigue)
# 3, 4 → 1 (Moderate-to-High Fatigue)
df['fatigue_binary'] = df['fatigue_label'].apply(lambda x: 0 if x <= 2 else 1)

print(f"Binary label distribution (Conservative):")
print(f"  Class 0 (Labels 0,1,2 - Not Fatigued): {(df['fatigue_binary']==0).sum()}")
print(f"  Class 1 (Labels 3,4 - Fatigued):       {(df['fatigue_binary']==1).sum()}")
print(f"\nClass balance: {df['fatigue_binary'].value_counts(normalize=True).round(3)}\n")

# ================== STEP 3: FEATURE ENGINEERING ==================
print("🔧 Engineering fatigue-specific features...")

# Fatigue-specific EEG ratios (proven indicators)
df['theta_alpha_ratio'] = df['t_mean'] / (df['a_mean'] + 1e-6)
df['alpha_beta_ratio'] = df['a_mean'] / (df['b_mean'] + 1e-6)
df['theta_beta_ratio'] = df['t_mean'] / (df['b_mean'] + 1e-6)
df['delta_theta_ratio'] = df['d_mean'] / (df['t_mean'] + 1e-6)

# Variability features (fatigue increases EEG variability)
df['alpha_variability'] = df['a_std'] / (df['a_mean'] + 1e-6)
df['theta_variability'] = df['t_std'] / (df['t_mean'] + 1e-6)
df['beta_variability'] = df['b_std'] / (df['b_mean'] + 1e-6)

# Power totals
df['total_relative_power'] = df['a_mean'] + df['b_mean'] + df['d_mean'] + df['g_mean'] + df['t_mean']
df['total_absolute_power'] = df['Aa_mean'] + df['Ab_mean'] + df['Ad_mean'] + df['Ag_mean'] + df['At_mean']

# Temporal features (fatigue accumulates over time)
df['cumulative_sessions'] = df.groupby('user_id').cumcount()
df['round_session_interaction'] = df['round_id'] * df['session_id']

print(f"✅ Added 11 engineered features\n")

# ================== STEP 4: PREPARE FEATURES ==================
print("📊 Preparing features and labels...")

# Drop non-feature columns
drop_cols = ['fatigue_label', 'fatigue_binary', 'user_id', 'filepath', 'condition']
X = df.drop([col for col in drop_cols if col in df.columns], axis=1)
y = df['fatigue_binary']

print(f"Features: {X.shape[1]} columns")
print(f"Target: {len(y)} samples\n")

# ================== STEP 5: TRAIN-TEST SPLIT ==================
print("✂️  Splitting data (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"  Train Class 0: {(y_train==0).sum()} | Class 1: {(y_train==1).sum()}")
print(f"  Test  Class 0: {(y_test==0).sum()} | Class 1: {(y_test==1).sum()}\n")

# ================== STEP 6: FEATURE SCALING ==================
print("⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features standardized\n")

# ================== STEP 7: HANDLE CLASS IMBALANCE ==================
print("🔄 Applying SMOTE + Tomek Links for class balancing...")
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)

print(f"After resampling:")
print(f"  Class 0: {(y_train_resampled==0).sum()}")
print(f"  Class 1: {(y_train_resampled==1).sum()}")
print(f"  Balance: {pd.Series(y_train_resampled).value_counts(normalize=True).round(3).to_dict()}\n")

# ================== STEP 8: MODEL TRAINING (MULTIPLE ALGORITHMS) ==================
print("="*70)
print("🤖 TRAINING MULTIPLE ALGORITHMS FOR COMPARISON\n")

models = {}
results = {}

# --- 1. Random Forest (Recall-Optimized) ---
print("1️⃣  Random Forest (Recall-Optimized)...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 3},
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_resampled, y_train_resampled)
models['Random Forest'] = rf

# --- 2. XGBoost (Best for imbalanced data) ---
print("2️⃣  XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    scale_pos_weight=3,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
xgb.fit(X_train_resampled, y_train_resampled)
models['XGBoost'] = xgb

# --- 3. Gradient Boosting ---
print("3️⃣  Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train_resampled, y_train_resampled)
models['Gradient Boosting'] = gb

# --- 4. SVM (with class weights) ---
print("4️⃣  Support Vector Machine...")
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight={0: 1, 1: 3},
    probability=True,
    random_state=42
)
svm.fit(X_train_resampled, y_train_resampled)
models['SVM'] = svm

# --- 5. Voting Ensemble (Soft voting) ---
print("5️⃣  Voting Ensemble (RF + XGB + GB)...")
voting = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)],
    voting='soft',
    weights=[1, 1.5, 1]
)
voting.fit(X_train_resampled, y_train_resampled)
models['Voting Ensemble'] = voting

print("\n✅ All models trained!\n")

# ================== STEP 9: EVALUATION ==================
print("="*70)
print("📊 MODEL EVALUATION - RECALL FOCUS (CONSERVATIVE CLASSIFICATION)\n")

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"📈 {model_name}")
    print('='*70)
    
    y_pred = model.predict(X_test_scaled)
    
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    
    results[model_name] = {
        'Recall (Not Fatigued)': recall_0,
        'Recall (Fatigued)': recall_1,
        'F1 (Not Fatigued)': f1_0,
        'F1 (Fatigued)': f1_1
    }
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fatigued (0,1,2)', 'Fatigued (3,4)']))
    
    print("\n🎯 Key Metrics:")
    print(f"  • Recall (Class 0 - Not Fatigued): {recall_0:.3f}")
    print(f"  • Recall (Class 1 - Fatigued):     {recall_1:.3f} ⭐")
    print(f"  • F1 (Class 0): {f1_0:.3f}")
    print(f"  • F1 (Class 1): {f1_1:.3f} ⭐")
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"              Not Fatigued  Fatigued")
    print(f"Actual  Not    [{cm[0,0]:4d}         {cm[0,1]:4d}]")
    print(f"        Fatigued [{cm[1,0]:4d}         {cm[1,1]:4d}]")
    print(f"\n  • False Negatives (Missed Fatigue): {cm[1,0]} ❌")
    print(f"  • True Positives (Detected Fatigue): {cm[1,1]} ✅")

# ================== STEP 10: BEST MODEL SUMMARY ==================
print("\n\n" + "="*70)
print("🏆 BEST MODEL FOR FATIGUE RECALL\n")

best_model_name = max(results, key=lambda x: results[x]['Recall (Fatigued)'])
best_recall = results[best_model_name]['Recall (Fatigued)']

print(f"🥇 Winner: {best_model_name}")
print(f"🎯 Best Recall (Fatigued): {best_recall:.1%}")
print(f"📊 F1 Score (Fatigued): {results[best_model_name]['F1 (Fatigued)']:.3f}")

print("\n📊 Full Ranking by Recall (Fatigued):")
for i, (name, metrics) in enumerate(sorted(results.items(), key=lambda x: x[1]['Recall (Fatigued)'], reverse=True), 1):
    print(f"  {i}. {name}: {metrics['Recall (Fatigued)']:.3f}")

print("\n" + "="*70)
