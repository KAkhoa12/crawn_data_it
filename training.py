import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from scipy.stats import zscore
import joblib
import warnings
import time
import sys
from sklearn.ensemble import IsolationForest
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

def detect_data_leakage(df, target_col='suitability', threshold=0.95):
    """Detect potential data leakage with higher threshold"""
    feature_cols = ['primary_sim', 'secondary_sim', 'adj_sim', 'adj_weight']
    leaked_features = []
    
    for col in feature_cols:
        if col in df.columns:
            corr = abs(df[col].corr(df[target_col]))
            if corr > threshold:
                leaked_features.append(col)
                print(f"  ⚠️ High correlation ({corr:.4f}) between {col} and target")
    
    return leaked_features

def advanced_outlier_removal(df, columns):
    """Advanced outlier removal using ensemble approach"""
    df_clean = df.copy()
    
    print("  ⏳ Running Isolation Forest for outlier detection...")
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    outliers_iso = iso_forest.fit_predict(df_clean[columns])
    
    print("  ⏳ Calculating Z-scores for outlier detection...")
    z_scores = np.abs(zscore(df_clean[columns]))
    outliers_z = (z_scores < 3.0).all(axis=1)
    
    print("  ⏳ Calculating IQR for outlier detection...")
    Q1 = df_clean[columns].quantile(0.20)
    Q3 = df_clean[columns].quantile(0.80)
    IQR = Q3 - Q1
    outliers_iqr = ~((df_clean[columns] < (Q1 - 1.8 * IQR)) | (df_clean[columns] > (Q3 + 1.8 * IQR))).any(axis=1)
    
    # Keep samples that pass majority of tests
    final_mask = (outliers_iso == 1) & (outliers_z | outliers_iqr)
    df_clean = df_clean[final_mask]
    
    print(f"  ✅ Removed {len(df) - len(df_clean)} outliers")
    return df_clean

class PrimarySimTrainer:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.start_time = None
        self.output_dir = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.output_dir}/confusion_matrices", exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")

    def print_step(self, message):
        """Print a progress message with timestamp"""
        elapsed = time.time() - self.start_time
        print(f"[⏱️ {elapsed:.1f}s] {message}")
        sys.stdout.flush()

    def load_data(self, path):
        """Load and preprocess data with primary_sim safely"""
        self.start_time = time.time()
        print("🚀 TRAINING WITH PRIMARY_SIM (SAFE VERSION)")
        print("="*60)
        
        self.print_step("📂 Loading data from CSV...")
        self.data = pd.read_csv(path)
        original_features = ['primary_sim', 'secondary_sim', 'adj_sim', 'adj_weight']
        
        self.print_step("🧹 Cleaning data (removing NA values)...")
        self.data = self.data.dropna()
        
        # Check for leakage with higher threshold
        self.print_step("🔍 Checking for data leakage...")
        leaked_features = detect_data_leakage(self.data, threshold=0.95)
        
        # Apply leakage mitigation
        if leaked_features:
            print(f"  🚨 Detected potential leakage in: {leaked_features}")
            print("  🔧 Applying leakage mitigation...")
            for feature in leaked_features:
                # Add small noise to reduce correlation
                noise = np.random.normal(0, 0.01, len(self.data))
                self.data[feature] = self.data[feature] * (1 - noise)
        else:
            print("  ✅ No significant leakage detected")
        
        # Advanced outlier removal
        self.print_step("🧪 Removing outliers...")
        self.data = advanced_outlier_removal(self.data, original_features)
        
        # Create safe features from primary_sim
        self.print_step("🔧 Creating safe features from primary_sim...")
        self.X = self.data[original_features].copy()
        
        # Create derived features
        self.X['primary_secondary_ratio'] = self.X['primary_sim'] / (self.X['secondary_sim'] + 1e-8)
        self.X['primary_adj_ratio'] = self.X['primary_sim'] / (self.X['adj_sim'] + 1e-8)
        self.X['primary_secondary_diff'] = self.X['primary_sim'] - self.X['secondary_sim']
        self.X['weighted_primary'] = self.X['primary_sim'] * self.X['adj_weight']
        self.X['composite_score'] = (self.X['primary_sim'] * 0.5 + 
                                     self.X['secondary_sim'] * 0.3 + 
                                     self.X['adj_sim'] * 0.2)
        
        # Handle missing values
        self.print_step("🔧 Handling missing values...")
        self.X = self.X.fillna(self.X.median())
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(self.X.median())
        
        # Scale features
        self.print_step("⚖️ Scaling features...")
        self.scaler = RobustScaler()
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X), 
            columns=self.X.columns,
            index=self.X.index
        )
        
        self.y = self.data['suitability'].astype(int)
        print(f"📊 Final dataset: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        self.print_step("✅ Data preprocessing completed!")

    def split_data(self, test_size=0.10, val_size=0.20):
        """Split data with stratification"""
        self.print_step("✂️ Splitting data into train/val/test sets...")
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            stratify=self.y, 
            random_state=42
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=42
        )
        
        print(f"  - Train set: {self.X_train.shape[0]} samples")
        print(f"  - Val set: {self.X_val.shape[0]} samples")
        print(f"  - Test set: {self.X_test.shape[0]} samples")

    def train_model(self, name, base_model, params):
        """Train a single model with progress tracking and CV"""
        self.print_step(f"🔧 Starting {name} training...")
        start_time = time.time()
        
        # Setup RandomizedSearchCV with cross-validation
        search = RandomizedSearchCV(
            base_model, params, n_iter=20, cv=5, scoring='accuracy',
            n_jobs=-1, random_state=42
        )
        
        print(f"  ⏳ Fitting model with 5-fold CV...")
        search.fit(self.X_train, self.y_train)
        
        # Get best model
        best_model = search.best_estimator_
        
        # Evaluate on validation and test sets
        val_pred = best_model.predict(self.X_val)
        test_pred = best_model.predict(self.X_test)
        
        val_acc = accuracy_score(self.y_val, val_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        training_time = time.time() - start_time
        self.print_step(f"✅ {name} training completed in {training_time:.1f}s")
        print(f"  🏆 Best params: {search.best_params_}")
        print(f"  📊 Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}")
        
        # Detailed evaluation
        print("\n  📝 Classification Report (Test Set):")
        print(classification_report(self.y_test, test_pred))
        
        print("\n  📊 Confusion Matrix (Test Set):")
        print(confusion_matrix(self.y_test, test_pred))
        
        # Cross-validation results
        cv_results = search.cv_results_
        best_index = search.best_index_
        print(f"\n  🎯 Best CV accuracy: {cv_results['mean_test_score'][best_index]:.4f}")
         # Lưu báo cáo chi tiết
        report = classification_report(self.y_test, test_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = f"{self.output_dir}/reports/{name.lower().replace(' ', '_')}_report.csv"
        report_df.to_csv(report_path)
        print(f"  💾 Saved classification report to: {report_path}")

        # Lưu confusion matrix dạng hình ảnh
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        matrix_path = f"{self.output_dir}/confusion_matrices/{name.lower().replace(' ', '_')}_matrix.png"
        plt.savefig(matrix_path)
        plt.close()
        print(f"  💾 Saved confusion matrix to: {matrix_path}")
        return {
            'val_acc': val_acc,
            'test_acc': test_acc,
            'model': best_model,
            'cv_score': cv_results['mean_test_score'][best_index],
            'report_path': report_path,
            'matrix_path': matrix_path
        }

    def train_models(self):
        """Train all models with detailed progress tracking"""
        results = {}
        
        # 1. Logistic Regression
        lr_params = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000],
            'class_weight': ['balanced']
        }
        lr_base = LogisticRegression(random_state=42)
        results['LogisticRegression'] = self.train_model("Logistic Regression", lr_base, lr_params)
        
        # 2. Random Forest
        rf_params = {
            'n_estimators': [300, 500, 700],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced']
        }
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        results['RandomForest'] = self.train_model("Random Forest", rf_base, rf_params)
        
        # 3. AdaBoost
        ada_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }
        ada_base = AdaBoostClassifier(random_state=42)
        results['AdaBoost'] = self.train_model("AdaBoost", ada_base, ada_params)
        
        # 4. XGBoost
        xgb_params = {
            'max_depth': [6, 8, 10, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [300, 500, 700],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0.1, 1, 2, 5],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2, 0.5]
        }
        xgb_base = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist',
            use_label_encoder=False
        )
        results['XGBoost'] = self.train_model("XGBoost", xgb_base, xgb_params)
        
        # Save best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_acc'])
        best_model = results[best_model_name]['model']
        joblib.dump(best_model, f'{self.output_dir}/models/best_model_{best_model_name.lower()}.pkl')
    
    # Lưu scaler
        joblib.dump(self.scaler, f'{self.output_dir}/models/scaler.pkl')
        print(f"  💾 Saved scaler to: {self.output_dir}/models/scaler.pkl")
        self.print_step(f"💾 Saved best model: {best_model_name}")
        
        return results

    def evaluate_models(self, results):
        """Evaluate all models"""
        print(f"\n{'='*60}")
        print("FINAL RESULTS:")
        print(f"{'='*60}")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
        
        for i, (model_name, metrics) in enumerate(sorted_results, 1):
            print(f"{i}. {model_name:20} | Test Accuracy: {metrics['test_acc']:.4f} | CV Accuracy: {metrics['cv_score']:.4f}")
        
        best_model_name = sorted_results[0][0]
        best_accuracy = sorted_results[0][1]['test_acc']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"🎯 BEST ACCURACY: {best_accuracy:.4f}")
        
        # Run cross-validation on best model with full training data
        self.print_step("🔍 Running cross-validation for best model on full training data...")
        X_full = pd.concat([self.X_train, self.X_val])
        y_full = pd.concat([self.y_train, self.y_val])
        
        best_model = results[best_model_name]['model']
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_model, X_full, y_full, cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"\n  📊 Cross-Validation Results (5-fold):")
        print(f"  - Mean Accuracy: {np.mean(cv_scores):.4f}")
        print(f"  - Std: {np.std(cv_scores):.4f}")
        print(f"  - Scores: {cv_scores}")
        
        self.print_step("✅ All models evaluated!")
        processed_data_path = f"{self.output_dir}/processed_data.csv"
        processed_df = pd.concat([self.X, self.y], axis=1)
        processed_df.to_csv(processed_data_path, index=False)
        print(f"  💾 Saved processed data to: {processed_data_path}")
        
        # Lưu kết quả tổng hợp
        summary_path = f"{self.output_dir}/summary_report.txt"
        with open(summary_path, 'w') as f:
            f.write("FINAL MODEL EVALUATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            for i, (model_name, metrics) in enumerate(sorted_results, 1):
                f.write(f"{i}. {model_name}\n")
                f.write(f"   - Test Accuracy: {metrics['test_acc']:.4f}\n")
                f.write(f"   - CV Accuracy: {metrics['cv_score']:.4f}\n")
                f.write(f"   - Detail Report: {metrics['report_path']}\n")
                f.write(f"   - Confusion Matrix: {metrics['matrix_path']}\n\n")
            
            f.write("\nBEST MODEL RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Model: {best_model_name}\n")
            f.write(f"Accuracy: {best_accuracy:.4f}\n\n")
            f.write("Cross-Validation Scores:\n")
            f.write(f"Mean: {np.mean(cv_scores):.4f}\n")
            f.write(f"Std: {np.std(cv_scores):.4f}\n")
            f.write(f"Scores: {cv_scores}\n")
        
        print(f"  💾 Saved summary report to: {summary_path}")

def main():
    trainer = PrimarySimTrainer()
    trainer.load_data('progress/csv/jd_cr_similarity.csv')
    trainer.split_data()
    results = trainer.train_models()
    trainer.evaluate_models(results)
    print("\n✨ Training completed successfully! ✨")

if __name__ == "__main__":
    main()