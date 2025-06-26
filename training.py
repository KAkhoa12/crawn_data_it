import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, label_binarize
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings
import joblib 

warnings.filterwarnings('ignore')

CLASS_MAPPING = {
    'Not Suitable': 0,
    'Moderately Suitable': 1, 
    'Most Suitable': 2
}


def remove_outliers_iqr(df, columns, multiplier=1.5):
    """Remove outliers using IQR method with configurable multiplier"""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        before = df_clean.shape[0]
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        after = df_clean.shape[0]
        print(f"✅ Cột '{col}': đã loại {before - after} outlier ({((before-after)/before)*100:.1f}%)")
    return df_clean

def analyze_data_distribution(df, target_col='suitability_label'):
    """Analyze data distribution and potential issues"""
    print("\n📊 PHÂN TÍCH DỮ LIỆU:")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Phân bố target:")
    target_counts = df[target_col].value_counts().sort_index()
    for val, count in target_counts.items():
        print(f"  Class {val} ({CLASS_MAPPING.get(val, 'Unknown')}): {count} ({count/len(df)*100:.1f}%)")
    
    # Check for data leakage indicators
    feature_cols = ['primary_skills_sim', 'secondary_skills_sim', 'adjectives_sim', 'adj_weight_log', 'total_similarity_v3']
    
    print("\n🔍 KIỂM TRA DATA LEAKAGE:")
    for col in feature_cols:
        # Check perfect correlation with target
        corr = df[col].corr(df[target_col])
        print(f"Correlation {col} vs target: {corr:.4f}")
        
        # Check for perfect separation
        for class_val in df[target_col].unique():
            class_data = df[df[target_col] == class_val][col]
            print(f"  Class {class_val}: mean={class_data.mean():.4f}, std={class_data.std():.4f}")

class ImprovedTrainer:
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
        self.models = {}
        self.scaler = None

    def load_data(self, path):
        """Load and preprocess data"""
        self.data = pd.read_csv(path)
        required_features = [
            'primary_skills_sim',
            'secondary_skills_sim',
            'adjectives_sim',
        ]
        self.data = self.data.dropna()
        self.data['suitability_label'] = self.data['suitability_label'].map(CLASS_MAPPING)
        print("📈 DỮ LIỆU GỐC:")
        analyze_data_distribution(self.data)

        # Remove outliers with more conservative approach
        self.data = remove_outliers_iqr(self.data, required_features, multiplier=2.0)
        
        print("\n📈 SAU KHI XỬ LÝ OUTLIERS:")
        analyze_data_distribution(self.data)
        

        
        self.X = self.data[required_features].copy()
        self.y = self.data['suitability_label'].astype(int)
        
        # Handle missing values
        self.X = self.X.fillna(self.X.median())
        
        # Scale features
        self.scaler = StandardScaler()
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X), 
            columns=required_features,
            index=self.X.index
        )

    def split_data(self, test_size=0.2, val_size=0.2):
        """Split data with proper stratification"""
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            stratify=self.y, 
            random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=42
        )
        
        print(f"\n📊 PHÂN CHIA DỮ LIỆU:")
        print(f"Train: {len(self.X_train)} samples")
        print(f"Validation: {len(self.X_val)} samples") 
        print(f"Test: {len(self.X_test)} samples")

    def train_conservative_models(self):
        """Train models with strong regularization"""
        results = {}
        
        # 1. Logistic Regression with strong regularization
        lr = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=0.01,  # Strong regularization
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        # 2. Decision Tree with very conservative parameters
        dt = DecisionTreeClassifier(
            max_depth=3,  # Very shallow
            min_samples_split=20,  # Higher threshold
            min_samples_leaf=10,   # Higher threshold
            max_features='sqrt',   # Feature subsampling
            class_weight='balanced',
            random_state=42
        )
        
        # 3. Random Forest (better than AdaBoost for avoiding overfitting)
        ada = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=0.5,
            random_state=42
        )
        
        # 4. XGBoost with very conservative settings
        xgb_clf = xgb.XGBClassifier(
            max_depth=1,           # Very shallow
            learning_rate=0.05,    # Slower learning
            n_estimators=50,       # Fewer trees
            subsample=0.7,         # More subsampling
            colsample_bytree=0.7,  # More feature subsampling
            reg_alpha=1.0,         # L1 regularization
            reg_lambda=2.0,        # L2 regularization
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
        
        models_config = {
            'LogisticRegression': lr,
            'DecisionTree': dt,
            'AdaBoost': ada,
            'XGBoost': xgb_clf
        }
        
        for name, model in models_config.items():
            print(f"\n🔧 Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Evaluate on validation set first
            val_pred = model.predict(self.X_val)
            val_acc = accuracy_score(self.y_val, val_pred)
            
            # Evaluate on test set
            test_pred = model.predict(self.X_test)
            test_acc = accuracy_score(self.y_test, test_pred)
            
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Overfitting Gap: {val_acc - test_acc:.4f}")
            
            # Cross-validation for more robust evaluation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            print(f"Classification Report for {name}:")
            CLASS_LABELS = ['Not Suitable', 'Moderately Suitable', 'Most Suitable']
            print(classification_report(self.y_test, test_pred, target_names=CLASS_LABELS))
            
            self.models[name] = model
            results[name] = {
                'val_acc': val_acc,
                'test_acc': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            if name == 'XGBoost':
                joblib.dump(model, 'xgboost_model.pkl')
                print("💾 XGBoost model saved to 'xgboost_model.pkl'")
        return results

    def plot_improved_roc_curves(self):
        """Plot ROC curves with proper multiclass handling"""
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        colors = cycle(['aqua', 'orange', 'cornflowerblue', 'green'])
        
        # Plot for each class
        for class_idx in range(n_classes):
            ax = axes[class_idx]
            
            for model_name, color in zip(self.models.keys(), colors):
                model = self.models[model_name]
                
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(self.X_test)[:, class_idx]
                else:
                    # For models without predict_proba
                    pred = model.predict(self.X_test)
                    y_score = (pred == class_idx).astype(float)
                
                fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, lw=2, 
                       label=f'{model_name} (AUC = {roc_auc:.3f})', 
                       color=color)
            REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - Class {class_idx} ({REVERSE_CLASS_MAPPING[class_idx]})')

            ax.legend(loc="lower right")
            ax.grid(True)
        
        # Confusion matrices
        ax = axes[3]
        # Pick best model based on CV score
        best_model_name = max(self.models.keys(), 
                             key=lambda x: cross_val_score(self.models[x], self.X_train, self.y_train, cv=3).mean())
        best_model = self.models[best_model_name]
        
        y_pred = best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=CLASS_MAPPING.values(),
                   yticklabels=CLASS_MAPPING.values())
        ax.set_title(f'Confusion Matrix - {best_model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig("improved_model_evaluation.png", dpi=300, bbox_inches='tight')
        print(f"\n✅ Model evaluation saved as 'improved_model_evaluation.png'")

def main():
    trainer = ImprovedTrainer()
    trainer.load_data('progress/csv/jd_cr_similarity.csv')
    trainer.split_data()
    results = trainer.train_conservative_models()
    
    print("\n📊 FINAL RESULTS SUMMARY:")
    print("=" * 80)
    for model, metrics in results.items():
        print(f"{model:20} | Val: {metrics['val_acc']:.4f} | Test: {metrics['test_acc']:.4f} | "
              f"CV: {metrics['cv_mean']:.4f}±{metrics['cv_std']:.3f} | "
              f"Gap: {abs(metrics['val_acc'] - metrics['test_acc']):.4f}")
    
    trainer.plot_improved_roc_curves()
    
    # Additional recommendations
    print("\n💡 KHUYẾN NGHỊ:")
    print("1. Nếu vẫn thấy overfitting, hãy thu thập thêm dữ liệu")
    print("2. Kiểm tra xem có data leakage không (features quá tương quan với target)")
    print("3. Thử feature selection để giảm số lượng features")
    print("4. Xem xét sử dụng validation curve để tune hyperparameters")

if __name__ == "__main__":
    main()