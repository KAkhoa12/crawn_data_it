import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# Class mapping constants
CLASS_MAPPING = {
    0: 'Not Suitable',
    1: 'Moderately Suitable',
    2: 'Most Suitable'
}

class ImprovedFourModelsTrainer:
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
        self.results = []
        self.feature_selector = None
        self.scaler = None

    def load_best_data(self):
        """Load Jaccard similarity dataset"""
        dataset_path = 'progress/csv/jd_cr_similarity.csv'

        try:
            self.data = pd.read_csv(dataset_path)
            print(f"✅ Loaded {len(self.data)} pairs from {dataset_path}")
            print("🎯 Using Jaccard similarity data from JD-CR matching")

            # Check if suitability data is already numeric or needs conversion
            if 'suitability_label' in self.data.columns:
                # Data already has numeric labels
                print("✅ Found numeric suitability_label column")
                self.data['suitability_label'] = self.data['suitability_label'].astype(int)
            elif 'suitability' in self.data.columns:
                # Check if suitability column contains numbers or text
                sample_value = self.data['suitability'].iloc[0]
                if isinstance(sample_value, (int, float)) or str(sample_value).isdigit():
                    # Already numeric
                    print("✅ Suitability column contains numeric values")
                    self.data['suitability_label'] = self.data['suitability'].astype(int)
                else:
                    # Convert text to numeric
                    print("🔄 Converting text suitability to numeric")
                    suitability_mapping = {
                        'Not Suitable': 0,
                        'Moderately Suitable': 1,
                        'Most Suitable': 2
                    }
                    self.data['suitability_label'] = self.data['suitability'].map(suitability_mapping)
                    self.data['suitability_label'] = self.data['suitability_label'].fillna(0).astype(int)
            else:
                print("❌ Neither 'suitability' nor 'suitability_label' column found in dataset")
                return False

            # Show label distribution
            label_counts = self.data['suitability_label'].value_counts().sort_index()
            total = len(self.data)
            print(f"\nLabel Distribution:")
            for label in [0, 1, 2]:
                count = label_counts.get(label, 0)
                percentage = (count / total) * 100
                print(f"  Class {label} ({CLASS_MAPPING[label]}): {count} samples ({percentage:.1f}%)")

            # Check for severe class imbalance
            min_class_pct = min([label_counts.get(i, 0) / total * 100 for i in [0, 1, 2]])
            if min_class_pct < 5.0:
                print(f"⚠️  CLASS IMBALANCE DETECTED!")
                print(f"   Smallest class: {min_class_pct:.2f}% - Will apply SMOTE for balancing")
                self.apply_smote = True
            else:
                self.apply_smote = False

            return True

        except FileNotFoundError:
            print(f"❌ Dataset not found: {dataset_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False

    def prepare_features_with_selection(self):
        """Prepare features with feature selection to reduce overfitting"""
        # Exclude problematic columns
        exclude_cols = [
            'jd_id', 'cr_category', 'suitability_label', 'suitability',
            'total_similarity', 'adj_weight'  # Potential data leakage
        ]

        # Get numeric columns
        feature_cols = []
        for col in self.data.columns:
            if col not in exclude_cols and self.data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)

        self.X = self.data[feature_cols].copy()
        self.y = self.data['suitability_label']

        # Handle missing values
        self.X = self.X.fillna(self.X.median())

        # Remove highly correlated features to reduce overfitting
        print("🔍 Removing highly correlated features...")
        corr_matrix = self.X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        if to_drop:
            print(f"   Dropping {len(to_drop)} highly correlated features: {to_drop}")
            self.X = self.X.drop(columns=to_drop)

        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Feature selection using SelectKBest
        print("🎯 Applying feature selection...")
        k_features = min(15, len(self.X.columns))  # Limit features to reduce overfitting
        self.feature_selector = SelectKBest(score_func=f_classif, k=k_features)
        X_selected = self.feature_selector.fit_transform(X_scaled, self.y)
        
        # Get selected feature names
        selected_features = self.X.columns[self.feature_selector.get_support()].tolist()
        print(f"   Selected {len(selected_features)} features: {selected_features}")
        
        # Update X with selected features
        self.X = pd.DataFrame(X_selected, columns=selected_features, index=self.X.index)

        print(f"Final feature matrix shape: {self.X.shape}")
        return self.X, self.y

    def apply_data_augmentation(self):
        """Apply SMOTE for class imbalance if needed"""
        if self.apply_smote:
            try:
                from imblearn.over_sampling import SMOTE
                print("🔄 Applying SMOTE for class balancing...")
                
                smote = SMOTE(random_state=42, k_neighbors=min(5, len(self.y_train[self.y_train == self.y_train.value_counts().idxmin()]) - 1))
                X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
                
                print(f"   Before SMOTE: {len(self.X_train)} samples")
                print(f"   After SMOTE: {len(X_train_balanced)} samples")
                
                self.X_train = pd.DataFrame(X_train_balanced, columns=self.X_train.columns)
                self.y_train = pd.Series(y_train_balanced)
                
            except ImportError:
                print("⚠️  SMOTE not available. Install with: pip install imbalanced-learn")
                print("   Continuing without SMOTE...")

    def split_data(self, test_size=0.15, val_size=0.15, random_state=42):
        """Split data with stratification"""
        X, y = self.prepare_features_with_selection()
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData Split:")
        print(f"  Training: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(self.X_val)} samples ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")

        # Apply data augmentation if needed
        self.apply_data_augmentation()

    def train_regularized_regression(self):
        """Train regularized regression models"""
        print(f"\n1. Training Regularized Regression (Ridge, Lasso, ElasticNet)...")

        # Try different regularization techniques
        models_to_try = {
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
        }

        best_model = None
        best_score = -np.inf
        best_name = None

        for name, model in models_to_try.items():
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error')
            cv_mean = -cv_scores.mean()
            
            if cv_mean < best_score or best_score == -np.inf:
                best_score = cv_mean
                best_model = model
                best_name = name

        # Train best model
        best_model.fit(self.X_train, self.y_train)

        # Predictions
        train_pred = best_model.predict(self.X_train)
        val_pred = best_model.predict(self.X_val)
        test_pred = best_model.predict(self.X_test)
        
        # Convert to classification
        train_pred_class = np.round(np.clip(train_pred, 0, 2)).astype(int)
        val_pred_class = np.round(np.clip(val_pred, 0, 2)).astype(int)
        test_pred_class = np.round(np.clip(test_pred, 0, 2)).astype(int)
        
        # Calculate accuracies
        train_acc = accuracy_score(self.y_train, train_pred_class)
        val_acc = accuracy_score(self.y_val, val_pred_class)
        test_acc = accuracy_score(self.y_test, test_pred_class)
        
        print(f"   Best Regularized Regression: {best_name}")
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Validation Accuracy: {val_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Overfitting Gap: {abs(train_acc - val_acc):.4f}")
        
        self.models['Regularized Regression'] = best_model
        self.results.append({
            'Model': f'Regularized Regression ({best_name})',
            'Train_Accuracy': train_acc,
            'Validation_Accuracy': val_acc,
            'Test_Accuracy': test_acc,
            'CV_Score': best_score,
            'CV_Std': cv_scores.std(),
            'Overfitting': abs(train_acc - val_acc)
        })
        
        return test_pred_class

    def train_improved_decision_tree(self):
        """Train improved decision tree with better regularization"""
        print(f"\n2. Training Improved Decision Tree...")
        
        # More aggressive regularization
        model = DecisionTreeClassifier(
            max_depth=4,  # Reduced further
            min_samples_split=max(50, len(self.X_train) // 100),  # Dynamic based on data size
            min_samples_leaf=max(20, len(self.X_train) // 200),   # Dynamic based on data size
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            ccp_alpha=0.01  # Cost complexity pruning
        )
        
        # Use validation set for early stopping simulation
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        test_pred = model.predict(self.X_test)
        
        # Calculate accuracies
        train_acc = accuracy_score(self.y_train, train_pred)
        val_acc = accuracy_score(self.y_val, val_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Validation Accuracy: {val_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Cross-Val Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"   Overfitting Gap: {abs(train_acc - val_acc):.4f}")
        
        self.models['Improved Decision Tree'] = model
        self.results.append({
            'Model': 'Improved Decision Tree',
            'Train_Accuracy': train_acc,
            'Validation_Accuracy': val_acc,
            'Test_Accuracy': test_acc,
            'CV_Score': cv_mean,
            'CV_Std': cv_std,
            'Overfitting': abs(train_acc - val_acc)
        })
        
        return test_pred

    def train_improved_adaboost(self):
        """Train improved AdaBoost with regularization"""
        print(f"\n3. Training Improved AdaBoost...")
        
        # Use a more regularized base estimator
        base_estimator = DecisionTreeClassifier(
            max_depth=2,  # Very shallow trees
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced'
        )
        
        model = AdaBoostClassifier(
            estimator=base_estimator,  # Changed from base_estimator to estimator
            n_estimators=30,  # Reduced to prevent overfitting
            learning_rate=0.5,  # Reduced learning rate
            algorithm='SAMME',
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        test_pred = model.predict(self.X_test)
        
        # Calculate accuracies
        train_acc = accuracy_score(self.y_train, train_pred)
        val_acc = accuracy_score(self.y_val, val_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Validation Accuracy: {val_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Cross-Val Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"   Overfitting Gap: {abs(train_acc - val_acc):.4f}")
        
        self.models['Improved AdaBoost'] = model
        self.results.append({
            'Model': 'Improved AdaBoost',
            'Train_Accuracy': train_acc,
            'Validation_Accuracy': val_acc,
            'Test_Accuracy': test_acc,
            'CV_Score': cv_mean,
            'CV_Std': cv_std,
            'Overfitting': abs(train_acc - val_acc)
        })
        
        return test_pred

    def train_improved_xgboost(self):
        """Train XGBoost with early stopping and regularization"""
        print(f"\n4. Training Improved XGBoost with Early Stopping...")
        
        model = xgb.XGBClassifier(
            n_estimators=1000,  # Large number for early stopping
            max_depth=4,        # Reduced depth
            learning_rate=0.05, # Reduced learning rate
            subsample=0.8,      # Row sampling
            colsample_bytree=0.8,  # Column sampling
            colsample_bylevel=0.8,  # Column sampling by level
            reg_alpha=0.1,      # L1 regularization
            reg_lambda=1.0,     # L2 regularization
            min_child_weight=3, # Minimum sum of instance weight
            gamma=0.1,          # Minimum loss reduction
            random_state=42,
            eval_metric='mlogloss',
            early_stopping_rounds=50,  # Stop if no improvement
            verbosity=0
        )
        
        # Fit with early stopping
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        
        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        test_pred = model.predict(self.X_test)
        
        # Calculate accuracies
        train_acc = accuracy_score(self.y_train, train_pred)
        val_acc = accuracy_score(self.y_val, val_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        # Cross-validation (create a model without early stopping for CV)
        cv_model = xgb.XGBClassifier(
            n_estimators=100,  # Fixed number for CV
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        )
        cv_scores = cross_val_score(cv_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Validation Accuracy: {val_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Cross-Val Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"   Overfitting Gap: {abs(train_acc - val_acc):.4f}")
        print(f"   Best Iteration: {model.best_iteration}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"   Top 5 Important Features:")
            for _, row in feature_importance.head(5).iterrows():
                print(f"     {row['feature']}: {row['importance']:.4f}")
        
        self.models['Improved XGBoost'] = model
        self.results.append({
            'Model': 'Improved XGBoost',
            'Train_Accuracy': train_acc,
            'Validation_Accuracy': val_acc,
            'Test_Accuracy': test_acc,
            'CV_Score': cv_mean,
            'CV_Std': cv_std,
            'Overfitting': abs(train_acc - val_acc)
        })

        return test_pred

    def plot_roc_curves(self):
        """Plot and save ROC curves for all 4 models"""
        print(f"\n📊 Plotting ROC Curves for all models...")

        # Set up the plot
        plt.figure(figsize=(12, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])

        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]

        # Store all ROC data for comparison
        all_fpr = dict()
        all_tpr = dict()
        all_roc_auc = dict()

        for model_name, color in zip(self.models.keys(), colors):
            model = self.models[model_name]

            # Get predictions
            if 'Regression' in model_name:
                # For regression models, we need to convert to probabilities
                pred_scores = model.predict(self.X_test)
                # Convert regression output to pseudo-probabilities
                y_score = np.zeros((len(pred_scores), n_classes))
                for i, score in enumerate(pred_scores):
                    # Clip and round to get class
                    pred_class = int(np.clip(np.round(score), 0, 2))
                    y_score[i, pred_class] = 1.0
            else:
                # For classifiers, get probability predictions
                y_score = model.predict_proba(self.X_test)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Plot micro-average ROC curve
            plt.plot(fpr["micro"], tpr["micro"],
                    color=color, linestyle='-', linewidth=2,
                    label=f'{model_name} (AUC = {roc_auc["micro"]:.3f})')

            # Store for comparison
            all_fpr[model_name] = fpr["micro"]
            all_tpr[model_name] = tpr["micro"]
            all_roc_auc[model_name] = roc_auc["micro"]

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')

        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison - All 4 Models\n(Micro-average for Multi-class)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.tight_layout()
        roc_filename = 'roc_curves_comparison.png'
        plt.savefig(roc_filename, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved as '{roc_filename}'")

        # Show plot
        plt.show()

        # Print AUC scores
        print(f"\n📈 ROC AUC Scores (Micro-average):")
        sorted_auc = sorted(all_roc_auc.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, auc_score) in enumerate(sorted_auc, 1):
            print(f"  {i}. {model_name}: {auc_score:.4f}")

        return all_roc_auc

    def plot_individual_roc_curves(self):
        """Plot ROC curves for each class separately"""
        print(f"\n📊 Plotting individual ROC curves for each class...")

        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        class_names = [CLASS_MAPPING[i] for i in range(n_classes)]

        # Create subplots for each class
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])

        for class_idx in range(n_classes):
            ax = axes[class_idx]
            colors_iter = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])

            for model_name, color in zip(self.models.keys(), colors_iter):
                model = self.models[model_name]

                # Get predictions
                if 'Regression' in model_name:
                    # For regression models
                    pred_scores = model.predict(self.X_test)
                    y_score = np.zeros((len(pred_scores), n_classes))
                    for i, score in enumerate(pred_scores):
                        pred_class = int(np.clip(np.round(score), 0, 2))
                        y_score[i, pred_class] = 1.0
                else:
                    # For classifiers
                    y_score = model.predict_proba(self.X_test)

                # Compute ROC curve for this class
                fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score[:, class_idx])
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                ax.plot(fpr, tpr, color=color, linewidth=2,
                       label=f'{model_name} (AUC = {roc_auc:.3f})')

            # Plot diagonal line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)

            # Customize subplot
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(f'ROC Curve - {class_names[class_idx]}', fontsize=12, fontweight='bold')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Save plot
        plt.tight_layout()
        individual_roc_filename = 'individual_roc_curves.png'
        plt.savefig(individual_roc_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Individual ROC curves saved as '{individual_roc_filename}'")

        # Show plot
        plt.show()

    def compare_models_and_save(self):
        """Compare models and save results"""
        print(f"\n" + "="*80)
        print("IMPROVED MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('Overfitting', ascending=True)  # Sort by least overfitting
        
        # Display results
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Find best model (balance between performance and overfitting)
        # Model with good test accuracy and low overfitting
        results_df['Combined_Score'] = results_df['Test_Accuracy'] - (results_df['Overfitting'] * 2)
        best_idx = results_df['Combined_Score'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        best_test_acc = results_df.loc[best_idx, 'Test_Accuracy']
        best_overfitting = results_df.loc[best_idx, 'Overfitting']
        
        print(f"\nBest Model (Balance of Performance & Generalization): {best_model_name}")
        print(f"Test Accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
        print(f"Overfitting Gap: {best_overfitting:.4f} ({best_overfitting*100:.2f}%)")
        
        # Overfitting analysis
        print(f"\n🎯 OVERFITTING ANALYSIS:")
        for _, row in results_df.iterrows():
            model_name = row['Model']
            overfitting = row['Overfitting']
            status = "✅ Good" if overfitting < 0.05 else "⚠️ Moderate" if overfitting < 0.1 else "❌ High"
            print(f"  {model_name}: {overfitting:.4f} ({overfitting*100:.2f}%) - {status}")
        
        # Save results
        output_file = 'improved_four_models_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        return results_df

    def save_best_model(self):
        """Save the best model for API usage"""
        print(f"\n💾 Saving best model for API...")

        # Create models directory
        models_dir = Path('api/models')
        models_dir.mkdir(parents=True, exist_ok=True)

        # Find best model
        results_df = pd.DataFrame(self.results)
        results_df['Combined_Score'] = results_df['Test_Accuracy'] - (results_df['Overfitting'] * 2)
        best_idx = results_df['Combined_Score'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        
        # Map display name to model key
        model_key_map = {
            'Regularized Regression (Ridge)': 'Regularized Regression',
            'Regularized Regression (Lasso)': 'Regularized Regression',
            'Regularized Regression (ElasticNet)': 'Regularized Regression',
            'Improved Decision Tree': 'Improved Decision Tree',
            'Improved AdaBoost': 'Improved AdaBoost',
            'Improved XGBoost': 'Improved XGBoost'
        }
        
        model_key = model_key_map.get(best_model_name, best_model_name)
        best_model = self.models.get(model_key)
        
        if best_model is not None:
            model_path = models_dir / 'best_model.joblib'
            joblib.dump(best_model, model_path)
            
            # Save preprocessing components
            preprocessing_path = models_dir / 'preprocessing.joblib'
            preprocessing_components = {
                'scaler': self.scaler,
                'feature_selector': self.feature_selector
            }
            joblib.dump(preprocessing_components, preprocessing_path)
            
            print(f"✅ Best model ({best_model_name}) saved to: {model_path}")
            print(f"✅ Preprocessing components saved to: {preprocessing_path}")

            # Save metadata
            metadata = {
                'best_model_name': best_model_name,
                'feature_names': list(self.X_train.columns),
                'feature_count': len(self.X_train.columns),
                'suitability_mapping': CLASS_MAPPING,
                'model_performance': {
                    'test_accuracy': float(results_df.loc[best_idx, 'Test_Accuracy']),
                    'overfitting_gap': float(results_df.loc[best_idx, 'Overfitting']),
                    'cv_score': float(results_df.loc[best_idx, 'CV_Score']),
                    'cv_std': float(results_df.loc[best_idx, 'CV_Std'])
                }
            }

            metadata_path = models_dir / 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Model metadata saved to: {metadata_path}")

            return True
        else:
            print("❌ Best model not found!")
            return False

    def run_improved_training(self):
        """Run complete improved training"""
        print("IMPROVED TRAINING WITH ANTI-OVERFITTING TECHNIQUES")
        print("="*80)
        
        # Load data
        if not self.load_best_data():
            return False
        
        # Split data
        print(f"\nSplitting data with better validation...")
        self.split_data()
        
        # Train all improved models
        print(f"\nTraining improved models...")
        self.train_regularized_regression()
        self.train_improved_decision_tree()
        self.train_improved_adaboost()
        self.train_improved_xgboost()
        
        # Compare and save results
        results_df = self.compare_models_and_save()

        # Plot ROC curves
        print(f"\n🎨 Generating ROC curve visualizations...")
        self.plot_roc_curves()
        self.plot_individual_roc_curves()

        # Save best model
        self.save_best_model()

        print(f"\n" + "="*80)
        print("IMPROVED TRAINING COMPLETED!")
        print("="*80)

        print(f"\nKey Improvements Applied:")
        print(f"  🎯 Feature Selection - Reduced feature count")
        print(f"  📊 Better Data Splitting - More validation data")
        print(f"  🔄 SMOTE - Class balancing (if needed)")
        print(f"  🛡️ Regularization - L1/L2 penalties")
        print(f"  ⏹️ Early Stopping - Prevents overtraining")
        print(f"  ✂️ Pruning - Cost complexity pruning")
        print(f"  🎲 Sampling - Row/column subsampling")

        print(f"\n📁 Generated Files:")
        print(f"  📄 improved_four_models_results.csv - Model comparison results")
        print(f"  📈 roc_curves_comparison.png - ROC curves comparison")
        print(f"  📊 individual_roc_curves.png - Individual class ROC curves")
        print(f"  🤖 api/models/ - Best model files for API deployment")

        return True

def main():
    """Main function"""
    trainer = ImprovedFourModelsTrainer()
    trainer.run_improved_training()

if __name__ == "__main__":
    main()