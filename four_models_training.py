#!/usr/bin/env python3
"""
Training 4 Models: Linear Regression, Decision Tree, AdaBoost, XGBoost
Xuất kết quả ra file CSV
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class FourModelsTrainer:
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
        self.scaler = StandardScaler()
        self.models = {}
        self.results = []

    def load_best_data(self):
        """Load Jaccard similarity dataset"""
        dataset_path = 'progress/jd_cr_similarity.csv'

        try:
            self.data = pd.read_csv(dataset_path)
            print(f"✅ Loaded {len(self.data)} pairs from {dataset_path}")
            print("🎯 Using Jaccard similarity data from JD-CR matching")

            # Convert suitability labels to numeric
            suitability_mapping = {
                'Not Suitable': 0,
                'Moderately Suitable': 1,
                'Most Suitable': 2  # Fixed: was "Highly Suitable"
            }

            # Check if suitability column exists and map it
            if 'suitability' in self.data.columns:
                self.data['suitability_label'] = self.data['suitability'].map(suitability_mapping)
                # Handle any unmapped values
                self.data['suitability_label'] = self.data['suitability_label'].fillna(0).astype(int)
            else:
                print("❌ 'suitability' column not found in dataset")
                return False

            # Show label distribution
            label_counts = self.data['suitability_label'].value_counts().sort_index()
            total = len(self.data)
            print(f"\nLabel Distribution:")
            labels = {0: 'Not Suitable', 1: 'Moderately Suitable', 2: 'Most Suitable'}
            for label in [0, 1, 2]:
                count = label_counts.get(label, 0)
                percentage = (count / total) * 100
                print(f"  Class {label} ({labels[label]}): {count} samples ({percentage:.1f}%)")

            # Check for severe class imbalance
            min_class_pct = min([label_counts.get(i, 0) / total * 100 for i in [0, 1, 2]])
            if min_class_pct < 1.0:
                print(f"⚠️  SEVERE CLASS IMBALANCE DETECTED!")
                print(f"   Smallest class: {min_class_pct:.2f}% - This may cause overfitting!")

            # Check for data leakage indicators
            if 'total_similarity' in self.data.columns and 'primary_skills_sim' in self.data.columns:
                correlation = self.data['total_similarity'].corr(self.data['primary_skills_sim'])
                if correlation > 0.99:
                    print(f"⚠️  POTENTIAL DATA LEAKAGE: total_similarity highly correlated with primary_skills_sim ({correlation:.4f})")

            # Show data info
            print(f"\nDataset Info:")
            print(f"  Columns: {list(self.data.columns)}")
            print(f"  Shape: {self.data.shape}")
            print(f"  Unique JDs: {self.data['jd_id'].nunique()}")
            print(f"  Unique CR Categories: {self.data['cr_category'].nunique()}")

            return True

        except FileNotFoundError:
            print(f"❌ Dataset not found: {dataset_path}")
            print("Please make sure the Jaccard similarity file exists in the progress folder.")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False

    def prepare_features(self):
        """Prepare feature matrix from Jaccard similarity data"""
        # Exclude ID columns, target, string columns, and data leakage features
        exclude_cols = [
            'jd_id', 'cr_category', 'suitability_label', 'suitability',  # ID, string, and target columns
            'total_similarity', 'adj_weight'  # Data leakage - these may be used to create suitability labels
        ]

        # Get numeric columns only
        feature_cols = []
        for col in self.data.columns:
            if col not in exclude_cols:
                # Check if column is numeric
                if self.data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    feature_cols.append(col)
                else:
                    print(f"   Excluding non-numeric column: {col} (dtype: {self.data[col].dtype})")

        # Create feature matrix
        self.X = self.data[feature_cols].copy()
        self.y = self.data['suitability_label']

        # Handle missing values
        self.X = self.X.fillna(0)

        # Add engineered features based on available similarity scores
        available_sim_cols = [col for col in self.X.columns if 'sim' in col.lower()]

        if len(available_sim_cols) > 1:
            # Create weighted similarity score if multiple similarity columns exist
            if 'primary_skills_sim' in self.X.columns:
                weights = {'primary_skills_sim': 0.4}
                if 'secondary_skills_sim' in self.X.columns:
                    weights['secondary_skills_sim'] = 0.3
                if 'adjectives_sim' in self.X.columns:
                    weights['adjectives_sim'] = 0.2
                if 'adverbs_sim' in self.X.columns:
                    weights['adverbs_sim'] = 0.1

                # Create weighted similarity if not already present
                if 'weighted_similarity' not in self.X.columns:
                    weighted_sim = 0
                    for col, weight in weights.items():
                        if col in self.X.columns:
                            weighted_sim += self.X[col] * weight
                    self.X['weighted_similarity'] = weighted_sim

            # Add similarity variance (measure of consistency across different skill types)
            sim_cols = [col for col in self.X.columns if col.endswith('_sim')]
            if len(sim_cols) > 1:
                self.X['similarity_variance'] = self.X[sim_cols].var(axis=1)
                self.X['similarity_mean'] = self.X[sim_cols].mean(axis=1)
                self.X['similarity_max'] = self.X[sim_cols].max(axis=1)
                self.X['similarity_min'] = self.X[sim_cols].min(axis=1)

        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Features: {list(self.X.columns)}")
        print(f"Target distribution: {self.y.value_counts().sort_index().to_dict()}")

        return self.X, self.y

    def split_data(self, test_size=0.1, val_size=0.2, random_state=42):
        """Split data into train/validation/test"""
        X, y = self.prepare_features()
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        # Scale features for Linear Regression
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nData Split:")
        print(f"  Training: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(self.X_val)} samples ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")

    def train_linear_regression(self):
        """Train Linear Regression"""
        print(f"\n1. Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        train_pred = model.predict(self.X_train_scaled)
        val_pred = model.predict(self.X_val_scaled)
        test_pred = model.predict(self.X_test_scaled)
        
        # Convert to classification (round to nearest integer)
        train_pred_class = np.round(np.clip(train_pred, 0, 2)).astype(int)
        val_pred_class = np.round(np.clip(val_pred, 0, 2)).astype(int)
        test_pred_class = np.round(np.clip(test_pred, 0, 2)).astype(int)
        
        # Calculate accuracies
        train_acc = accuracy_score(self.y_train, train_pred_class)
        val_acc = accuracy_score(self.y_val, val_pred_class)
        test_acc = accuracy_score(self.y_test, test_pred_class)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mean = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Validation Accuracy: {val_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Cross-Val MSE: {cv_mean:.4f} (±{cv_std:.4f})")
        
        self.models['Linear Regression'] = model
        self.results.append({
            'Model': 'Linear Regression',
            'Train_Accuracy': train_acc,
            'Validation_Accuracy': val_acc,
            'Test_Accuracy': test_acc,
            'CV_Score': cv_mean,
            'CV_Std': cv_std,
            'Overfitting': abs(train_acc - val_acc)
        })
        
        return test_pred_class

    def train_decision_tree(self):
        """Train Decision Tree"""
        print(f"\n2. Training Decision Tree...")
        
        model = DecisionTreeClassifier(
            max_depth=5,  # Reduced to prevent overfitting
            min_samples_split=20,  # Increased to prevent overfitting
            min_samples_leaf=10,   # Increased to prevent overfitting
            max_features='sqrt',   # Feature subsampling
            random_state=42,
            class_weight='balanced'
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
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 5 Important Features:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
        
        self.models['Decision Tree'] = model
        self.results.append({
            'Model': 'Decision Tree',
            'Train_Accuracy': train_acc,
            'Validation_Accuracy': val_acc,
            'Test_Accuracy': test_acc,
            'CV_Score': cv_mean,
            'CV_Std': cv_std,
            'Overfitting': abs(train_acc - val_acc)
        })
        
        return test_pred

    def train_adaboost(self):
        """Train AdaBoost"""
        print(f"\n3. Training AdaBoost...")
        
        model = AdaBoostClassifier(
            n_estimators=50,  # Reduced to prevent overfitting
            learning_rate=0.001,  # Increased from 0.001
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
        
        self.models['AdaBoost'] = model
        self.results.append({
            'Model': 'AdaBoost',
            'Train_Accuracy': train_acc,
            'Validation_Accuracy': val_acc,
            'Test_Accuracy': test_acc,
            'CV_Score': cv_mean,
            'CV_Std': cv_std,
            'Overfitting': abs(train_acc - val_acc)
        })
        
        return test_pred

    def train_xgboost(self):
        """Train XGBoost"""
        print(f"\n4. Training XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,  # Reduced to prevent overfitting
            learning_rate=0.001,  # Increased from 0.001
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            class_weight='balanced'  # Handle class imbalance
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
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 5 Important Features:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
        
        self.models['XGBoost'] = model
        self.results.append({
            'Model': 'XGBoost',
            'Train_Accuracy': train_acc,
            'Validation_Accuracy': val_acc,
            'Test_Accuracy': test_acc,
            'CV_Score': cv_mean,
            'CV_Std': cv_std,
            'Overfitting': abs(train_acc - val_acc)
        })
        
        return test_pred

    def compare_models_and_save(self):
        """Compare models and save results to CSV"""
        print(f"\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('Test_Accuracy', ascending=False)
        
        # Display results
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Find best model
        best_idx = results_df['Test_Accuracy'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        best_test_acc = results_df.loc[best_idx, 'Test_Accuracy']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Test Accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
        
        # Target comparison
        target_accuracy = 0.9514  # From research paper
        if best_test_acc >= target_accuracy:
            print(f"✅ TARGET ACHIEVED! Test Accuracy >= {target_accuracy*100:.2f}%")
        else:
            print(f"⚠️  Target not reached. Need {target_accuracy*100:.2f}%, got {best_test_acc*100:.2f}%")
            gap = target_accuracy - best_test_acc
            print(f"   Gap: {gap*100:.2f}%")
        
        # Save results to CSV
        output_file = 'four_models_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Create detailed results with predictions
        detailed_results = []
        
        for model_name, model in self.models.items():
            if model_name == 'Linear Regression':
                test_pred = model.predict(self.X_test_scaled)
                test_pred = np.round(np.clip(test_pred, 0, 2)).astype(int)
            else:
                test_pred = model.predict(self.X_test)
            
            test_acc = accuracy_score(self.y_test, test_pred)
            
            # Classification report
            report = classification_report(self.y_test, test_pred, output_dict=True)
            
            # Handle missing classes safely
            result_dict = {
                'Model': model_name,
                'Test_Accuracy': test_acc,
                'Macro_Avg_Precision': report['macro avg']['precision'],
                'Macro_Avg_Recall': report['macro avg']['recall'],
                'Macro_Avg_F1': report['macro avg']['f1-score'],
                'Weighted_Avg_Precision': report['weighted avg']['precision'],
                'Weighted_Avg_Recall': report['weighted avg']['recall'],
                'Weighted_Avg_F1': report['weighted avg']['f1-score']
            }

            # Add class-specific metrics if they exist
            for class_id in ['0', '1', '2']:
                if class_id in report:
                    result_dict[f'Precision_Class_{class_id}'] = report[class_id]['precision']
                    result_dict[f'Recall_Class_{class_id}'] = report[class_id]['recall']
                    result_dict[f'F1_Class_{class_id}'] = report[class_id]['f1-score']
                else:
                    result_dict[f'Precision_Class_{class_id}'] = 0.0
                    result_dict[f'Recall_Class_{class_id}'] = 0.0
                    result_dict[f'F1_Class_{class_id}'] = 0.0

            detailed_results.append(result_dict)
        
        # Save detailed results
        detailed_df = pd.DataFrame(detailed_results)
        detailed_output_file = 'four_models_detailed_results.csv'
        detailed_df.to_csv(detailed_output_file, index=False)
        print(f"💾 Detailed results saved to: {detailed_output_file}")
        
        return results_df, detailed_df

    def run_four_models_training(self):
        """Run complete training for 4 models"""
        print("TRAINING 4 MODELS: Linear Regression, Decision Tree, AdaBoost, XGBoost")
        print("="*80)
        
        # Load data
        if not self.load_best_data():
            return False
        
        # Split data
        print(f"\nSplitting data...")
        self.split_data()
        
        # Train all 4 models
        print(f"\nTraining models...")
        self.train_linear_regression()
        self.train_decision_tree()
        self.train_adaboost()
        self.train_xgboost()
        
        # Compare and save results
        results_df, detailed_df = self.compare_models_and_save()
        
        print(f"\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        
        print(f"\nOutput Files:")
        print(f"  📄 four_models_results.csv - Summary results")
        print(f"  📄 four_models_detailed_results.csv - Detailed metrics")
        
        return True

def main():
    """Main function"""
    trainer = FourModelsTrainer()
    trainer.run_four_models_training()

if __name__ == "__main__":
    main()
