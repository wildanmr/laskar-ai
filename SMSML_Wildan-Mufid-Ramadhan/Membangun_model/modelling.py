import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
import os
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def setup_mlflow_tracking(use_dagshub=False):
    """Setup MLflow tracking (local or DagsHub)"""
    if use_dagshub:
        try:
            import dagshub
            # Setup DagsHub integration
            dagshub.init(repo_owner="wildanmr", repo_name="SMSML_Wildan-Mufid-Ramadhan", mlflow=True)
            mlflow.set_tracking_uri("https://dagshub.com/wildanmr/SMSML_Wildan-Mufid-Ramadhan.mlflow")
            print("✅ Using DagsHub MLflow tracking")
            return True
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to DagsHub: {e}")
            print("Falling back to local MLflow tracking")
            mlflow.set_tracking_uri("file:./mlruns")
            return False
    else:
        # Set MLflow tracking URI to local (default)
        mlflow.set_tracking_uri("file:./mlruns")
        print("✅ Using local MLflow tracking")
        print("✅ Make sure to run 'mlflow ui' in terminal to view results")
        return True

def setup_experiment():
    """Setup MLflow experiment properly"""
    experiment_name = "Basic_ML_Experiment"
    
    try:
        # Try to get existing experiment first
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"✅ Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
        
        # Set the experiment as active
        mlflow.set_experiment(experiment_name)
        
        # Verify the experiment is set correctly
        current_experiment = mlflow.get_experiment_by_name(experiment_name)
        print(f"✅ Current active experiment: {current_experiment.name}")
        
        return current_experiment.experiment_id
        
    except Exception as e:
        print(f"❌ Error setting up experiment: {e}")
        raise

def load_and_prepare_data():
    """Load preprocessed data"""
    try:
        df = pd.read_csv('diabetes_preprocessed.csv')
        print(f"✅ Data loaded successfully: {df.shape}")
        
        # Pisahkan features dan target
        X = df.drop('Diabetes_binary', axis=1)
        y = df['Diabetes_binary']
        
        print(f"✅ Features: {X.shape[1]}, Samples: {len(X)}")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    except FileNotFoundError:
        print("❌ Error: 'diabetes_preprocessed.csv' not found")
        print("Please make sure the file exists in the current directory")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """Create and save confusion matrix plot"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Diabetes', 'Diabetes'], 
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Create artifacts directory if not exists
        os.makedirs('artifacts', exist_ok=True)
        plot_path = f'artifacts/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved to: {plot_path}")
        return plot_path
    except Exception as e:
        print(f"Warning: Could not create confusion matrix plot: {e}")
        return None

def log_additional_metrics(y_true, y_pred, y_pred_proba=None):
    """Log additional metrics manually"""
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # Calculate additional metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Log metrics
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)
        
        # ROC AUC if probabilities are available
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                mlflow.log_metric("roc_auc", roc_auc)
                print(f"✅ ROC AUC: {roc_auc:.4f}")
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")
        
        print(f"✅ Additional metrics logged - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    except Exception as e:
        print(f"Warning: Could not log additional metrics: {e}")
        return {}

def train_basic_model(experiment_id):
    """Train basic Random Forest model with controlled MLflow logging"""
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Disable autolog to have full control
    mlflow.sklearn.autolog(disable=True)
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="RandomForest_Basic") as run:
        print(f"✅ Started MLflow run: {run.info.run_id}")
        print(f"✅ Experiment ID: {run.info.experiment_id}")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        print("🔄 Training RandomForest model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters manually
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": X_train.shape[1],
            "timestamp": datetime.now().isoformat()
        })
        
        # Log metrics manually
        mlflow.log_metric("accuracy", accuracy)
        
        # Log additional metrics
        additional_metrics = log_additional_metrics(y_test, y_pred, y_pred_proba)
        
        # Create and log confusion matrix
        cm_path = create_confusion_matrix_plot(y_test, y_pred, "Random Forest")
        if cm_path and os.path.exists(cm_path):
            try:
                mlflow.log_artifact(cm_path)
                print("✅ Confusion matrix logged to MLflow")
            except Exception as e:
                print(f"Warning: Could not log confusion matrix: {e}")
        
        # Log classification report
        try:
            report = classification_report(y_test, y_pred)
            os.makedirs('artifacts', exist_ok=True)
            report_path = 'artifacts/classification_report_rf.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            mlflow.log_artifact(report_path)
            print("✅ Classification report logged")
        except Exception as e:
            print(f"Warning: Could not log classification report: {e}")
        
        # Log model (single model only)
        try:
            input_example = X_train.head(1) if hasattr(X_train, 'head') else X_train[:1]
            
            # Log model without registering to avoid duplicates
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="random_forest_model",
                input_example=input_example
            )
            print("✅ Random Forest model logged to MLflow")
            
        except Exception as e:
            print(f"Error: Could not log model: {e}")
        
        print(f"✅ Random Forest model trained successfully!")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   MLflow Run ID: {run.info.run_id}")
        
        # Save model to local file
        os.makedirs('saved_models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'saved_models/basic_random_forest_model_{timestamp}.pkl'
        
        joblib.dump(model, model_filename)
        print(f"✅ Model saved locally to: {model_filename}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, accuracy, model_filename, additional_metrics

def train_logistic_regression(experiment_id):
    """Train Logistic Regression model with controlled MLflow logging"""
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Disable autolog to have full control
    mlflow.sklearn.autolog(disable=True)
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="LogisticRegression_Basic") as run:
        print(f"✅ Started MLflow run: {run.info.run_id}")
        print(f"✅ Experiment ID: {run.info.experiment_id}")
        
        # Train model
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        print("🔄 Training Logistic Regression model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters manually
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "random_state": 42,
            "max_iter": 1000,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": X_train.shape[1],
            "feature_scaling": "StandardScaler",
            "timestamp": datetime.now().isoformat()
        })
        
        # Log metrics manually
        mlflow.log_metric("accuracy", accuracy)
        
        # Log additional metrics
        additional_metrics = log_additional_metrics(y_test, y_pred, y_pred_proba)
        
        # Create and log confusion matrix
        cm_path = create_confusion_matrix_plot(y_test, y_pred, "Logistic Regression")
        if cm_path and os.path.exists(cm_path):
            try:
                mlflow.log_artifact(cm_path)
                print("✅ Confusion matrix logged to MLflow")
            except Exception as e:
                print(f"Warning: Could not log confusion matrix: {e}")
        
        # Log classification report
        try:
            report = classification_report(y_test, y_pred)
            os.makedirs('artifacts', exist_ok=True)
            report_path = 'artifacts/classification_report_lr.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            mlflow.log_artifact(report_path)
            print("✅ Classification report logged")
        except Exception as e:
            print(f"Warning: Could not log classification report: {e}")
        
        # Log model (single model only)
        try:
            input_example = X_train.head(1) if hasattr(X_train, 'head') else X_train[:1]
            
            # Log model without registering to avoid duplicates
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="logistic_regression_model",
                input_example=input_example
            )
            print("✅ Logistic Regression model logged to MLflow")
            
        except Exception as e:
            print(f"Error: Could not log model: {e}")
        
        print(f"✅ Logistic Regression trained successfully!")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   MLflow Run ID: {run.info.run_id}")
        
        # Save model to local file
        os.makedirs('saved_models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'saved_models/basic_logistic_regression_model_{timestamp}.pkl'
        
        joblib.dump(model, model_filename)
        print(f"✅ Model saved locally to: {model_filename}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, accuracy, model_filename, additional_metrics

def compare_models(rf_accuracy, rf_metrics, lr_accuracy, lr_metrics):
    """Compare model performances"""
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    
    print(f"Random Forest:")
    print(f"  - Accuracy: {rf_accuracy:.4f}")
    print(f"  - Precision: {rf_metrics.get('precision', 'N/A'):.4f}")
    print(f"  - Recall: {rf_metrics.get('recall', 'N/A'):.4f}")
    print(f"  - F1-Score: {rf_metrics.get('f1', 'N/A'):.4f}")
    
    print(f"\nLogistic Regression:")
    print(f"  - Accuracy: {lr_accuracy:.4f}")
    print(f"  - Precision: {lr_metrics.get('precision', 'N/A'):.4f}")
    print(f"  - Recall: {lr_metrics.get('recall', 'N/A'):.4f}")
    print(f"  - F1-Score: {lr_metrics.get('f1', 'N/A'):.4f}")
    
    # Determine best model
    best_model = "Random Forest" if rf_accuracy > lr_accuracy else "Logistic Regression"
    print(f"\n🏆 Best Model: {best_model}")

if __name__ == "__main__":
    print("🚀 Starting Basic MLflow Training...")
    print("="*60)
    
    # Choose tracking method (set to True if you want to use DagsHub)
    USE_DAGSHUB = False 
    
    # Setup MLflow tracking
    setup_mlflow_tracking(use_dagshub=USE_DAGSHUB)
    
    try:
        # Setup experiment BEFORE training
        experiment_id = setup_experiment()
        
        print("\n🔄 Training Model 1: Random Forest")
        print("-" * 40)
        rf_model, rf_accuracy, rf_model_file, rf_metrics = train_basic_model(experiment_id)
        
        print("\n🔄 Training Model 2: Logistic Regression")
        print("-" * 40)
        lr_model, lr_accuracy, lr_model_file, lr_metrics = train_logistic_regression(experiment_id)
        
        # Compare models
        compare_models(rf_accuracy, rf_metrics, lr_accuracy, lr_metrics)
        
        print("\n" + "="*60)
        print("📁 SAVED FILES:")
        print("="*60)
        print(f"Random Forest Model: {rf_model_file}")
        print(f"Logistic Regression Model: {lr_model_file}")
        
        if not USE_DAGSHUB:
            print(f"\n🌐 MLflow Tracking UI available at: http://localhost:5000")
            print("   Run 'mlflow ui' in terminal to view the dashboard")
        else:
            print(f"\n🌐 Check DagsHub MLflow UI at:")
            print("   https://dagshub.com/wildanmr/SMSML_Wildan-Mufid-Ramadhan.mlflow")
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your data file and dependencies.")