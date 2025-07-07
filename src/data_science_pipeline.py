"""
Johns Hopkins Data Science Capstone Project
Advanced Predictive Analytics Platform

This module implements a comprehensive data science pipeline demonstrating
all competencies from the Johns Hopkins Data Science Specialization:
- Data acquisition and cleaning
- Exploratory data analysis
- Statistical inference
- Regression modeling
- Machine learning
- Reproducible research
- Data products development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DataSciencePipeline:
    """
    Comprehensive Data Science Pipeline for Johns Hopkins Capstone
    Demonstrates mastery of the complete data science workflow
    """
    
    def __init__(self):
        self.raw_data = {}
        self.processed_data = {}
        self.models = {}
        self.results = {}
        self.pipeline_start_time = pd.Timestamp.now()
        
    def generate_synthetic_datasets(self):
        """
        Generate multiple synthetic datasets for comprehensive analysis
        Simulates real-world data science scenarios
        """
        print("🔄 Generating synthetic datasets...")
        
        # Dataset 1: Customer Analytics
        np.random.seed(42)
        n_customers = 5000
        
        customer_data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(40, 15, n_customers).clip(18, 80),
            'income': np.random.lognormal(10.5, 0.5, n_customers),
            'education_years': np.random.normal(14, 3, n_customers).clip(8, 20),
            'family_size': np.random.poisson(2.5, n_customers) + 1,
            'years_customer': np.random.exponential(3, n_customers).clip(0, 20),
            'monthly_spending': np.random.gamma(2, 200, n_customers),
            'satisfaction_score': np.random.beta(2, 1, n_customers) * 10,
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers),
            'channel_preference': np.random.choice(['Online', 'Store', 'Mobile', 'Phone'], n_customers)
        }
        
        # Add correlation between variables
        for i in range(n_customers):
            # Income affects spending
            customer_data['monthly_spending'][i] *= (customer_data['income'][i] / 50000) ** 0.3
            # Satisfaction affects spending
            customer_data['monthly_spending'][i] *= (customer_data['satisfaction_score'][i] / 10) ** 0.2
            
        # Create churn target variable
        churn_probability = (
            0.1 + 
            0.3 * (1 - customer_data['satisfaction_score'] / 10) +
            0.2 * (customer_data['years_customer'] < 1).astype(int) +
            0.1 * (customer_data['monthly_spending'] < np.percentile(customer_data['monthly_spending'], 25)).astype(int)
        )
        customer_data['churned'] = np.random.binomial(1, churn_probability, n_customers)
        
        self.raw_data['customers'] = pd.DataFrame(customer_data)
        
        # Dataset 2: Financial Market Data
        n_days = 1000
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        # Generate correlated stock prices
        returns = np.random.multivariate_normal(
            [0.0005, 0.0003, 0.0007], 
            [[0.0004, 0.0002, 0.0001],
             [0.0002, 0.0006, 0.0003],
             [0.0001, 0.0003, 0.0008]], 
            n_days
        )
        
        prices = np.zeros((n_days, 3))
        prices[0] = [100, 50, 200]  # Initial prices
        
        for i in range(1, n_days):
            prices[i] = prices[i-1] * (1 + returns[i])
        
        market_data = {
            'date': dates,
            'stock_a_price': prices[:, 0],
            'stock_b_price': prices[:, 1],
            'stock_c_price': prices[:, 2],
            'volume_a': np.random.lognormal(12, 0.5, n_days),
            'volume_b': np.random.lognormal(11.5, 0.6, n_days),
            'volume_c': np.random.lognormal(11.8, 0.4, n_days),
            'market_sentiment': np.random.normal(0, 1, n_days)
        }
        
        self.raw_data['market'] = pd.DataFrame(market_data)
        
        # Dataset 3: Healthcare Outcomes
        n_patients = 3000
        
        healthcare_data = {
            'patient_id': range(1, n_patients + 1),
            'age': np.random.normal(55, 20, n_patients).clip(0, 100),
            'bmi': np.random.normal(26, 5, n_patients).clip(15, 50),
            'blood_pressure_systolic': np.random.normal(130, 20, n_patients).clip(80, 200),
            'cholesterol': np.random.normal(200, 40, n_patients).clip(100, 400),
            'smoking': np.random.binomial(1, 0.2, n_patients),
            'exercise_hours_week': np.random.exponential(3, n_patients).clip(0, 20),
            'family_history': np.random.binomial(1, 0.3, n_patients),
            'stress_level': np.random.uniform(1, 10, n_patients)
        }
        
        # Create health risk score
        risk_score = (
            0.02 * healthcare_data['age'] +
            0.1 * (healthcare_data['bmi'] > 30).astype(int) +
            0.05 * (healthcare_data['blood_pressure_systolic'] > 140).astype(int) +
            0.03 * (healthcare_data['cholesterol'] > 240).astype(int) +
            0.15 * healthcare_data['smoking'] +
            0.1 * healthcare_data['family_history'] +
            0.02 * healthcare_data['stress_level'] -
            0.02 * healthcare_data['exercise_hours_week']
        )
        
        healthcare_data['health_risk_score'] = risk_score
        healthcare_data['high_risk'] = (risk_score > np.percentile(risk_score, 75)).astype(int)
        
        self.raw_data['healthcare'] = pd.DataFrame(healthcare_data)
        
        print(f"✅ Generated {len(self.raw_data)} datasets:")
        for name, df in self.raw_data.items():
            print(f"   - {name}: {len(df)} records, {len(df.columns)} features")
    
    def exploratory_data_analysis(self):
        """
        Comprehensive exploratory data analysis
        Demonstrates statistical analysis and visualization skills
        """
        print("\n📊 Performing Exploratory Data Analysis...")
        
        self.eda_results = {}
        
        for dataset_name, df in self.raw_data.items():
            print(f"\n--- Analyzing {dataset_name.upper()} Dataset ---")
            
            # Basic statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            stats = {
                'shape': df.shape,
                'missing_values': df.isnull().sum().sum(),
                'numeric_features': len(numeric_cols),
                'categorical_features': len(categorical_cols),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Statistical summaries
            if len(numeric_cols) > 0:
                stats['numeric_summary'] = df[numeric_cols].describe()
                stats['correlations'] = df[numeric_cols].corr()
            
            if len(categorical_cols) > 0:
                stats['categorical_summary'] = {col: df[col].value_counts() for col in categorical_cols}
            
            self.eda_results[dataset_name] = stats
            
            print(f"   Shape: {stats['shape']}")
            print(f"   Missing values: {stats['missing_values']}")
            print(f"   Features: {stats['numeric_features']} numeric, {stats['categorical_features']} categorical")
    
    def feature_engineering(self):
        """
        Advanced feature engineering and data preprocessing
        """
        print("\n🔧 Performing Feature Engineering...")
        
        # Customer data feature engineering
        customers = self.raw_data['customers'].copy()
        
        # Create derived features
        customers['income_per_family_member'] = customers['income'] / customers['family_size']
        customers['spending_to_income_ratio'] = customers['monthly_spending'] / customers['income']
        customers['customer_value_score'] = (
            customers['monthly_spending'] * customers['years_customer'] * 
            (customers['satisfaction_score'] / 10)
        )
        customers['age_group'] = pd.cut(customers['age'], bins=[0, 30, 50, 70, 100], 
                                       labels=['Young', 'Middle', 'Senior', 'Elder'])
        customers['income_tier'] = pd.qcut(customers['income'], q=4, 
                                          labels=['Low', 'Medium', 'High', 'Premium'])
        
        self.processed_data['customers'] = customers
        
        # Market data feature engineering
        market = self.raw_data['market'].copy()
        
        # Technical indicators
        for stock in ['stock_a_price', 'stock_b_price', 'stock_c_price']:
            # Moving averages
            market[f'{stock}_ma_7'] = market[stock].rolling(window=7).mean()
            market[f'{stock}_ma_30'] = market[stock].rolling(window=30).mean()
            
            # Returns
            market[f'{stock}_return'] = market[stock].pct_change()
            market[f'{stock}_volatility'] = market[f'{stock}_return'].rolling(window=30).std()
            
            # Price momentum
            market[f'{stock}_momentum'] = market[stock] / market[stock].shift(10) - 1
        
        # Market indicators
        market['market_trend'] = (
            (market['stock_a_price_ma_7'] > market['stock_a_price_ma_30']).astype(int) +
            (market['stock_b_price_ma_7'] > market['stock_b_price_ma_30']).astype(int) +
            (market['stock_c_price_ma_7'] > market['stock_c_price_ma_30']).astype(int)
        )
        
        self.processed_data['market'] = market
        
        # Healthcare data feature engineering
        healthcare = self.raw_data['healthcare'].copy()
        
        # Create composite health indicators
        healthcare['bmi_category'] = pd.cut(healthcare['bmi'], 
                                           bins=[0, 18.5, 25, 30, 50], 
                                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        healthcare['bp_category'] = pd.cut(healthcare['blood_pressure_systolic'],
                                          bins=[0, 120, 140, 180, 250],
                                          labels=['Normal', 'Elevated', 'High', 'Crisis'])
        healthcare['lifestyle_score'] = (
            (healthcare['exercise_hours_week'] > 5).astype(int) * 2 +
            (1 - healthcare['smoking']) * 3 +
            (healthcare['stress_level'] < 5).astype(int) * 1
        )
        
        self.processed_data['healthcare'] = healthcare
        
        print("✅ Feature engineering completed for all datasets")
    
    def build_predictive_models(self):
        """
        Build and evaluate multiple machine learning models
        Demonstrates ML competencies from the specialization
        """
        print("\n🤖 Building Predictive Models...")
        
        # Model 1: Customer Churn Prediction (Classification)
        customers = self.processed_data['customers']
        
        # Prepare features for churn prediction
        feature_cols = ['age', 'income', 'education_years', 'family_size', 
                       'years_customer', 'monthly_spending', 'satisfaction_score',
                       'income_per_family_member', 'spending_to_income_ratio', 
                       'customer_value_score']
        
        X_churn = customers[feature_cols].fillna(customers[feature_cols].median())
        y_churn = customers['churned']
        
        # Add categorical features
        le_region = LabelEncoder()
        le_channel = LabelEncoder()
        X_churn['region_encoded'] = le_region.fit_transform(customers['region'])
        X_churn['channel_encoded'] = le_channel.fit_transform(customers['channel_preference'])
        
        # Split and scale data
        X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
            X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
        )
        
        scaler_churn = StandardScaler()
        X_train_churn_scaled = scaler_churn.fit_transform(X_train_churn)
        X_test_churn_scaled = scaler_churn.transform(X_test_churn)
        
        # Train models
        models_churn = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        churn_results = {}
        for name, model in models_churn.items():
            if name == 'Logistic Regression':
                model.fit(X_train_churn_scaled, y_train_churn)
                y_pred = model.predict(X_test_churn_scaled)
            else:
                model.fit(X_train_churn, y_train_churn)
                y_pred = model.predict(X_test_churn)
            
            accuracy = accuracy_score(y_test_churn, y_pred)
            churn_results[name] = {
                'accuracy': accuracy,
                'model': model,
                'predictions': y_pred
            }
            
            print(f"   {name} - Churn Prediction Accuracy: {accuracy:.3f}")
        
        self.models['churn_prediction'] = churn_results
        
        # Model 2: Stock Price Prediction (Regression)
        market = self.processed_data['market'].dropna()
        
        # Predict next day stock price
        feature_cols_market = ['stock_a_price_ma_7', 'stock_a_price_ma_30', 
                              'volume_a', 'market_sentiment']
        
        # Only use available columns
        available_cols = [col for col in feature_cols_market if col in market.columns]
        X_market = market[available_cols].fillna(method='ffill').dropna()
        y_market = market['stock_a_price'].shift(-1).dropna()  # Next day price
        
        # Align X and y
        min_len = min(len(X_market), len(y_market))
        X_market = X_market.iloc[:min_len]
        y_market = y_market.iloc[:min_len]
        
        X_train_market, X_test_market, y_train_market, y_test_market = train_test_split(
            X_market, y_market, test_size=0.2, random_state=42
        )
        
        models_market = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        market_results = {}
        for name, model in models_market.items():
            model.fit(X_train_market, y_train_market)
            y_pred = model.predict(X_test_market)
            
            mse = mean_squared_error(y_test_market, y_pred)
            rmse = np.sqrt(mse)
            
            market_results[name] = {
                'rmse': rmse,
                'model': model,
                'predictions': y_pred,
                'actual': y_test_market
            }
            
            print(f"   {name} - Stock Price Prediction RMSE: {rmse:.3f}")
        
        self.models['stock_prediction'] = market_results
        
        # Model 3: Health Risk Assessment (Classification)
        healthcare = self.processed_data['healthcare']
        
        feature_cols_health = ['age', 'bmi', 'blood_pressure_systolic', 'cholesterol',
                              'smoking', 'exercise_hours_week', 'family_history', 
                              'stress_level', 'lifestyle_score']
        
        X_health = healthcare[feature_cols_health]
        y_health = healthcare['high_risk']
        
        X_train_health, X_test_health, y_train_health, y_test_health = train_test_split(
            X_health, y_health, test_size=0.2, random_state=42, stratify=y_health
        )
        
        scaler_health = StandardScaler()
        X_train_health_scaled = scaler_health.fit_transform(X_train_health)
        X_test_health_scaled = scaler_health.transform(X_test_health)
        
        models_health = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        health_results = {}
        for name, model in models_health.items():
            if name == 'Logistic Regression':
                model.fit(X_train_health_scaled, y_train_health)
                y_pred = model.predict(X_test_health_scaled)
            else:
                model.fit(X_train_health, y_train_health)
                y_pred = model.predict(X_test_health)
            
            accuracy = accuracy_score(y_test_health, y_pred)
            health_results[name] = {
                'accuracy': accuracy,
                'model': model,
                'predictions': y_pred
            }
            
            print(f"   {name} - Health Risk Prediction Accuracy: {accuracy:.3f}")
        
        self.models['health_risk'] = health_results
        
        print("✅ All predictive models built and evaluated")
    
    def generate_insights_report(self):
        """
        Generate comprehensive insights and recommendations
        """
        print("\n📋 Generating Insights Report...")
        
        insights = {
            'executive_summary': {
                'total_records_analyzed': sum(len(df) for df in self.raw_data.values()),
                'models_built': len(self.models),
                'datasets_processed': len(self.processed_data),
                'analysis_completion_time': pd.Timestamp.now() - self.pipeline_start_time
            },
            'key_findings': {},
            'model_performance': {},
            'recommendations': []
        }
        
        # Customer insights
        customers = self.processed_data['customers']
        insights['key_findings']['customer_analytics'] = {
            'churn_rate': customers['churned'].mean(),
            'avg_customer_value': customers['customer_value_score'].mean(),
            'high_value_customers_pct': (customers['customer_value_score'] > 
                                       customers['customer_value_score'].quantile(0.8)).mean(),
            'satisfaction_correlation_spending': customers['satisfaction_score'].corr(customers['monthly_spending'])
        }
        
        # Market insights
        market = self.processed_data['market']
        insights['key_findings']['market_analytics'] = {
            'avg_daily_return_stock_a': market['stock_a_price_return'].mean(),
            'volatility_stock_a': market['stock_a_price_return'].std(),
            'correlation_sentiment_returns': market['market_sentiment'].corr(market['stock_a_price_return'])
        }
        
        # Healthcare insights
        healthcare = self.processed_data['healthcare']
        insights['key_findings']['healthcare_analytics'] = {
            'high_risk_patients_pct': healthcare['high_risk'].mean(),
            'avg_health_risk_score': healthcare['health_risk_score'].mean(),
            'lifestyle_impact': healthcare['lifestyle_score'].corr(healthcare['health_risk_score'])
        }
        
        # Model performance summary
        for model_type, results in self.models.items():
            best_model = max(results.items(), key=lambda x: x[1].get('accuracy', 1-x[1].get('rmse', float('inf'))))
            insights['model_performance'][model_type] = {
                'best_model': best_model[0],
                'performance': best_model[1].get('accuracy', f"RMSE: {best_model[1].get('rmse', 'N/A')}")
            }
        
        # Generate recommendations
        insights['recommendations'] = [
            "Implement targeted retention campaigns for high-risk churn customers",
            "Focus on improving customer satisfaction to increase spending",
            "Develop risk-based pricing models for financial products",
            "Create personalized health intervention programs for high-risk patients",
            "Implement real-time monitoring systems for all predictive models"
        ]
        
        self.results = insights
        
        # Save results to file
        results_summary = f"""
# Data Science Analysis Results Summary

## Executive Summary
- **Total Records Analyzed:** {insights['executive_summary']['total_records_analyzed']:,}
- **Models Built:** {insights['executive_summary']['models_built']}
- **Datasets Processed:** {insights['executive_summary']['datasets_processed']}
- **Analysis Time:** {insights['executive_summary']['analysis_completion_time']}

## Key Performance Metrics
"""
        
        for model_type, perf in insights['model_performance'].items():
            results_summary += f"- **{model_type.replace('_', ' ').title()}:** {perf['best_model']} ({perf['performance']})\n"
        
        results_summary += f"""
## Business Insights
- **Customer Churn Rate:** {insights['key_findings']['customer_analytics']['churn_rate']:.1%}
- **High-Value Customers:** {insights['key_findings']['customer_analytics']['high_value_customers_pct']:.1%}
- **High-Risk Patients:** {insights['key_findings']['healthcare_analytics']['high_risk_patients_pct']:.1%}

## Recommendations
"""
        for i, rec in enumerate(insights['recommendations'], 1):
            results_summary += f"{i}. {rec}\n"
        
        with open('../reports/analysis_summary.md', 'w') as f:
            f.write(results_summary)
        
        print("✅ Comprehensive insights report generated")
        return insights
    
    def run_complete_analysis(self):
        """
        Execute the complete data science pipeline
        """
        print("🚀 Starting Johns Hopkins Data Science Capstone Analysis...")
        print("=" * 60)
        
        try:
            # Execute pipeline steps
            self.generate_synthetic_datasets()
            self.exploratory_data_analysis()
            self.feature_engineering()
            self.build_predictive_models()
            insights = self.generate_insights_report()
            
            print("\n" + "=" * 60)
            print("🎉 Data Science Pipeline Completed Successfully!")
            print("=" * 60)
            
            return insights
            
        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            return None

if __name__ == "__main__":
    # Execute the complete data science pipeline
    pipeline = DataSciencePipeline()
    results = pipeline.run_complete_analysis()
    
    if results:
        print("\n📊 FINAL RESULTS SUMMARY:")
        print(f"✅ Analyzed {results['executive_summary']['total_records_analyzed']:,} records")
        print(f"✅ Built {results['executive_summary']['models_built']} predictive models")
        print(f"✅ Generated actionable insights for business decision-making")
        print("\n🎓 Johns Hopkins Data Science Capstone - COMPLETED!")
    else:
        print("❌ Analysis failed. Please check the logs for details.")

