#!/usr/bin/env python3
"""
Johns Hopkins Data Science Pipeline
Comprehensive Data Science Capstone Project
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSciencePipeline:
    """Comprehensive Data Science Pipeline"""
    
    def __init__(self):
        self.db_path = "data_science.db"
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.init_database()
        logger.info("Data Science Pipeline initialized")
    
    def init_database(self):
        """Initialize data science database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Customer churn data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_churn (
                customer_id TEXT PRIMARY KEY,
                tenure INTEGER,
                monthly_charges DECIMAL(8,2),
                total_charges DECIMAL(10,2),
                contract_type TEXT,
                payment_method TEXT,
                internet_service TEXT,
                churn INTEGER
            )
        """)
        
        # Stock price data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                date DATE,
                symbol TEXT,
                open_price DECIMAL(10,2),
                high_price DECIMAL(10,2),
                low_price DECIMAL(10,2),
                close_price DECIMAL(10,2),
                volume INTEGER
            )
        """)
        
        # Health risk data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_risk (
                patient_id TEXT PRIMARY KEY,
                age INTEGER,
                bmi DECIMAL(5,2),
                blood_pressure_systolic INTEGER,
                blood_pressure_diastolic INTEGER,
                cholesterol DECIMAL(5,2),
                smoking INTEGER,
                exercise_hours DECIMAL(3,1),
                risk_level INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Data science database initialized")
    
    def generate_sample_datasets(self):
        """Generate comprehensive sample datasets"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM customer_churn")
        cursor.execute("DELETE FROM stock_prices")
        cursor.execute("DELETE FROM health_risk")
        
        # Generate customer churn data
        churn_data = []
        contract_types = ['Month-to-month', 'One year', 'Two year']
        payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
        internet_services = ['DSL', 'Fiber optic', 'No']
        
        for i in range(3000):
            customer_id = f"CUST_{i+1:05d}"
            tenure = np.random.randint(1, 72)
            monthly_charges = np.random.uniform(20, 120)
            total_charges = monthly_charges * tenure + np.random.uniform(-100, 100)
            contract_type = np.random.choice(contract_types)
            payment_method = np.random.choice(payment_methods)
            internet_service = np.random.choice(internet_services)
            
            # Churn probability based on features
            churn_prob = 0.1
            if contract_type == 'Month-to-month':
                churn_prob += 0.3
            if payment_method == 'Electronic check':
                churn_prob += 0.2
            if tenure < 12:
                churn_prob += 0.2
            if monthly_charges > 80:
                churn_prob += 0.1
            
            churn = 1 if np.random.random() < churn_prob else 0
            
            churn_data.append((customer_id, tenure, monthly_charges, total_charges,
                             contract_type, payment_method, internet_service, churn))
        
        cursor.executemany("""
            INSERT INTO customer_churn 
            (customer_id, tenure, monthly_charges, total_charges, contract_type, payment_method, internet_service, churn)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, churn_data)
        
        # Generate stock price data
        stock_data = []
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        base_date = datetime.now() - timedelta(days=365)
        
        for symbol in symbols:
            price = np.random.uniform(100, 300)  # Starting price
            
            for i in range(365):
                date = base_date + timedelta(days=i)
                
                # Random walk for price
                change = np.random.normal(0, 0.02)  # 2% daily volatility
                price = price * (1 + change)
                
                # Generate OHLC
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = price * (1 + np.random.normal(0, 0.005))
                close_price = price
                volume = np.random.randint(1000000, 10000000)
                
                stock_data.append((date.date(), symbol, open_price, high, low, close_price, volume))
        
        cursor.executemany("""
            INSERT INTO stock_prices 
            (date, symbol, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, stock_data)
        
        # Generate health risk data
        health_data = []
        for i in range(2000):
            patient_id = f"PAT_{i+1:05d}"
            age = np.random.randint(18, 80)
            bmi = np.random.normal(25, 5)
            bp_systolic = np.random.randint(90, 180)
            bp_diastolic = np.random.randint(60, 120)
            cholesterol = np.random.normal(200, 40)
            smoking = np.random.choice([0, 1], p=[0.7, 0.3])
            exercise_hours = np.random.uniform(0, 10)
            
            # Calculate risk level based on factors
            risk_score = 0
            if age > 50:
                risk_score += 1
            if bmi > 30:
                risk_score += 1
            if bp_systolic > 140:
                risk_score += 1
            if cholesterol > 240:
                risk_score += 1
            if smoking:
                risk_score += 2
            if exercise_hours < 2:
                risk_score += 1
            
            risk_level = min(risk_score, 3)  # 0=Low, 1=Medium, 2=High, 3=Critical
            
            health_data.append((patient_id, age, bmi, bp_systolic, bp_diastolic,
                              cholesterol, smoking, exercise_hours, risk_level))
        
        cursor.executemany("""
            INSERT INTO health_risk 
            (patient_id, age, bmi, blood_pressure_systolic, blood_pressure_diastolic, cholesterol, smoking, exercise_hours, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, health_data)
        
        conn.commit()
        conn.close()
        logger.info("Generated comprehensive datasets: 3000 customers, 1825 stock records, 2000 health records")
    
    def train_churn_model(self):
        """Train customer churn prediction model"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM customer_churn", conn)
        conn.close()
        
        if df.empty:
            logger.warning("No churn data available")
            return None
        
        # Prepare features
        le_contract = LabelEncoder()
        le_payment = LabelEncoder()
        le_internet = LabelEncoder()
        
        df['contract_encoded'] = le_contract.fit_transform(df['contract_type'])
        df['payment_encoded'] = le_payment.fit_transform(df['payment_method'])
        df['internet_encoded'] = le_internet.fit_transform(df['internet_service'])
        
        features = ['tenure', 'monthly_charges', 'total_charges', 
                   'contract_encoded', 'payment_encoded', 'internet_encoded']
        
        X = df[features]
        y = df['churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store model and preprocessors
        self.models['churn'] = model
        self.scalers['churn'] = scaler
        self.encoders['churn'] = {
            'contract': le_contract,
            'payment': le_payment,
            'internet': le_internet
        }
        
        logger.info(f"Churn model trained with accuracy: {accuracy:.3f}")
        return accuracy
    
    def train_stock_model(self):
        """Train stock price prediction model"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM stock_prices ORDER BY symbol, date", conn)
        conn.close()
        
        if df.empty:
            logger.warning("No stock data available")
            return None
        
        # Create features for stock prediction
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date'])
        
        # Technical indicators
        df['price_change'] = df.groupby('symbol')['close_price'].pct_change()
        df['volume_ma'] = df.groupby('symbol')['volume'].rolling(window=5).mean().reset_index(0, drop=True)
        df['price_ma'] = df.groupby('symbol')['close_price'].rolling(window=5).mean().reset_index(0, drop=True)
        
        # Target: next day's price
        df['target'] = df.groupby('symbol')['close_price'].shift(-1)
        
        # Remove NaN values
        df = df.dropna()
        
        if df.empty:
            logger.warning("No valid stock data after preprocessing")
            return None
        
        features = ['open_price', 'high_price', 'low_price', 'volume', 'price_change', 'volume_ma', 'price_ma']
        
        X = df[features]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Store model and scaler
        self.models['stock'] = model
        self.scalers['stock'] = scaler
        
        logger.info(f"Stock model trained with RMSE: {rmse:.3f}")
        return rmse
    
    def train_health_model(self):
        """Train health risk prediction model"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM health_risk", conn)
        conn.close()
        
        if df.empty:
            logger.warning("No health data available")
            return None
        
        features = ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                   'cholesterol', 'smoking', 'exercise_hours']
        
        X = df[features]
        y = df['risk_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store model and scaler
        self.models['health'] = model
        self.scalers['health'] = scaler
        
        logger.info(f"Health model trained with accuracy: {accuracy:.3f}")
        return accuracy
    
    def run_complete_pipeline(self):
        """Run the complete data science pipeline"""
        logger.info("Starting complete data science pipeline...")
        
        # Generate data
        self.generate_sample_datasets()
        
        # Train all models
        churn_accuracy = self.train_churn_model()
        stock_rmse = self.train_stock_model()
        health_accuracy = self.train_health_model()
        
        # Summary
        results = {
            'churn_accuracy': churn_accuracy,
            'stock_rmse': stock_rmse,
            'health_accuracy': health_accuracy
        }
        
        logger.info("Data science pipeline completed successfully")
        return results

def main():
    """Main function to run the data science pipeline"""
    pipeline = DataSciencePipeline()
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "="*50)
    print("DATA SCIENCE PIPELINE RESULTS")
    print("="*50)
    
    if results['churn_accuracy']:
        print(f"Customer Churn Model Accuracy: {results['churn_accuracy']:.1%}")
    
    if results['stock_rmse']:
        print(f"Stock Price Model RMSE: ${results['stock_rmse']:.2f}")
    
    if results['health_accuracy']:
        print(f"Health Risk Model Accuracy: {results['health_accuracy']:.1%}")
    
    print("="*50)

if __name__ == "__main__":
    main()
