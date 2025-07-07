#!/usr/bin/env python3
"""
Main Platform for Johns Hopkins Data Science Capstone
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

def main():
    """Main application"""
    st.title("📊 Johns Hopkins Data Science Capstone")
    st.write("Platform is running successfully!")
    
    # Simple demo data
    data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=30),
        'Value': np.random.randn(30).cumsum()
    })
    
    st.line_chart(data.set_index('Date'))
    st.success("✅ Platform operational!")

if __name__ == "__main__":
    main()
