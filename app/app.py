import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import streamlit as st

lr_model = joblib.load("../models/lr_model.pkl")
dt_model = joblib.load("../models/dt_model.pkl")
knn_model = joblib.load("../models/knn_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

st.title("❤️ CardioPredict")