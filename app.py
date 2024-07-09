import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

st.set_page_config(page_title="CHARTOGRAPH")
st.header("Data Visualization")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    columns = data.columns.tolist()
    
    chart_type = st.selectbox(
        "Select the type of chart",
        ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot"]
    )


    if chart_type == "Line Chart":
        x_axis = st.selectbox("Select X-axis", columns)
        y_axis = st.selectbox("Select Y-axis", columns)
        fig = px.line(data, x=x_axis, y=y_axis)
        st.plotly_chart(fig)

    elif chart_type == "Bar Chart":
        x_axis = st.selectbox("Select X-axis", columns)
        y_axis = st.selectbox("Select Y-axis", columns)
        fig = plt.figure(figsize=(10, 4))
        sns.barplot(x=x_axis, y=y_axis, data=data)
        st.pyplot(fig)

    elif chart_type == "Scatter Plot":
        x_axis = st.selectbox("Select X-axis", columns)
        y_axis = st.selectbox("Select Y-axis", columns)
        fig = px.scatter(data, x=x_axis, y=y_axis)
        st.plotly_chart(fig)

    elif chart_type == "Histogram":  
        x_axis = st.selectbox("Select X-axis", columns)
        fig = plt.figure(figsize=(10, 4))
        sns.histplot(data[x_axis])
        st.pyplot(fig)

    elif chart_type == "Box Plot":
        x_axis = st.selectbox("Select X-axis", columns)
        y_axis = st.selectbox("Select Y-axis", columns)
        fig = plt.figure(figsize=(10, 4))
        sns.boxplot(x=data[x_axis], y=data[y_axis])
        st.pyplot(fig)

    st.header("Machine Learning Algorithm")

    feature_vars = st.multiselect("Select Feature Variables", columns)
    target_var = st.selectbox("Select Target Variable", columns)
    
    ml_algorithm = st.selectbox(
        "Select the Machine Learning Algorithm",
        ["Linear Regression", "Random Forest", "Gradient Boosting Regressor", "XGBoost"]
    )
    
    test_size = st.slider("Select Test Size for Train/Test Split", 0.1, 0.5, 0.2)
    
    if st.button("Run Model"):
        X = data[feature_vars]
        y = data[target_var]

        # Encode categorical data
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = X.select_dtypes(exclude=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        X = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if ml_algorithm == "Linear Regression":
            model = LinearRegression()
        elif ml_algorithm == "Random Forest":
            model = RandomForestRegressor()
        elif ml_algorithm == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor()
        elif ml_algorithm == "XGBoost":
            model = xgb.XGBRegressor()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.3f}")
