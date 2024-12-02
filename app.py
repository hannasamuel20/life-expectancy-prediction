import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('./model/extraTreesRegressor.pkl')
scaler = joblib.load('./model/scaler.pkl')


# App title
st.title("Life Expectancy Prediction")
st.subheader("Sample Data")
st.write(
    pd.DataFrame(
        {
            "Country": ["Australia"],
            "Year": [2008],
            "Status": ["Developed"],
            "Life expectancy": [81.3],
            "Adult Mortality": [66.0],
            "Alcohol": [10.76],
            "percentage expenditure": [8547.292357],
            "Hepatitis B": [94.0],
            "Measles": [65],
            "BMI": [62.9],
            "under-five deaths": [1],
            "Polio": [92.0],
            "Total expenditure": [8.78],
            "Diphtheria": [92.0],
            "HIV/AIDS": [0.1],
            "GDP": [49664.6854],
            "Population": [212492.0],
            "thinness 1-19 years": [0.7],
            "Income composition of resources": [0.921],
            "Schooling": [19.1],
        }
    )
)

# # Input fields
st.write("Enter features:")

status = st.radio(
    "What's the status of the Country",
    ["Developing", "Developed"],
)


adult_mortality = st.number_input(label="Adult Mortality", min_value=0.0,
                                  max_value=1000.0,
                                  value=66.6)
st.caption("Death between age 15-60 years per 1000 population")

year = st.slider(
    label="Select a Year",
    min_value=1900,
    max_value=2100,
    value=2023,
    step=1
)


alcohol_consumption = st.slider(
    label="Alcohol Consumption",
    min_value=0.0,
    value=10.7,
    max_value=30.0,
    step=0.1
)
st.caption(
    "Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)")


percentage_expenditure = st.number_input(
    label="Expenditure",
    min_value=0.0,
    max_value=50000.0,
    value=8547.29,
)
st.caption(
    "Expenditure on health as a percentage of Gross Domestic Product per capita(%)")


reported_measles = st.number_input(
    label="Reported Measles", min_value=0, max_value=1000, value=65, step=1)
st.caption("Number of reported cases per 1000 population")


bmi = st.slider(
    label="BMI",
    min_value=0.0,
    max_value=100.0,
    value=62.0,
    step=0.1
)
st.caption(
    "Average Body Mass Index of entire population")


under_5_deaths = st.number_input(
    label="Under-Five Deaths", min_value=0, max_value=1000, value=1, step=1)
st.caption("Number of under-five deaths per 1000 population")


polio_coverage = st.slider(
    label="Polio Immunization Coverage",
    min_value=0.0,
    max_value=100.0,
    value=98.0,
    step=0.1
)
st.caption(
    "Polio (Pol3) immunization coverage among 1-year-olds (%)")


total_expenditure = st.slider(
    label="Total Government Health Expenditure",
    min_value=0.0,
    max_value=100.0,
    value=8.7,
    step=0.1
)
st.caption(
    "General government expenditure on health as a percentage of total government expenditure (%)")

tetanus_coverage = st.slider(
    label="Tetanus Immunization Coverage",
    min_value=0.0,
    max_value=100.0,
    value=92.0,
    step=0.1
)
st.caption(
    "Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)")


hiv_aids = st.number_input("HIV/AIDS", min_value=0.0,
                           max_value=1000.0, value=0.1)
st.caption("Deaths per 1000 live births HIV/AIDS (0-4 years)")

gdp = st.number_input(
    label="GDP",
    min_value=0.0,
    value=49664.6,
)
st.caption(
    "Gross Domestic Product per capita (in USD)")

population = st.number_input(
    label="Population",
    min_value=0.0,
    value=212492.0,
)
st.caption(
    "Population of the country")

thinness_10_to_19 = st.slider(
    label="Thinness 10 to 19",
    min_value=0.0,
    max_value=100.0,
    value=0.7,
    step=0.1
)
st.caption(
    "Prevalence of thinness among children and adolescents for Age 10 to 19 (%)")

income_comp = st.number_input(
    label="Income composition of resources",
    min_value=0.0,
    max_value=1.0,
    value=0.921,
    step=0.01,
)
st.caption(
    "Human Development Index in terms of income composition of resources (index ranging from 0 to 1)")

schooling = st.slider(
    label="School Years",
    min_value=0.0,
    max_value=50.0,
    value=20.1,
    step=0.1
)
st.caption(
    "Number of years of Schooling(years)")


if st.button("Predict"):
    status = 1 if status == "Developing" else 0
    numerical_input_data = np.array([year, adult_mortality, alcohol_consumption,
                                     percentage_expenditure, reported_measles, bmi,
                                     under_5_deaths, polio_coverage, total_expenditure,
                                     tetanus_coverage, hiv_aids, gdp,
                                     population, thinness_10_to_19, income_comp, schooling]).reshape(1, -1)
    scaled_numerical_input_data = scaler.transform(numerical_input_data)
    input_data = np.hstack(
        [scaled_numerical_input_data, np.array([status]).reshape(1, -1)])

    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
