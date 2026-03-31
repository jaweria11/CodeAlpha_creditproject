# Credit Project - Credit Score Prediction Web App

A Django-based web application that predicts credit risk for users based on input financial data using a machine learning model. Users can fill a form with personal and financial details, and the app returns a prediction (e.g., "Good Credit" or "High Risk").

---



## Features

- Predicts credit score using a pre-trained ML model (`credit_model.pkl`)  
- Uses `scaler.pkl` for feature scaling  
- Simple web form interface built with Django templates  
- Clean project structure for easy understanding and maintenance  

---


---

## Technologies

- Python 3.13  
- Django 4.x  
- scikit-learn / joblib for ML model  
- HTML for frontend templates  

---

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/jaweria11/CodeAlpha_creditproject>
cd CREDIT_PROJECT


2.Create a virtual environment:
python -m venv venv

3.Activate the virtual environment:
Windows:
venv\Scripts\activate
4.Install dependencies:
pip install django joblib scikit-learn

5.Usage
Run Django server from the project root (where manage.py is):
python manage.py runserver