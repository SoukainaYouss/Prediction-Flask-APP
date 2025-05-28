import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # exemple de modèle
from flask import Flask, request, render_template, jsonify
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
app = Flask(__name__)

# --- Liste des colonnes one-hot et colonnes numériques attendues ---

features = [
    # Colonnes numériques exemples (à adapter)
    # "outstanding_amount_in_currency",
    # "outstanding_rate",
    # "weighting",
 'index',
 'customer_code',
 'internal_contract_number',
 'outstanding_amount_in_currency',
 'outstanding_amount_in_local_currency',
 'total_customer_outstanding_amount',
 'total_customer_provision_base',
 'outstanding_rate',
 'weighting',
 'riskweightedexposureamounts',
 'internal_segment',
 'net_outstanding_in_local_currency',
 'net_outstanding_not_covered',
 'net_outstanding_s_post_cover',
 'net_outstanding_post_cover',
 'net_outstanding_post_cover_ccf',
 'is_provisionable',
 'provision_rate',
 'is_guarantable',
 'total_customer_guarantee_base',
 'exposure_concentration_ratio',
 'provision_coverage_ratio',
 'coverage_effectiveness',
 'high_risk_weighting',
 'weighting_squared',
 'weighting_log',
 'provision_risk_score',
 'guarantee_effectiveness',

 'sub_cat_Autre actif',
 'sub_cat_Grandes entreprises',
 'sub_cat_Particuliers',
 'sub_cat_Petite ou Moyenne entreprise',
 'sub_cat_Très petite entreprise (TPE)',

 'sub_port_Autre actifs - Caisse et valeur',
 'sub_port_Autre actifs - Divers autre actifs',
 'sub_port_Grandes entreprises',
 'sub_port_Particuliers',
 'sub_port_Petite ou moyenne enreprise',
 'sub_port_Très petite entreprise (TPE)',

 'balance_B',
 'balance_H',

 'expo_1',
 'expo_2b',

 'out_Low',
 'out_Medium',
 'out_High',
 'out_Very_High',

 'tot_Low',
 'tot_Medium',
 'tot_High',
 'tot_Very_High']

# --- Valeurs possibles pour encodage ---
sub_category_options = [
    "Très petite entreprise (TPE)",
    "Particuliers",
    "Petite ou Moyenne entreprise",
    "Autre actif",
    "Grandes entreprises"
]

sub_portfolio_options = [
    "Autre actifs - Caisse et valeur",
    "Autre actifs - Divers autre actifs",
    "Grandes entreprises",
    "Particuliers",
    "Petite ou moyenne enreprise",
    "Très petite entreprise (TPE)"
]

total_customer_outstanding_amount_quartile_options = [
    "Low",
    "Medium",
    "High",
    "Very_High"
]

outstanding_amount_in_local_currency_quartile_options = [
    "Low",
    "Medium",
    "High",
    "Very_High"
]

balance_off_sheet_options = ["B", "H"]

exposition_type_options = ["1", "2b"]

# --- Modèle fictif (à remplacer par ton vrai modèle chargé) ---
MODEL_PATH = 'best_model2.pkl'
model = pickle.load(open(MODEL_PATH, 'rb'))
print("Loaded model type:", type(model))


def encode_one_hot(data, field_name, options, prefix):
    for opt in options:
        col_name = f"{prefix}_{opt}"
        data[col_name] = 0
    if field_name in data:
        val = data[field_name]
        col_name = f"{prefix}_{val}"
        if col_name in data:
            data[col_name] = 1
        del data[field_name]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # First try to get JSON, fallback to form data
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()

        # Now continue with your encoding and prediction logic
        encode_one_hot(data, "sub_category_description", sub_category_options, "sub_cat")
        encode_one_hot(data, "sub_portfolio_description", sub_portfolio_options, "sub_port")
        encode_one_hot(data, "total_customer_outstanding_amount_quartile", total_customer_outstanding_amount_quartile_options, "tot")
        encode_one_hot(data, "outstanding_amount_in_local_currency_quartile", outstanding_amount_in_local_currency_quartile_options, "out")
        encode_one_hot(data, "balance_off_sheet", balance_off_sheet_options, "balance")
        encode_one_hot(data, "exposition_type", exposition_type_options, "expo")

        # Fill missing features with 0
        for col in model.feature_names_in_:
            if col not in data:
                data[col] = 0

        df = pd.DataFrame([data], columns=model.feature_names_in_)

        prediction = int(model.predict(df)[0])
        texte = "Risque élevé" if prediction == 1 else "Risque faible"

        return render_template('index.html',
                               prediction_text=f"Résultat : {texte}",
                               form_data=data)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Erreur : {str(e)}",
                               form_data=request.form.to_dict())



if __name__ == '__main__':
    app.run(debug=True)
