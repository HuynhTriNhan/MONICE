"""
This module containts helper functions to load data and get meta deta.
"""
import os
import pickle
import shutil
import zipfile
from urllib.request import urlretrieve
import requests

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from scipy.io import arff
import torch

import dice_ml_x

dataset_links = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "compas": "https://api.openml.org/data/download/22111929/dataset",
    "german-credit-risk": "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
    "lending-club": "https://www.openintro.org/data/csv/loans_full_schema.csv"
}


def preprocess_compas_dataset(df: pd.DataFrame) -> pd.DataFrame:
    age_cat_columns = ['age_cat_25-45', 'age_cat_Greaterthan45', 'age_cat_Lessthan25']
    df['age_cat'] = df[age_cat_columns].idxmax(axis=1).str.replace('age_cat_', '')
    race_columns = ['race_African-American', 'race_Caucasian']
    df['race'] = df[race_columns].idxmax(axis=1).str.replace('race_', '')
    charge_degree_columns = ['c_charge_degree_F', 'c_charge_degree_M']
    df['c_charge_degree'] = df[charge_degree_columns].idxmax(axis=1).str.replace('c_charge_degree_', '')
    df['sex'] = df["sex"].map({0: "Female", 1: "Male"})
    df = df.drop(columns=age_cat_columns + race_columns + charge_degree_columns)
    return df


def load_compas_dataset() -> pd.DataFrame:
    outdirname = "compas"
    arff_file_name = f"{outdirname}.arff"
    if not os.path.isfile(arff_file_name):
        urlretrieve(dataset_links["compas"], arff_file_name)
    arff_data = arff.loadarff(arff_file_name)
    df = pd.DataFrame(arff_data[0])
    byte_string_cols = [col for col in df.columns if df[col].dtype == "object"]
    df[byte_string_cols] = df[byte_string_cols].applymap(lambda x: int(x.decode("utf-8")))
    df = preprocess_compas_dataset(df)
    cols_to_get = ["age", "sex", "race", "priors_count", "c_charge_degree", "twoyearrecid"]
    drop_cols = df.columns.difference(cols_to_get)
    df.drop(columns=drop_cols, inplace=True)
    return df


'''def load_compas_dataset() -> pd.DataFrame:
    """Download (if needed), decode and tidy the COMPAS two-year-recidivism data."""
    outname = "compas.arff"

    # 1 ─ Download once
    if not os.path.isfile(outname):
        urlretrieve(dataset_links["compas"], outname)

    # 2 ─ Read the ARFF file
    arff_data = arff.loadarff(outname)
    df = pd.DataFrame(arff_data[0])

    # 3 ─ Make *column names* plain strings (some are bytes)
    decoded_cols = [
        c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else c
        for c in df.columns
    ]
    df.columns = decoded_cols

    #    Deduplicate any names that collide after decoding (foo, foo.1, foo.2 …)
    if df.columns.has_duplicates:
        df.columns = (
            pd.io.parsers.ParserBase({"names": df.columns})
            ._maybe_dedup_names(df.columns)
        )

    # 4 ─ Cell-level decoding, one column at a time
    def safe_decode(val):
        if isinstance(val, (bytes, bytearray)):
            try:
                return val.decode("utf-8")
            except UnicodeDecodeError:
                return pd.NA               # bad byte sequence → missing
        if val is None or pd.isna(val):
            return pd.NA
        return val

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(safe_decode)

    # 5 ─ Cast numeric columns
    numeric_cols = ["age", "priors_count", "twoyearrecid"]
    df[numeric_cols] = (
        df[numeric_cols]
        .apply(lambda s: pd.to_numeric(s, errors="coerce"))
        .astype("Int64")                 # keeps NA as <NA>
    )

    # 6 ─ Your existing custom clean-up
    df = preprocess_compas_dataset(df)
    keep = ["age", "sex", "race", "priors_count", "c_charge_degree", "twoyearrecid"]
    return df[keep]'''


def _preprocess_german_data(data: np.array) -> pd.DataFrame:
    data_split = [row.split() for row in data]
    df = pd.DataFrame(data_split)
    column_names = ['status_of_existing_checking_account',
                    'duration_in_month',
                    'credit_history',
                    'purpose',
                    'credit_amount',
                    'savings_account_bonds',
                    'present_employment_since',
                    'installment_rate_in_percentage_of_disposable_income',
                    'personal_status_and_sex',
                    'other_debtors_guarantors',
                    'present_residence_since',
                    'property',
                    'age_in_years',
                    'other_installment_plans',
                    'housing',
                    'number_of_existing_credits_at_this_bank',
                    'job',
                    'number_of_people_being_liable_to_provide_maintenance_for',
                    'telephone',
                    'foreign_worker',
                    'credit_risk'
    ]
    df.columns = column_names
    categorical_value_mapping = {
       'status_of_existing_checking_account': {
            'A11': '... < 0 DM',
            'A12': '0 <= ... < 200 DM',
            'A13': '... >= 200 DM / salary assignments for at least 1 year',
            'A14': 'no checking account'
        },
        'credit_history':{
            'A30': 'no credits taken/ all credits paid back duly',
            'A31': 'all credits at this bank paid back duly',
            'A32': 'existing credits paid back duly till now',
            'A33': 'delay in paying off in the past',
            'A34': 'critical account/ other credits existing (not at this bank)'
        },
        'purpose': {
            'A40': 'car (new)',
            'A41': 'car (used)',
            'A42': 'furniture/equipment',
            'A43': 'radio/television',
            'A44': 'domestic appliances',
            'A45': 'repairs',
            'A46': 'education',
            'A47': '(vacation - does not exist?)',
            'A48': 'retraining',
            'A49': 'business',
            'A410': 'others'
        },
        'savings _account_bonds': {
            'A61': '... < 100 DM',
            'A62': '100 <= ... < 500 DM',
            'A63': '500 <= ... < 1000 DM',
            'A64': '.. >= 1000 DM',
            'A65': 'unknown/ no savings account'
        },
        'present_employment_since': {
            'A71': 'unemployed',
            'A72': '... < 1 year',
            'A73': '1 <= ... < 4 years',
            'A74': '4 <= ... < 7 years',
            'A75': '.. >= 7 years'
        },
        'personal_status_and_sex': {
            'A91': 'male : divorced/separated',
            'A92': 'female : divorced/separated/married',
            'A93': 'male : single',
            'A94': 'male : married/widowed',
            'A95': 'female : single'
        },
        'other_debtors_guarantors': {
            'A101': 'none',
            'A102': 'co-applicant',
            'A103': 'guarantor'
        },
        'property': {
            'A121': 'real estate',
            'A122': 'if not A121 : building society savings agreement/ life insurance',
            'A123': 'if not A121/A122 : car or other, not in attribute 6',
            'A124': 'unknown / no property'
        },
        'other_installment_plans': {
            'A141': 'bank',
            'A142': 'stores',
            'A143': 'none'
        },
        'housing': {
            'A151': 'rent',
            'A152': 'own',
            'A153': 'for free'
        },
        'job': {
            'A171': 'unemployed/ unskilled - non-resident',
            'A172': 'unskilled - resident',
            'A173': 'skilled employee / official',
            'A174': 'management/ self-employed/ highly qualified employee/ officer'
        },
        'telephone': {
            'A191': 'none',
            'A192': 'yes, registered under the customers name'
        },
        'foreign_worker': {
            'A201': 'yes',
            'A202': 'no'
        },
        'credit_risk': {
            '1': 0,
           '2': 1
        } 
    }
    for col, mapping in categorical_value_mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    # duration_in_month
    # credit_amount
    # installment_rate_in_percentage_of_disposable_income
    # present_residence_since
    # age_in_years
    # number_of_existing_credits_at_this_bank
    # number_of_people_being_liable_to_provide_maintenance_for
    number_cols = ['duration_in_month', 'credit_amount', 'installment_rate_in_percentage_of_disposable_income',
                   'present_residence_since', 'age_in_years', 'number_of_existing_credits_at_this_bank',
                   'number_of_people_being_liable_to_provide_maintenance_for']
    df[number_cols] = df[number_cols].astype(int)
    return df

def load_german_credit_dataset(model_type: str=None) -> pd.DataFrame:
    outdirname = "german_credit"
    german_credit_file_name = f"{outdirname}.zip"
    if not os.path.isfile(german_credit_file_name):
        urlretrieve(dataset_links['german-credit-risk'], german_credit_file_name)
    with zipfile.ZipFile(german_credit_file_name) as unzip:
        unzip.extractall(outdirname)

    raw_data = np.genfromtxt(f"{outdirname}/german.data", delimiter=", ",
                             dtype=str, invalid_raise=False)
    df = _preprocess_german_data(raw_data)
    return df

def load_lending_club_dataset() -> pd.DataFrame:
    """
    As described in the DiCE paper by the authors.
    """
    df = pd.read_csv("lending_club_dataset/loan.csv", low_memory=False)
    import math
    def parse_year(year_as_str):
        year = int(year_as_str)
        if year <= 99:
            return 1900 + year if year > 50 else 2000 + year
        return 2000 + year
    # raw features
    new_df = pd.DataFrame()
    new_df['employment_years'] = df['emp_length']
    new_df['num_open_credit_acc'] = df['open_acc']
    new_df['annual_income'] = df['annual_inc']
    new_df['loan_grade'] = df['grade']

    # credit history
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    lending_months = df['issue_d'].apply(lambda x: x.split('-')[0]).__deepcopy__().copy()
    issue_month = pd.DataFrame(pd.Index(months).get_indexer(lending_months))
    issue_year = pd.DataFrame(df['issue_d'].apply(lambda x: parse_year(x.split('-')[1])).__deepcopy__().copy())
    issue_yearmonth = pd.DataFrame(issue_year.values * 100 + issue_month.values)
    adjusted_last_ym = issue_yearmonth[0].apply(lambda x: (float(x) / 100.0) * 100 + ((float(x) - math.floor(float(x) / 100.0) * 100) - 1) / 12 * 100) / 100
    earliest_credit_line_year = pd.DataFrame(df['earliest_cr_line'].apply(lambda x: parse_year(x.split('-')[1])).__deepcopy__().copy())
    earliest_credit_line_months = df['earliest_cr_line'].apply(lambda x: x.split('-')[0]).__deepcopy__().copy()
    earliest_credit_line_months = pd.DataFrame(pd.Index(months).get_indexer(earliest_credit_line_months))
    adjusted_credit_line_ym = pd.DataFrame(earliest_credit_line_year.values * 100 + \
                            earliest_credit_line_months.apply(
                                lambda x: x - 1
                            ) / 12 * 100) / 100
    adjusted_credit_line_ym = adjusted_credit_line_ym.apply(lambda x: round(x, 2))
    adjusted_last_ym = adjusted_last_ym.apply(lambda x: round(x, 2)).to_frame()
    credit_ym = (adjusted_last_ym - adjusted_credit_line_ym).apply(lambda x: round(x, 1))
    new_df['credit_history'] = credit_ym.values

    # purpose
    conditions = [
        df['purpose'].isin(['credit_card', 'debt_consolidation']),  # Condition for "debt"
        df['purpose'].isin(['car', 'major_purchase', 'vacation', 'wedding', 'medical', 'other']),  # Condition for "purchase"
        df['purpose'].isin(['house', 'home_improvement', 'moving', 'renewable_energy'])  # Another condition for "purchase"
    ]

    outputs = ['debt', 'purchase', 'purchase']

    new_df['purpose'] = np.select(conditions, outputs, default=df['purpose'])

    # home ownership

    new_df['home'] = np.where(
        df['home_ownership'].isin(['ANY', 'NONE']), 
        'OTHER', 
        df['home_ownership']
    )

    # state

    new_df['addr_state'] = df['addr_state']
    new_df.replace('n/a', np.nan,inplace=True)
    new_df['employment_years'].fillna(value=0,inplace=True)
    new_df['employment_years'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    new_df['employment_years'] = new_df['employment_years'].astype(int)

    # target column (loan_status) 0 if never paid or not paid yet 1 if paid

    ls_conditions = [
        df['loan_status'].isin(['Charged Off', 'Current']),
        df['loan_status'] == 'Fully Paid'
    ]
    ls_outputs = [0, 1]

    new_df['loan_status'] = np.select(ls_conditions, ls_outputs, default=df['loan_status'])
    new_df['loan_status'] = new_df['loan_status'].astype(int)

    # Remove rows with negative credit history
    neg_credi_his_idx = new_df.loc[new_df['credit_history'] < 0].index
    new_df.drop(neg_credi_his_idx, inplace=True)
    return new_df

def get_compas_data_info() -> dict:
    return {
        
    }


def dummy_function():
    pass


def load_adult_income_dataset(only_train=False):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares
       the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :return adult_data: returns preprocessed adult income dataset.
    """
    # Download the adult dataset from https://archive.ics.uci.edu/static/public/2/adult.zip as a zip folder
    outdirname = 'adult'
    zipfilename = outdirname + '.zip'
    if not os.path.isfile(zipfilename):
        urlretrieve('https://archive.ics.uci.edu/static/public/2/adult.zip', zipfilename)
    with zipfile.ZipFile(zipfilename, 'r') as unzip:
        unzip.extractall(outdirname)

    raw_data = np.genfromtxt(outdirname + '/adult.data',
                             delimiter=', ', dtype=str, invalid_raise=False)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']

    adult_data = pd.DataFrame(raw_data, columns=column_names)

    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                     'Local-gov': 'Government'}})
    adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace(
        {
            'occupation': {
                'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                'Handlers-cleaners': 'Blue-Collar',
                'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                'Priv-house-serv': 'Service',
                'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                'Tech-support': 'Service',
                'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
            }
        }
    )

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                        'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                              'Amer-Indian-Eskimo': 'Other'}})

    adult_data = adult_data[['age', 'workclass', 'education', 'marital-status', 'occupation',
                             'race', 'gender', 'hours-per-week', 'income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                   '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                   '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                   '1st-4th': 'School', 'Preschool': 'School'}})

    adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    if only_train:
        train, _ = train_test_split(adult_data, test_size=0.2, random_state=17)
        adult_data = train.reset_index(drop=True)

    # Remove the downloaded dataset
    if os.path.isdir(outdirname):
        entire_path = os.path.abspath(outdirname)
        shutil.rmtree(entire_path)

    return adult_data


def save_adult_income_model(modelpath, test_fraction=0.2, random_state=0):
    dataset = load_adult_income_dataset()
    target = dataset["income"]
    train_dataset, x, y_train, y = train_test_split(dataset,
                                                    target,
                                                    test_size=test_fraction,
                                                    random_state=random_state,
                                                    stratify=target)
    x_train = train_dataset.drop('income', axis=1)
    numerical = ["age", "hours_per_week"]
    categorical = x_train.columns.difference(numerical)

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical)])

    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier', RandomForestClassifier())])
    model = clf.fit(x_train, y_train)
    pickle.dump(model, open(modelpath, 'wb'))


def load_custom_testing_dataset():
    data = [['a', 10, 0], ['b', 10000, 0], ['c', 14, 0], ['a', 88, 0], ['c', 14, 0]]
    return pd.DataFrame(data, columns=['Categorical', 'Numerical', 'Outcome'])


def load_min_max_equal_dataset():
    data = [['a', 10, 0], ['b', 10, 0], ['c', 10, 0], ['a', 10, 0], ['c', 10, 0]]
    return pd.DataFrame(data, columns=['Categorical', 'Numerical', 'Outcome'])


def load_outcome_not_last_column_dataset():
    data = [['a', 0, 10], ['a', 0, 10000], ['a', 0, 14], ['a', 0, 10], ['a', 0, 10]]
    return pd.DataFrame(data, columns=['Categorical', 'Outcome', 'Numerical'])


def load_custom_testing_dataset_binary():
    data = [
        ['a', 1, 0],
        ['b', 5, 1],
        ['c', 2, 0],
        ['a', 3, 0],
        ['c', 4, 1],
        ['c', 10, 0],
        ['a', 7, 0],
        ['c', 8, 1],
        ['b', 10, 1],
    ]
    return pd.DataFrame(data, columns=['Categorical', 'Numerical', 'Outcome'])


def load_custom_testing_dataset_binary_str():
    data = [
        ["a", 1, "zero"],
        ["b", 5, "one"],
        ["c", 2, "zero"],
        ["a", 3, "one"],
        ["c", 4, "one"],
    ]
    return pd.DataFrame(data, columns=["Categorical", "Numerical", "Outcome"])


def load_custom_testing_dataset_multiclass():
    data = [['a', 10, 1], ['b', 20, 2], ['c', 14, 1], ['a', 23, 2], ['c', 7, 0]]
    return pd.DataFrame(data, columns=['Categorical', 'Numerical', 'Outcome'])


def load_custom_testing_dataset_multiclass_str():
    data = [
        ["a", 1, "zero"],
        ["b", 5, "one"],
        ["c", 2, "two"],
        ["a", 3, "one"],
        ["c", 4, "zero"],
    ]
    return pd.DataFrame(data, columns=["Categorical", "Numerical", "Outcome"])


def load_custom_testing_dataset_regression():
    data = [['a', 10, 1], ['b', 21, 2.1], ['c', 14, 1.4], ['a', 23, 2.3], ['c', 7, 0.7]]
    return pd.DataFrame(data, columns=['Categorical', 'Numerical', 'Outcome'])


def get_adult_income_modelpath(backend='TF1'):
    pkg_path = dice_ml_x.__path__[0]
    model_ext = '.h5' if 'TF' in backend else ('.pth' if backend == 'PYT' else '.pkl')
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'adult'+model_ext)
    return modelpath


def get_custom_dataset_modelpath_pipeline():
    pkg_path = dice_ml_x.__path__[0]
    model_ext = '.sav'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'custom'+model_ext)
    return modelpath


def get_custom_vars_dataset_modelpath_pipeline():
    pkg_path = dice_ml_x.__path__[0]
    model_ext = '.sav'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'custom_vars'+model_ext)
    return modelpath


def get_custom_dataset_modelpath_pipeline_binary():
    pkg_path = dice_ml_x.__path__[0]
    model_ext = '.sav'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'custom_binary'+model_ext)
    return modelpath


def get_custom_dataset_modelpath_pipeline_multiclass():
    pkg_path = dice_ml_x.__path__[0]
    model_ext = '.sav'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'custom_multiclass'+model_ext)
    return modelpath


def get_custom_dataset_modelpath_pipeline_regression():
    pkg_path = dice_ml_x.__path__[0]
    model_ext = '.sav'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'custom_regression'+model_ext)
    return modelpath


def get_adult_data_info():
    feature_description = {
        'age': 'age',
        'workclass': 'type of industry (Government, Other/Unknown, Private, Self-Employed)',
        'education': 'education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)',
        'marital_status': 'marital status (Divorced, Married, Separated, Single, Widowed)',
        'occupation': 'occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)',
        'race': 'white or other race?',
        'gender': 'male or female?',
        'hours_per_week': 'total work hours per week',
        'income': '0 (<=50K) vs 1 (>50K)'}
    return feature_description


def get_base_gen_cf_initialization(data_interface, encoded_size, cont_minx, cont_maxx, margin, validity_reg, epochs,
                                   wm1, wm2, wm3, learning_rate):
    # Dice Imports - TODO: keep this method for VAE as a spearate module or move it to feasible_base_vae.py.
    #                      Check dependencies.
    from torch import optim

    from dice_ml_x.utils.sample_architecture.vae_model import CF_VAE

    # Dataset for training Variational Encoder Decoder model for CF Generation
    df = data_interface.normalize_data(data_interface.one_hot_encoded_data)
    encoded_data = df[data_interface.ohe_encoded_feature_names + [data_interface.outcome_name]]
    dataset = encoded_data.to_numpy()
    print('Dataset Shape:',  encoded_data.shape)
    print('Datasets Columns:', encoded_data.columns)

    # Normalise_Weights
    normalise_weights = {}
    for idx in range(len(cont_minx)):
        _max = cont_maxx[idx]
        _min = cont_minx[idx]
        normalise_weights[idx] = [_min, _max]

    # Train, Val, Test Splits
    np.random.shuffle(dataset)
    test_fraction = 0.2
    # TODO: create an input parameter for data interface
    test_size = int(test_fraction*len(data_interface.data_df))
    vae_test_dataset = dataset[:test_size]
    dataset = dataset[test_size:]
    vae_val_dataset = dataset[:test_size]
    vae_train_dataset = dataset[test_size:]

    # BaseGenCF Model
    cf_vae = CF_VAE(data_interface, encoded_size)

    # Optimizer
    cf_vae_optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean.parameters()), 'weight_decay': wm1},
        {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var.parameters()), 'weight_decay': wm2},
        {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean.parameters()), 'weight_decay': wm3},
        ], lr=learning_rate
    )

    # Check: If base_obj was passsed via reference and it mutable; might not need to have a return value at all
    return vae_train_dataset, vae_val_dataset, vae_test_dataset, normalise_weights, cf_vae, cf_vae_optimizer


def ohe_min_max_transformation(data, data_interface):
    """the data is one-hot-encoded and min-max normalized and fed to the ML model"""
    return data_interface.get_ohe_min_max_normalized_data(data)


def inverse_ohe_min_max_transformation(data, data_interface):
    return data_interface.get_inverse_ohe_min_max_normalized_data(data)


class DataTransfomer:
    """A class to transform data based on user-defined function to get predicted outcomes.
       This class calls FunctionTransformer of scikit-learn internally
       (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)."""

    def __init__(self, func=None, kw_args=None):
        self.func = func
        self.kw_args = kw_args

    def feed_data_params(self, data_interface):
        if self.kw_args is not None:
            self.kw_args['data_interface'] = data_interface
        else:
            self.kw_args = {'data_interface': data_interface}

    def initialize_transform_func(self):
        if self.func == 'ohe-min-max':
            self.data_transformer = FunctionTransformer(
                    func=ohe_min_max_transformation,
                    inverse_func=inverse_ohe_min_max_transformation,
                    check_inverse=False,
                    validate=False,
                    kw_args=self.kw_args,
                    inv_kw_args=self.kw_args)
        elif self.func is None:
            # identity transformation
            # add more ready-to-use transformers (such as label-encoding) in elif loops.
            self.data_transformer = FunctionTransformer(func=self.func, kw_args=None, validate=False)
        else:
            # add more ready-to-use transformers (such as label-encoding) in elif loops.
            self.data_transformer = FunctionTransformer(func=self.func, kw_args=self.kw_args, validate=False)

    def transform(self, data):
        return self.data_transformer.transform(data)  # should return a numpy array

    def inverse_transform(self, data):
        return self.data_transformer.inverse_transform(data)  # should return a numpy array
    
