import datetime
import pandas as pd
import numpy as np
import datetime

from sklearn.base import BaseEstimator, TransformerMixin


def create_submission_file(prediction, model_name, params):
    dict_string = ''

    for key in params.keys():
        dict_string += (key + '_' + str(params[key]) + '__')

    filename = str(datetime.datetime.now())[:-7] + model_name + dict_string[:-2] + '.csv'
    sub = pd.Series(submisstion, name='target')
    sub.to_csv(filename, index=False, header=True)


def impute_selected_variables(df, test, categ, quanti, dates):
    _df = df.copy()
    _test = test.copy() if test is not None else None

    replace = _df[categ].mode()
    replace_values = {k:v.iloc[0] for k,v in replace.items()}
    _df.fillna(replace_values, inplace=True)

    replace_quanti = _df[quanti].mean()
    _df.fillna(replace_quanti, inplace=True)

    _df[dates] = _df[dates].fillna(method='pad')

    if test is not None:

        _test.fillna(replace_values, inplace=True)
        _test.fillna(replace_quanti, inplace=True)
        _test[dates] = _df[dates].fillna(method='pad')

    return _df, _test

def impute_contract_variables(df):
    _df = df.copy()

    for var in categ_contract:
        try:
            _df[var] = _df[var].fillna('NAN')
        except ValueError as e:
            _df[var] = _df[var].cat.add_categories(['NAN'])
            _df[var] = _df[var].fillna('NAN')

    _df[quanti_contract] = _df[quanti_contract].fillna(-9999)
    _df[date_contract] = _df[date_contract].fillna(datetime.datetime(1970, 1, 1))
    return _df


def commentaire_bi(df):
    _df = df.copy()

    _df.COMMENTAIRE_BI = _df.COMMENTAIRE_BI.str.upper()
    COMMENTAIRE_BI_vc = _df.COMMENTAIRE_BI.value_counts()
    common_commentaire_bi = COMMENTAIRE_BI_vc[COMMENTAIRE_BI_vc > 100].index
    _df['COMMENTAIRE_BI_common'] = _df.COMMENTAIRE_BI.where(_df.COMMENTAIRE_BI.isin(common_commentaire_bi), "Rare")

    _df['nb_char_commentaire'] = [len(txt) for txt in _df.COMMENTAIRE_BI]
    _df['nb_mots_commentaire'] = [len(txt.split()) for txt in _df.COMMENTAIRE_BI]
    _df['has_number_commentaire'] = [any(char.isdigit() for char in txt) for txt in _df.COMMENTAIRE_BI]
    _df['is_empty_commentaire'] = [(txt == '.') for txt in _df.COMMENTAIRE_BI]
    _df.drop('COMMENTAIRE_BI', axis=1, inplace=True)

    return _df


def nature_code_split(df):
    _df = df.copy()
    nature_code_splitted = [nc.split('-') for nc in df.NATURE_CODE]
    nature_code_df = pd.DataFrame(nature_code_splitted, columns=['nc_1', 'nc_2', 'nc_3', 'nc_4', 'nc_5'])
    nature_code_df.fillna('-1', inplace=True)
    for nc_i in ['nc_1', 'nc_2', 'nc_3', 'nc_4', 'nc_5']:
        nature_code_df[nc_i] = nature_code_df[nc_i].astype('category')

    #_df.drop('NATURE_CODE', axis=1, inplace=True)
    _df = _df.merge(nature_code_df, left_index=True, right_index=True)
    return _df


def add_dates_features(data):
    data['age_installation'] = (data['CRE_DATE_GZL'] - data['INSTALL_DATE']).dt.days // 365
    data['mois_appel'] = data['CRE_DATE_GZL'].dt.month
    data['joursemaine_appel'] = data['CRE_DATE_GZL'].dt.isoweekday().astype('category')
    data['jour_appel'] = data['CRE_DATE_GZL'].dt.day
    data['mois_intervention'] = data['SCHEDULED_START_DATE'].dt.month
    data['joursemaine_intervention'] = data['SCHEDULED_START_DATE'].dt.isoweekday().astype('category')
    data['jour_intervention'] = data['SCHEDULED_START_DATE'].dt.day
    data['duree_avant_intervention'] = (data['SCHEDULED_START_DATE'] - data['CRE_DATE_GZL']).dt.days
    data['duree_prevue'] = (data['SCHEDULED_END_DATE'] - data['SCHEDULED_START_DATE']).dt.days
    data['temps_depuis_debut_contrat'] = (data['CRE_DATE_GZL'] - data['DATE_DEBUT']).dt.days
    data['temps_jusqua_fin_contrat'] = (data['CRE_DATE_GZL'] - data['DATE_FIN']).dt.days  #souvent nan ? (mettre 0)
    data['temps_depuis_maj_contrat'] = (data['CRE_DATE_GZL'] - data['UPD_DATE']).dt.days

    data.drop(['CRE_DATE_GZL', 'INSTALL_DATE', 'SCHEDULED_START_DATE', 'SCHEDULED_END_DATE', 'DATE_DEBUT', 'DATE_FIN', 'UPD_DATE'], axis=1, inplace=True)
    return data

def add_features_from_file(df, csv_file):
    features = pd.read_csv(csv_file)
    #features['canceled_pred'] = features['canceled_pred'].astype(bool)

    df = df.join(features['canceled_proba_pred'])
    return df

class CategoriesDroper(BaseEstimator, TransformerMixin):
    '''Drop categories which are not in the train set'''

    def __init__(self, columns):
        self.columns = columns
        self.categories_dict = dict()

    def fit(self, df, y=None):
        self.categories_dict = {column: df[column].cat.categories for column in self.columns}
        return self

    def transform(self, df):
        _df = df.copy()
        for column in self.columns:
            _df[column] = _df[column].cat.set_categories(self.categories_dict[column])
            try:
                _df[column] = _df[column].fillna('NAN')
            except ValueError as e:
                _df[column] = _df[column].cat.add_categories(['NAN'])
                _df[column] = _df[column].fillna('NAN')
        return _df

def add_categories_in_columns(df, columns, categories=['NAN']):
for var in columns:
    try:
        df.loc[:, var] = df[var].cat.add_categories(categories)
    except ValueError as e:
        next
return df


def randomly_alter_modalities_on_same_line(df, columns_categ, columns_quanti, rate, modality_categ='NAN', modality_quanti=-9999):
    _df = df.copy()
    
    _idx = np.random.choice(_df.index, size=int(_df.shape[0] * rate), replace=False)
    _df.loc[_idx, columns_categ] = modality_categ
    _df.loc[_idx, columns_quanti] = modality_quanti
    return _df

def randomly_alter_modalities(df, columns, rate, modality='NAN'):
    _df = df.copy()

    for var in columns:
        _idx = np.random.choice(_df.index, size=int(_df.shape[0] * rate), replace=False)
        _df.loc[_idx, var] = modality

    return _df
