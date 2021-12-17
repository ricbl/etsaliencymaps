import pandas as pd
import statsmodels.api as sm

csv_comparisons = './uniform_all_models_majority.csv'
results_csv = pd.DataFrame()
df = pd.read_csv(csv_comparisons)
for metric in ['sncc','sauc','ncc','auc']:
    df_ = df[df['metric'] == metric]
    for y_name in ['score_ag','score_interobserver']:
        print(metric)
        print(y_name)
        if y_name=='score_interobserver':
            df_ = df_[df_['model number']==1]
        y = df_[y_name]
        X = df_[['abnormal_majority', 'parenchymal_majority', 'pleural_majority', 'cardiomediastinal_majority']]
        model = sm.OLS(y,sm.tools.tools.add_constant(X))
        results = model.fit()
        coefs = results.params
        results_csv = results_csv.append({'metric':metric, 'y_name': y_name, 'coef parenchymal':coefs['parenchymal_majority'],
         'coef pleural':coefs['pleural_majority'], 'coef cardimediastinal':coefs['cardiomediastinal_majority'], 'se parenchymal':results.bse['parenchymal_majority'],
          'se pleural':results.bse['pleural_majority'], 'se cardimediastinal':results.bse['cardiomediastinal_majority']}, ignore_index=True)

results_csv.to_csv('./multiple_variable_linear_regression_coefficients.csv')