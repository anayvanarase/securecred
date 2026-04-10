import json

try:
    nb = json.load(open('PerfectModel.ipynb', encoding='utf-8'))
    src_cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    src = '\n\n'.join(src_cells)

    if src.startswith('!pip'):
        src = src.split('\n', 1)[1]

    # disable optuna long runs and fix multiprocessing
    src = src.replace('n_trials=30', 'n_trials=1')
    src = src.replace('n_jobs=-1', 'n_jobs=1')
    src = src.replace('display(', 'print(')
    src = src.replace('test_transactions.csv', 'train_transactions.csv')
    src = src.replace('plt.show()', 'pass')

    # truncate at SHAP
    idx = src.find('print("🔍 Computing SHAP Values')
    if idx > 0:
        src = src[:idx]
        
    src += '''
import joblib
print("Saving...")
joblib.dump({'model': best_model, 'scaler': scaler, 'encoders': encoders, 'train_cols': list(X_train_df.columns)}, 'fraud_model.joblib')
print("DONE SAVING")
'''
    with open('model_clean.py', 'w', encoding='utf-8') as f:
        f.write(src)
        
    print("model_clean.py generated successfully.")
except Exception as e:
    print("Error:", e)
