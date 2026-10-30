#### load model 
import pickle 
from flask import Flask
from flask import request
from flask import jsonify
model_file = 'model_C=0.001.bin'
with open (model_file,'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

   
@app.route('/predict', methods=['POST'])
def predict():

    customer= request.get_json()

    X_test = dv.transform([customer])

    y_pred = model.predict_proba(X_test)[0, 1]

    converted = y_pred >= 0.5

    result = {
        'conversion_probability': float(y_pred),
        'converted': bool(converted)
    }

    return jsonify(result)


if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

"""print('input', test_customers)
print('probability', y_pred )"""