from flask import Flask, render_template, request
from waitress import serve
from datetime import datetime
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

def add_header(response):
    response.headers['ngrok-skip-browser-warning'] = '1'
    return response

@app.after_request
def after_request(response):
    return add_header(response)

def categorize_part_of_day(seconds):
    hour = (seconds // 3600) % 24
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

# Load the saved label encoders
with open('fraud_detection_pipeline.pkl', 'rb') as file:
    saved_objects = pickle.load(file)

model = saved_objects['model']
label_encoders = saved_objects['label_encoders']


def preprocess_input(form_data):
    # Extract domain from email
    email_domain = form_data['P_emaildomain'].split('@')[-1]

    # Create a new DataFrame for the single input sample
    data = pd.DataFrame([{
        'P_emaildomain': email_domain,
        'ProductCD': form_data['ProductCD'],
        'card4': form_data['card4'],
        'card6': form_data['card6'],
        'DeviceType': form_data['DeviceType'],
        'TransactionAmt': form_data['TransactionAmt'],
    }])

    # Categorize part of the day
    current_time = datetime.now()
    part_of_day = categorize_part_of_day(current_time.timestamp())
    data['part_of_day'] = part_of_day

    bins = [0, 5000, 10000, 15000, 20000, float('inf')] # Bins
    bin_labels = ['0-5000','5000-10000','10000-15000','15000-20000','20000 and above']
    data['TransactionAmt_Bins'] = pd.cut(data['TransactionAmt'].values, bins=bins, labels=bin_labels)

    # Apply label encoding
    for col, encoder in label_encoders.items():
        if col in data:
            try:
                data[col] = encoder.transform(data[col])
            except ValueError:
                if col == 'ProductCD':
                    data[col] = encoder.transform(['W']) 

        # Convert categorical columns to numeric
    categorical_cols = data.select_dtypes(include=['category', 'object']).columns
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    return data


product_mapping = {
    'phones_tablets': 'W',
    'laptop_desktops': 'H',
    'tv_home_theatre': 'C',
    'printers_scanners': 'S',
    'cameras': 'R'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(request.form)

        # Extract email domain and map product
        email_domain = request.form['email'].split('@')[-1]
        product_cd = product_mapping.get(request.form['product'], 'missing')

        # Create dictionary for prediction
        form_data = {
            'P_emaildomain': email_domain,
            'ProductCD': product_cd,
            'card4': request.form['card_type'],
            'DeviceType': request.form['DeviceType'],
            'card6': request.form['card_options'],
            'TransactionAmt': float(request.form['amount'])
        }

        preprocessed_data = preprocess_input(form_data)
        preprocessed_data.drop(columns=['TransactionAmt'], inplace=True)

        # Reorder columns as expected by the model
        preprocessed_data = preprocessed_data[['P_emaildomain', 'ProductCD', 'DeviceType', 'card6', 'part_of_day', 'card4', 'TransactionAmt_Bins']]

        print(preprocessed_data)

        # Make prediction
        prediction = model.predict(preprocessed_data)

        # Prepare the message based on the prediction
        if prediction[0] == 1:  # Assuming 1 indicates fraud
            result_message = "Your transaction is being reviewed. Someone will reach out as soon as possible."
        else:
            result_message = "You can proceed with the next steps."

        return render_template('index.html', result_message=result_message)

    # GET request, render template without results
    return render_template('index.html', result_message=None)



if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)