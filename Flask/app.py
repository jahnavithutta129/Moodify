from flask import Flask,render_template,redirect,request
import pickle
import numpy

cv=pickle.load(open('vectorizer.pkl','rb'))
clf=pickle.load(open('lr_clf.pkl','rb'))

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    pred_result = ""
    quote = ""
    
    if request.method == 'POST':
        if 'input_data' in request.form:
            input_data = request.form['input_data']
            try:
                # Process the input data and make predictions
                processed_input = cv.transform([input_data]).toarray()
                prediction = clf.predict(processed_input)
                
                # Determine the prediction result
                pred_result = "Stressed" if prediction == 1 else "Not Stressed"
                print(pred_result)
                
                # Set quote based on prediction result
                if pred_result == "Stressed":
                    quote = "Tough times never last, but tough people do."
                elif pred_result == "Not Stressed":
                    quote = "Happiness is not something ready-made. It comes from your own actions."
                    
            except Exception as e:
                # Handle prediction error
                print(f"Prediction error: {e}")
                pred_result = "Prediction Error"

    # Pass the prediction result and quote to the frontend
    return render_template('index.html', prediction=pred_result, quote=quote)

# if __name__ == "__main__":
#     app.run(debug=True)