from flask import Flask, request, render_template_string
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Training Data
X = np.array([
[6,148,72,35,0,33.6,0.627,50],
[1,85,66,29,0,26.6,0.351,31],
[8,183,64,0,0,23.3,0.672,32],
[1,89,66,23,94,28.1,0.167,21],
[0,137,40,35,168,43.1,2.288,33],
[5,116,74,0,0,25.6,0.201,30],
[3,78,50,32,88,31.0,0.248,26],
[10,115,0,0,0,35.3,0.134,29],
[2,197,70,45,543,30.5,0.158,53],
[8,125,96,0,0,0,0.232,54]
])

y = np.array([1,0,1,0,1,0,0,1,1,1])

model = LogisticRegression()
model.fit(X,y)

html = """

<!DOCTYPE html>
<html>
<head>

<title>AI Diabetes Dashboard</title>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

body{
background:linear-gradient(120deg,#4facfe,#00f2fe);
font-family:Segoe UI;
}

.container{
margin-top:50px;
max-width:700px;
}

.card{
border-radius:20px;
padding:30px;
box-shadow:0 10px 25px rgba(0,0,0,0.2);
}

.title{
text-align:center;
font-weight:bold;
margin-bottom:20px;
}

input{
margin-bottom:8px;
}

.progress{
height:20px;
margin-top:10px;
}

.tips{
margin-top:20px;
padding:10px;
background:#f5f5f5;
border-radius:10px;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h2 class="title">AI Diabetes Prediction System</h2>

<form method="post">

<input class="form-control" name="preg" placeholder="Pregnancies" required>

<input class="form-control" name="glu" placeholder="Glucose Level" required>

<input class="form-control" name="bp" placeholder="Blood Pressure" required>

<input class="form-control" name="skin" placeholder="Skin Thickness" required>

<input class="form-control" name="ins" placeholder="Insulin" required>

<input class="form-control" name="bmi" placeholder="BMI" required>

<input class="form-control" name="dpf" placeholder="Diabetes Pedigree Function" required>

<input class="form-control" name="age" placeholder="Age" required>

<button class="btn btn-primary w-100 mt-2">Predict</button>

</form>

{% if prediction %}

<h4 class="text-center mt-3">{{prediction}}</h4>

<div class="progress">

<div class="progress-bar bg-danger" style="width:{{prob}}%">
{{prob}}%

</div>

</div>

<canvas id="chart" height="120"></canvas>

<script>

var ctx=document.getElementById('chart');

new Chart(ctx,{
type:'doughnut',
data:{
labels:['Risk','Safe'],
datasets:[{
data:[{{prob}},100-{{prob}}],
backgroundColor:['#ff4d4d','#4caf50']
}]
}
});

</script>

<div class="tips">

<h5>Health Tips</h5>

<ul>

<li>Exercise 30 minutes daily</li>

<li>Avoid sugary foods</li>

<li>Maintain healthy BMI</li>

<li>Drink plenty of water</li>

<li>Monitor glucose regularly</li>

</ul>

</div>

{% endif %}

</div>

</div>

</body>

</html>

"""

@app.route('/',methods=['GET','POST'])

def home():

    prediction=""
    prob=0

    if request.method=="POST":

        data=[
        float(request.form['preg']),
        float(request.form['glu']),
        float(request.form['bp']),
        float(request.form['skin']),
        float(request.form['ins']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
        ]

        pred=model.predict([data])
        probability=model.predict_proba([data])[0][1]

        prob=round(probability*100,2)

        if pred[0]==1:
            prediction="⚠️ High Risk of Diabetes"
        else:
            prediction="✅ Low Risk (Healthy)"

    return render_template_string(html,
                                  prediction=prediction,
                                  prob=prob)

if __name__=="__main__":
    app.run(debug=True)
