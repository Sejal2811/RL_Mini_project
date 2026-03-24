# from flask import Flask, jsonify, render_template
# from rl_model import train_agent

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/run")
# def run_rl():

#     policy, path = train_agent()

#     return jsonify({
#         "policy": policy,
#         "path": path
#     })

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, jsonify, render_template
from rl_model import train_agent

app = Flask(__name__)

# 🔥 Run once when server starts
policy, path = train_agent()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run")
def run_rl():
    return jsonify({
        "policy": policy,
        "path": path
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)