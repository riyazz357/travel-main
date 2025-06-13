import subprocess
from flask import Flask

app = Flask(__name__)

@app.route("/open_chatbot")
def open_chatbot():
    # Run the chatbot.py script to open the Tkinter UI
    subprocess.Popen(["python", "chatbot.py"])
    return "Chatbot opened!"

if __name__ == "__main__":
    app.run(debug=True, port=5002)
