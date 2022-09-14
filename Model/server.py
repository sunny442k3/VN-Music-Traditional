from flask import Flask, jsonify
from generate import *
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    model_note = load_model("./checkpoints/model_v7.pt")
    gen_data = generate(model_note, [1])
    midi_object = token2midi(gen_data,"./midi_gen/music1.mid")

    return jsonify({"text": midi2raw_data(midi_object)})

if __name__ == "__main__":
    app.run("127.0.0.1", 5000, debug=True)