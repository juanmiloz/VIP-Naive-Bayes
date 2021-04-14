from flask import Flask, render_template #this has changed
from src.visualization import KNNVisualizer

app = Flask(__name__)
visualizer = KNNVisualizer()


@app.route('/data', methods=["POST", "GET"])
def show_data():
    pass


if __name__ == '__main__':
    app.run()