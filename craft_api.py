from os import getenv
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from predict_box import CraftDetection


load_dotenv()


CRAFT_PORT = int(getenv('CRAFT_PORT'))

model = CraftDetection()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["CACHE_TYPE"] = "null"


def create_error_response(error=None):
    return jsonify({
        'error': 'Error: ' + str(error)
    })


@app.route("/query_box", methods=['POST'])
def query_box():
    try:
        data = request.get_data()
        img = Image.open(BytesIO(data)).convert('RGB')
    except Exception as ex:
        print(ex)
        return create_error_response(ex)

    img = np.array(img)
    boxes, _, _ = model.text_detect(img)
    h, w, _ = img.shape

    boxes_dict = []
    for box in boxes:
        [c1, c2, c3, c4] = box
        box_dict = [
            {'x': round(c1[0]/w, 4), 'y': round(c1[1]/h, 4)},
            {'x': round(c2[0]/w, 4), 'y': round(c2[1]/h, 4)},
            {'x': round(c3[0]/w, 4), 'y': round(c3[1]/h, 4)},
            {'x': round(c4[0]/w, 4), 'y': round(c4[1]/h, 4)}
        ]
        boxes_dict.append(box_dict)
    return jsonify({"regions": boxes_dict})


@app.route("/query_display", methods=['POST'])
def query_display():
    try:
        data = request.get_data()
        img = Image.open(BytesIO(data)).convert('RGB')
    except Exception as ex:
        print(ex)
        return create_error_response(ex)

    img = np.array(img)
    _, img_draw, _ = model.text_detect(img)

    pil_im = Image.fromarray(cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
    buff = BytesIO()
    pil_im.save(buff, format='JPEG')
    buff.seek(0)
    return send_file(buff, attachment_filename='processed.jpeg')


if __name__ == "__main__":
    app.run('localhost', CRAFT_PORT, debug=True)
