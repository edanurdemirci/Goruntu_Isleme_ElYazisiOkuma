"""
Single-file Flask web app for handwritten text prediction using a saved Keras model.
Place this file in the same folder as your saved model `handwriting_model.h5` and
`classes.npy` (save LabelBinarizer classes with `np.save('classes.npy', LB.classes_)` when training).

Features:
- Upload an image (jpg/png)
- Server extracts letter bounding boxes, predicts each character with the model
- Returns predicted text and an image with bounding boxes
- Reset button clears the preview on the client

Run: python handwriting_web_app.py
Visit: http://127.0.0.1:5000

Dependencies: flask, tensorflow/keras, numpy, opencv-python, imutils, pillow
"""

from flask import Flask, request, jsonify, render_template_string, send_file
import numpy as np
import cv2
import imutils
import os
import io
import base64
from keras.models import load_model

from werkzeug.utils import secure_filename
from PIL import Image

MODEL_PATH = 'handwriting_model.h5'    # your saved model
CLASSES_PATH = 'classes.npy'           # saved LabelBinarizer.classes_
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
IMG_SIZE = 32

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

print('Loading model...')
model = load_model(MODEL_PATH)
print('Model loaded.')

if os.path.exists(CLASSES_PATH):
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    print(f'Loaded classes: {len(classes)} labels')
else:
    print('Warning: classes.npy not found. Predictions may be incorrect. Please save LabelBinarizer.classes_ as classes.npy')
    classes = np.array([str(i) for i in range(35)])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


def get_letters_and_image_from_path(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError('Unable to read image')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return [], image
    cnts = sort_contours(cnts, method='left-to-right')[0]

    letters = []
    boxes = []
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            try:
                resized = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            except Exception:
                continue
            resized = resized.astype('float32') / 255.0
            resized = np.expand_dims(resized, axis=-1)  # (32,32,1)
            letters.append(resized)

    vis = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return letters, vis


@app.route('/')
def index():
    return render_template_string('''
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>El Yazısı Tanıma</title>
  <style>
    body{ font-family: Arial, sans-serif; max-width:900px; margin:20px auto; padding:10px }
    .preview{ max-width:100%; height:auto; border:1px solid #ddd; padding:6px }
    #result{ font-size:1.25rem; margin-top:10px }
    .controls{ margin-top:10px }
    button{ padding:8px 12px; margin-right:8px }
  </style>
</head>
<body>
  <h2>El Yazısı Tanıma - Foto Yükle, Metni Tahmin Et</h2>
  <input type="file" id="file" accept="image/*"><br>
  <div class="controls">
    <button id="predict">Tahmin Et</button>
    <button id="reset">Reset</button>
  </div>
  <p id="result">Tahmin: <span id="predText">-</span></p>
  <img id="preview" class="preview" src="" alt="preview"/>
  <script>
    const fileInput = document.getElementById('file');
    const preview = document.getElementById('preview');
    const predictBtn = document.getElementById('predict');
    const resetBtn = document.getElementById('reset');
    const predText = document.getElementById('predText');

    fileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      preview.src = url;
      predText.textContent = '-';
    });

    predictBtn.addEventListener('click', async () => {
      if (!fileInput.files[0]) { alert('Lütfen önce bir foto yükleyin.'); return; }
      const form = new FormData();
      form.append('file', fileInput.files[0]);
      predictBtn.disabled = true;
      predictBtn.textContent = 'Tahmin ediliyor...';
      try {
        const res = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json();
        if (data.error) { alert(data.error); }
        else {
          predText.textContent = data.text || '-';
          if (data.image) preview.src = 'data:image/png;base64,' + data.image;
        }
      } catch (err) {
        alert('Tahmin sırasında hata: ' + err);
      }
      predictBtn.disabled = false;
      predictBtn.textContent = 'Tahmin Et';
    });

    resetBtn.addEventListener('click', () => {
      fileInput.value = '';
      preview.src = '';
      predText.textContent = '-';
    });
  </script>
</body>
</html>
''')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya gönderilmedi.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya ismi boş.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Sadece png/jpg/jpeg izinli.'}), 400

    filename = secure_filename(file.filename)
    img_bytes = file.read()
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Resim okunamadı veya bozuk.'}), 400

    tmp_path = 'tmp_upload.png'
    cv2.imwrite(tmp_path, img)

    letters, vis = get_letters_and_image_from_path(tmp_path)
    if not letters:
        _, buf = cv2.imencode('.png', vis)
        b64 = base64.b64encode(buf).decode('utf-8')
        return jsonify({'text': '', 'image': b64, 'message': 'Hiç harf bulunamadı.'})

    batch = np.stack(letters, axis=0)
    preds = model.predict(batch)
    pred_indices = np.argmax(preds, axis=1)
    predicted_chars = [str(classes[idx]) if idx < len(classes) else '?' for idx in pred_indices]

    predicted_text = ''.join(predicted_chars)

    _, buf = cv2.imencode('.png', vis)
    b64 = base64.b64encode(buf).decode('utf-8')

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return jsonify({'text': predicted_text, 'image': b64})


if __name__ == '__main__':
    app.run(debug=True)