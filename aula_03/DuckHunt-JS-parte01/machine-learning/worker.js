importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENTIONS = 640;

// Os rótulos são carregados nesta primeira etapa para preparar a evolução feita na parte 2.
// eslint-disable-next-line no-unused-vars
let _labels = [];
let _model = null;
async function loadModelAndLabels() {
  await tf.ready();

  _labels = await (await fetch(LABELS_PATH)).json();
  _model = await tf.loadGraphModel(MODEL_PATH);

  // warmup
  const dummyInput = tf.ones(_model.inputs[0].shape);
  await _model.executeAsync(dummyInput);
  tf.dispose(dummyInput);

  postMessage({ type: 'model-loaded' });

}

/**
 * Pré-processa a imagem para o formato aceito pelo YOLO:
 * - tf.browser.fromPixels(): converte ImageBitmap/ImageData para tensor [H, W, 3]
 * - tf.image.resizeBilinear(): redimensiona para [INPUT_DIM, INPUT_DIM]
 * - .div(255): normaliza os valores para [0, 1]
 * - .expandDims(0): adiciona dimensão batch [1, H, W, 3]
 *
 * Uso de tf.tidy():
 * - Garante que tensores temporários serão descartados automaticamente,
 *   evitando vazamento de memória.
 */
function preprocessImage(input) {
  return tf.tidy(() => {
    const image = tf.browser.fromPixels(input);

    return tf.image
      .resizeBilinear(image, [INPUT_MODEL_DIMENTIONS, INPUT_MODEL_DIMENTIONS])
      .div(255)
      .expandDims(0);
  });
}

async function runInference(tensor) {
  const output = await _model.executeAsync(tensor);
  tf.dispose(tensor);
  // Assume que as 3 primeiras saídas são:
  // caixas (boxes), pontuações (scores) e classes

  const [boxes, scores, classes] = output.slice(0, 3);
  const [boxesData, scoresData, classesData] = await Promise.all(
    [
      boxes.data(),
      scores.data(),
      classes.data(),
    ]
  );

  output.forEach((t) => t.dispose());

  return {
    boxes: boxesData,
    scores: scoresData,
    classes: classesData
  };
}
loadModelAndLabels();

self.onmessage = async({ data }) => {
  if (data.type !== 'predict') return;
  if (!_model) return;

  const input = preprocessImage(data.image);
  await runInference(input);

  postMessage({
    type: 'prediction',
    x: 400,
    y: 400,
    score: 0
  });


};

// Este log confirma, durante a aula, que o Worker terminou sua inicialização.
// eslint-disable-next-line no-console
console.log('🧠 YOLOv5n Web Worker initialized');
