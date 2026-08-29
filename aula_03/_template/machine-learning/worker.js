importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

// Estes caminhos ficam prontos para a implementação do carregamento na atividade.
// eslint-disable-next-line no-unused-vars
const MODEL_PATH = `yolov5n_web_model/model.json`;
// eslint-disable-next-line no-unused-vars
const LABELS_PATH = `yolov5n_web_model/labels.json`;


self.onmessage = async({ data }) => {
  if (data.type !== 'predict') return;

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
