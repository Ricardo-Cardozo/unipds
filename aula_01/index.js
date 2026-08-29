import tf from "@tensorflow/tfjs-node";

// 1. DADOS BRUTOS
// As características são as entradas; categoria é a resposta correta.
const categorias = ["premium", "medium", "basic"];
const pessoas = [
  { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo", categoria: "premium" },
  { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio", categoria: "medium" },
  { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba", categoria: "basic" },
];

// 2. ORDEM FIXA DAS COLUNAS
// Set elimina repetições. Esta mesma ordem deve ser usada nas previsões.
const cores = [...new Set(pessoas.map((pessoa) => pessoa.cor))];
const localizacoes = [...new Set(pessoas.map((pessoa) => pessoa.localizacao))];
const ordemDasEntradas = ["idade_normalizada", ...cores, ...localizacoes];

// Guardamos os limites do treino para normalizar pessoas novas da mesma forma.
const idades = pessoas.map((pessoa) => pessoa.idade);
const idadeMinima = Math.min(...idades);
const idadeMaxima = Math.max(...idades);

function normalizarIdade(idade) {
  const intervalo = idadeMaxima - idadeMinima;
  return intervalo === 0 ? 0 : (idade - idadeMinima) / intervalo;
}

// Exemplo: ("vermelho", ["azul", "vermelho", "verde"]) => [0, 1, 0]
function gerarOneHot(valorDaPessoa, valoresPossiveis) {
  return valoresPossiveis.map((valor) => valorDaPessoa === valor ? 1 : 0);
}

// Produz: [idade normalizada, ...cores, ...localizações].
function codificarEntrada(pessoa) {
  return [
    normalizarIdade(pessoa.idade),
    ...gerarOneHot(pessoa.cor, cores),
    ...gerarOneHot(pessoa.localizacao, localizacoes),
  ];
}

// 3. MATRIZES X E Y
// O mesmo índice liga entrada e resposta: entradas[0] e saidas[0] são de Erick.
const entradas = pessoas.map(codificarEntrada);
const saidas = pessoas.map((pessoa) =>
  gerarOneHot(pessoa.categoria, categorias),
);

console.log("Ordem das entradas:", ordemDasEntradas);
console.log("Ordem das saídas:", categorias);

// tensor2d apenas transforma as matrizes prontas em tensores.
const inputXs = tf.tensor2d(entradas);
const outputYs = tf.tensor2d(saidas);
inputXs.print();
outputYs.print();

// 4. REDE NEURAL E TREINAMENTO
async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  // A camada recebe uma posição para cada coluna da matriz de entrada.
  // ReLU permite à camada oculta aprender relações não lineares.
  model.add(tf.layers.dense({
    inputShape: [ordemDasEntradas.length],
    units: 80,
    activation: "relu",
  }));

  // Um neurônio por categoria. Softmax transforma a saída em probabilidades.
  model.add(tf.layers.dense({
    units: categorias.length,
    activation: "softmax",
  }));

  model.compile({
    optimizer: "adam", // ajusta os pesos para reduzir o erro
    loss: "categoricalCrossentropy", // compara previsão e resposta one-hot
    metrics: ["accuracy"],
  });

  await model.fit(inputXs, outputYs, {
    epochs: 200, // 200 passagens completas pelos exemplos
    verbose: 0,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
      },
    },
  });
  return model;
}

// 5. PREVISÃO
async function predict(model, pessoa) {
  // Reutilizar codificarEntrada garante a mesma ordem utilizada no treino.
  // O array externo cria um lote com uma pessoa.
  const inputTensor = tf.tensor2d([codificarEntrada(pessoa)]);
  const predictionTensor = model.predict(inputTensor);
  const predictionArray = await predictionTensor.array();

  inputTensor.dispose();
  predictionTensor.dispose();

  return predictionArray[0].map((prob, index) => ({ prob, index }));
}

const model = await trainModel(inputXs, outputYs);

const pessoa = {
  nome: "Guilherme",
  idade: 28,
  cor: "verde",
  localizacao: "Curitiba",
};

const predictions = await predict(model, pessoa);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map(({ prob, index }) =>
    `${categorias[index]} (${(prob * 100).toFixed(2)}%)`,
  )
  .join("\n");

console.log(`Predições para ${pessoa.nome}:\n${results}`);

inputXs.dispose();
outputYs.dispose();
