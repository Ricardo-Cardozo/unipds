function main() {
  try {
    const categorias = ["premium", "medium", "basic"];

    const pessoas = [
      {
        nome: "Erick",
        idade: 30,
        cor: "azul",
        localizacao: "São Paulo",
        categoria: "premium",
      },
      {
        nome: "Ana",
        idade: 25,
        cor: "vermelho",
        localizacao: "Rio",
        categoria: "medium",
      },
      {
        nome: "Carlos",
        idade: 40,
        cor: "verde",
        localizacao: "Curitiba",
        categoria: "basic",
      },
    ];

    // Descobre todas as possibilidades existentes
    const cores = [...new Set(pessoas.map((pessoa) => pessoa.cor))];

    const localizacoes = [
      ...new Set(pessoas.map((pessoa) => pessoa.localizacao)),
    ];

    const idades = pessoas.map((pessoa) => pessoa.idade);
    const idadeMinima = Math.min(...idades);
    const idadeMaxima = Math.max(...idades);

    function normalizarIdade(idade) {
      return (idade - idadeMinima) / (idadeMaxima - idadeMinima);
    }

    function gerarOneHot(valorDaPessoa, valoresPossiveis) {
      return valoresPossiveis.map((valor) => {
        return valorDaPessoa === valor ? 1 : 0;
      });
    }

    const entradas = pessoas.map((pessoa) => {
      const idadeNormalizada = normalizarIdade(pessoa.idade);
      const corCodificada = gerarOneHot(pessoa.cor, cores);
      const localizacaoCodificada = gerarOneHot(
        pessoa.localizacao,
        localizacoes,
      );

      return [
        idadeNormalizada,
        ...corCodificada,
        ...localizacaoCodificada,
      ];
    });

    const saidas = pessoas.map((pessoa) => {
      return gerarOneHot(pessoa.categoria, categorias);
    });

    console.log("Ordem das entradas:");
    console.log(["idade", ...cores, ...localizacoes]);

    console.log("Ordem das saídas:");
    console.log(categorias);

    console.log("Entradas:");
    console.log(entradas);

    console.log("Saídas:");
    console.log(saidas);
  } catch (error) {
    console.error("Erro ao processar os dados:", error);
  }
}

main();