// EXEMPLO DIDÁTICO — VALIDAÇÃO DE RESPOSTA E RE-PROMPT COM ZOD
//
// Instalação da dependência:
// npm install zod
//
// Este arquivo concentra o fluxo de validação estudado na Aula 2 de Prompt.
// A função que chama o provedor de IA foi recebida como dependência para que o
// exemplo não fique preso a uma API ou a um modelo específico.

import { z } from "zod";

// O schema transforma o contrato de saída em regras executáveis.
// .strict() impede o uso silencioso de propriedades não declaradas.
export const AnswerSchema = z.object({
  title: z.string().min(1),
  summary: z.string().min(1).max(1000),
  bullets: z.array(z.string().min(1)).max(7),
  example_js: z.string().nullable(),
  source_ids: z.array(z.string()).refine(
    (values) => new Set(values).size === values.length,
    "source_ids não pode conter valores repetidos",
  ),
}).strict();

// JSON.parse pode falhar antes mesmo da validação do schema.
// A função devolve undefined para representar esse erro de maneira controlada.
function tryParseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

// O modelo não precisa receber um stack trace. Enviamos somente o caminho do
// campo e uma mensagem de validação curta, reduzindo ruído e vazamento interno.
function formatZodIssues(issues) {
  return issues.map((issue) => ({
    path: issue.path.join("."),
    message: issue.message,
  }));
}

// Este prompt de reparo não muda a tarefa original. Ele apenas solicita a
// correção do contrato que falhou e proíbe a invenção de novos fatos.
function buildRepairPrompt(errors) {
  return JSON.stringify({
    task: "repair_invalid_response",
    instructions: [
      "Corrija somente a estrutura da resposta anterior",
      "Preserve apenas fatos apoiados pelas fontes originais",
      "Não acrescente explicações fora do JSON",
      "Retorne somente o JSON corrigido",
    ],
    validation_errors: errors,
  });
}

// O limite impede loops infinitos, gasto imprevisível de tokens e excesso de
// latência. MAX_REPROMPTS = 2 significa uma chamada inicial e até dois reparos.
const MAX_REPROMPTS = 2;

export async function generateValidatedAnswer({ callModel, originalMessages }) {
  // Clonamos a lista para não alterar o array que pertence ao chamador.
  const messages = [...originalMessages];

  for (let attempt = 0; attempt <= MAX_REPROMPTS; attempt += 1) {
    const raw = await callModel(messages);
    const parsedJson = tryParseJson(raw);

    // Se o JSON nem sequer puder ser lido, criamos um erro específico para o
    // re-prompt. Caso contrário, Zod verifica tipos, limites e chaves extras.
    const result = parsedJson === undefined
      ? {
          success: false,
          error: {
            issues: [{ path: [], message: "A resposta não é JSON válido" }],
          },
        }
      : AnswerSchema.safeParse(parsedJson);

    if (result.success) {
      // result.data é o objeto validado e tipado pelo schema.
      return result.data;
    }

    if (attempt === MAX_REPROMPTS) {
      // Em produção, este ponto também deve gerar uma métrica e um log seguro.
      throw new Error("Resposta inválida apó as tentativas permitidas");
    }

    const validationErrors = formatZodIssues(result.error.issues);

    // Mantemos a resposta inválida no histórico para o modelo saber exatamente
    // qual conteúdo precisa reparar.
    messages.push({ role: "assistant", content: raw });
    messages.push({
      role: "user",
      content: buildRepairPrompt(validationErrors),
    });
  }

  // O laço sempre retorna uma resposta ou lança um erro. Esta linha protege o
  // código caso o fluxo seja modificado incorretamente no futuro.
  throw new Error("Fluxo de validação terminou em estado inesperado");
}
