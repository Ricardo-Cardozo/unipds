
const aiContext = {
    session: null,
    abortController: null,
    isGenerating: false,
    availability: null,
    supportsSamplingParameters: false,
};

const languageOptions = {
    expectedInputLanguages: ['pt'],
    expectedOutputLanguages: ['pt'],
};

const elements = {
    temperature: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temp-value'),
    topKValue: document.getElementById('topk-value'),
    topK: document.getElementById('topK'),
    form: document.getElementById('question-form'),
    questionInput: document.getElementById('question'),
    output: document.getElementById('output'),
    button: document.getElementById('ask-button'),
    year: document.getElementById('year'),
}

async function setupEventListeners() {

    // Update display values for range inputs
    elements.temperature.addEventListener('input', (e) => {
        elements.temperatureValue.textContent = e.target.value;
    });

    elements.topK.addEventListener('input', (e) => {
        elements.topKValue.textContent = e.target.value;
    });

    elements.form.addEventListener('submit', async function (event) {
        event.preventDefault();

        if (aiContext.isGenerating) {
            toggleSendOrStopButton(false)
            return;
        }

        await onSubmitQuestion();
    });
}

async function onSubmitQuestion() {
    const questionInput = elements.questionInput;
    const output = elements.output;
    const question = questionInput.value;

    if (!question.trim()) {
        return;
    }

    // Get parameters from form
    const temperature = parseFloat(elements.temperature.value);
    const topK = parseInt(elements.topK.value);
    console.log('Using parameters:', { temperature, topK });

    // Change button to stop mode
    toggleSendOrStopButton(true);

    output.textContent = aiContext.availability === 'available'
        ? 'Processing your question...'
        : 'Preparando o Gemini Nano... O primeiro download pode demorar alguns minutos.';

    let receivedFirstChunk = false;

    try {
        const aiResponseChunks = askAI(question, temperature, topK);

        for await (const chunk of aiResponseChunks) {
            if (aiContext.abortController.signal.aborted) {
                break;
            }

            if (!receivedFirstChunk) {
                output.textContent = '';
                receivedFirstChunk = true;
            }

            console.log('Received chunk:', chunk);
            output.textContent += chunk;
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.error('Error generating response:', error);
            output.textContent = `⚠️ Erro ao executar o Gemini Nano: ${error.message}`;
        }
    } finally {
        toggleSendOrStopButton(false);
    }
}

function toggleSendOrStopButton(isGenerating) {
    if (isGenerating) {
        // Switch to stop mode
        aiContext.isGenerating = isGenerating;
        elements.button.textContent = 'Parar';
        elements.button.classList.add('stop-button');
    } else {
        // Switch to send mode
        aiContext.abortController?.abort();
        aiContext.isGenerating = isGenerating;
        elements.button.textContent = 'Enviar';
        elements.button.classList.remove('stop-button');
    }
}
async function* askAI(question, temperature, topK) {
    aiContext.abortController?.abort();
    aiContext.abortController = new AbortController();

    // Destroy previous session and create new one with updated parameters
    if (aiContext.session) {
        aiContext.session.destroy();
    }

    const sessionOptions = {
        ...languageOptions,
        initialPrompts: [
            {
                role: 'system', content: `
                Você é um assistente de IA que responde de forma clara e objetiva.
                Responda sempre em formato de texto ao invés de markdown`

            },
        ],
        // O download só pode começar depois de um gesto do usuário. Esta função é
        // executada a partir do submit, portanto o clique em "Enviar" satisfaz a regra.
        monitor(monitor) {
            monitor.addEventListener('downloadprogress', (event) => {
                const percent = event.total
                    ? ((event.loaded / event.total) * 100).toFixed(0)
                    : (event.loaded * 100).toFixed(0);

                elements.output.textContent = `Baixando o Gemini Nano: ${percent}%`;
                console.log(`Downloaded ${percent}%`);
            });
        },
    };

    // topK e temperature pertencem à API experimental anterior. Mantemos os
    // controles da aula quando a versão atual do Chrome ainda oferece params().
    if (aiContext.supportsSamplingParameters) {
        sessionOptions.temperature = temperature;
        sessionOptions.topK = topK;
    }

    const session = await LanguageModel.create(sessionOptions);
    aiContext.session = session;
    aiContext.availability = 'available';

    const responseStream = await session.promptStreaming(
        [
            {
                role: 'user',
                content: question,
            },
        ],
        {
            signal: aiContext.abortController.signal,
        }
    );

    for await (const chunk of responseStream) {
        if (aiContext.abortController.signal.aborted) {
            break;
        }
        yield chunk;
    }
}

async function checkRequirements() {
    const errors = [];
    const returnResults = () => errors.length ? errors : null;

    // @ts-ignore
    const isChrome = !!window.chrome;
    if (!isChrome)
        errors.push("⚠️ Este recurso só funciona no Google Chrome ou Chrome Canary (versão recente).");
    if (!('LanguageModel' in self)) {
        errors.push("⚠️ As APIs nativas de IA não estão ativas.");
        errors.push("Ative a seguinte flag em chrome://flags/:");
        errors.push("- Prompt API for Gemini Nano (chrome://flags/#prompt-api-for-gemini-nano)");
        errors.push("Depois reinicie o Chrome e tente novamente.");
        return returnResults();
    }

    const availability = await LanguageModel.availability(languageOptions);
    aiContext.availability = availability;
    console.log('Language Model Availability:', availability);
    if (availability === 'available') {
        return returnResults();
    }

    if (availability === 'unavailable') {
        errors.push(`⚠️ O seu dispositivo não suporta modelos de linguagem nativos de IA.`);
    }

    if (availability === 'downloading') {
        elements.output.textContent = 'O Gemini Nano está sendo baixado. Digite uma pergunta e clique em "Enviar" para acompanhar.';
        return null;
    }

    if (availability === 'downloadable') {
        // LanguageModel.create() não pode rodar automaticamente durante o load.
        // O Chrome exige um clique/tecla para autorizar o download inicial.
        elements.output.textContent = 'O Gemini Nano precisa ser baixado. Digite uma pergunta e clique em "Enviar" para iniciar.';
        elements.button.textContent = 'Baixar modelo e enviar';
        return null;
    }

    return returnResults();

}

(async function main() {
    elements.year.textContent = new Date().getFullYear();

    const reqErrors = await checkRequirements();
    if (reqErrors) {
        elements.output.innerHTML = reqErrors.join('<br/>');
        elements.button.disabled = true;
        return;
    }

    const fallbackParams = {
        defaultTemperature: 1,
        defaultTopK: 3,
        maxTemperature: 2,
        maxTopK: 128,
    };

    let params = fallbackParams;
    if (typeof LanguageModel.params === 'function') {
        try {
            params = await LanguageModel.params();
            aiContext.supportsSamplingParameters = true;
            console.log('Language Model Params:', params);
        } catch (error) {
            console.warn('Legacy sampling parameters are not available:', error);
        }
    }

    if (!aiContext.supportsSamplingParameters) {
        // A API moderna da web não libera topK/temperature por padrão. Os
        // valores continuam visíveis para preservar a explicação da aula.
        elements.temperature.disabled = true;
        elements.topK.disabled = true;
        elements.temperature.title = 'Parâmetro legado indisponível nesta versão do Chrome.';
        elements.topK.title = 'Parâmetro legado indisponível nesta versão do Chrome.';
        console.warn('topK and temperature are unavailable in this Chrome Prompt API version.');
    }
    /*
    defaultTemperature: 1
    defaultTopK:3
    maxTemperature:2
    maxTopK:128
    */

    elements.topK.max = params.maxTopK;
    elements.topK.min = 1;
    elements.topK.value = params.defaultTopK;
    elements.topKValue.textContent = params.defaultTopK;

    elements.temperatureValue.textContent = params.defaultTemperature;
    elements.temperature.max = params.maxTemperature;
    elements.temperature.min = 0;
    elements.temperature.value = params.defaultTemperature;
    return setupEventListeners()
})();
