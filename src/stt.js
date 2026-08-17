// Speech-to-text factory. Decoupled from the LLM provider because Anthropic has
// no audio API — we transcribe with whatever audio-capable key is available, and
// fall back across providers. Returns { text, provider } or { text:'', error }.
const { pcmToWav } = require('./wav');

async function transcribeOpenAI(apiKey, wav, model, baseURL) {
  const OpenAI = require('openai');
  const toFile = OpenAI.toFile || require('openai/uploads').toFile;
  const client = new OpenAI({ apiKey, ...(baseURL ? { baseURL } : {}) });
  const file = await toFile(wav, 'audio.wav', { type: 'audio/wav' });
  const res = await client.audio.transcriptions.create({ file, model: model || 'whisper-1' });
  return (res.text || '').trim();
}

// Models tried in order — skips quota/overload errors and falls through to the next
const GEMINI_STT_MODELS = [
  'gemini-3.1-flash-lite',
  'gemini-3.5-flash',
  'gemini-2.0-flash-lite',
  'gemini-3.1-flash-image',
  'gemini-2.0-flash',
];

async function transcribeGemini(apiKey, wav) {
  const { GoogleGenAI } = require('@google/genai');
  const ai = new GoogleGenAI({ apiKey });
  let lastErr = null;
  for (const model of GEMINI_STT_MODELS) {
    try {
      const res = await ai.models.generateContent({
        model,
        contents: [{ role: 'user', parts: [
          { text: 'Transcribe this audio verbatim. Return only the spoken words with no commentary. If there is no clear speech, return an empty response.' },
          { inlineData: { mimeType: 'audio/wav', data: wav.toString('base64') } }
        ] }]
      });
      return (res.text || '').trim();
    } catch (e) {
      const status = e && e.status;
      // 429 = quota, 503 = overloaded — try the next model
      if (status === 429 || status === 503 || (e.message && (e.message.includes('429') || e.message.includes('503')))) {
        lastErr = e;
        continue;
      }
      throw e; // other errors (404, 400) — propagate immediately
    }
  }
  throw lastErr || new Error('All Gemini STT models unavailable');
}

function createSTT(settings) {
  const keys = settings.apiKeys || {};
  const chain = [];
  // Groq Whisper: free, fast, reliable — first priority
  if (keys.groq)   chain.push({ p: 'groq',   fn: (wav) => transcribeOpenAI(keys.groq,   wav, 'whisper-large-v3-turbo', 'https://api.groq.com/openai/v1') });
  if (keys.openai) chain.push({ p: 'openai', fn: (wav) => transcribeOpenAI(keys.openai, wav, settings.sttModel) });
  if (keys.gemini) chain.push({ p: 'gemini', fn: (wav) => transcribeGemini(keys.gemini, wav) });

  return {
    available: chain.length > 0,
    providers: chain.map((c) => c.p),
    async transcribe(pcm) {
      if (!chain.length || !pcm || pcm.length < 3200) return { text: '' };
      const wav = pcmToWav(pcm, 16000, 1);
      let lastErr = null;
      for (const c of chain) {
        try {
          const text = await c.fn(wav);
          return { text, provider: c.p };
        } catch (e) {
          lastErr = { status: e && e.status, code: e && e.code, message: (e && e.message) || String(e), provider: c.p };
        }
      }
      return { text: '', error: lastErr };
    }
  };
}

module.exports = { createSTT };
