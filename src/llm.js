const DEBUG = true; // Set to false to disable debug logging
// LLM factory — OpenAI / Anthropic / Gemini behind one streaming interface.
// stream({ system, turns:[{role,text}], imageDataUrl, maxTokens, onToken }) -> Promise<fullText>

function stripDataUrl(dataUrl) {
  const m = /^data:(.+?);base64,(.*)$/s.exec(dataUrl || '');
  return m ? { mime: m[1], b64: m[2] } : null;
}

async function streamOpenAI({ apiKey, model, system, turns, imageDataUrl, maxTokens, onToken, baseURL }) {
  if (DEBUG) console.log('[DEBUG LLM] streamOpenAI called', { model, baseURL, hasImage: !!imageDataUrl, maxTokens });
  const OpenAI = require('openai');
  const client = new OpenAI({ apiKey, baseURL });
  const messages = [{ role: 'system', content: system }];
  turns.forEach((t, i) => {
    const last = i === turns.length - 1;
    if (last && imageDataUrl && t.role === 'user') {
      messages.push({ role: 'user', content: [
        { type: 'text', text: t.text },
        { type: 'image_url', image_url: { url: imageDataUrl } }
      ] });
    } else {
      messages.push({ role: t.role, content: t.text });
    }
  });
  if (DEBUG) console.log('[DEBUG LLM] streamOpenAI sending request to OpenAI SDK with messages count:', messages.length);
  try {
    const stream = await client.chat.completions.create({ model, messages, stream: true, max_tokens: maxTokens });
    let full = '';
    for await (const part of stream) {
      const d = part.choices && part.choices[0] && part.choices[0].delta && part.choices[0].delta.content;
      if (d) { full += d; onToken(d); }
    }
    if (DEBUG) console.log('[DEBUG LLM] streamOpenAI finished successfully, total length:', full.length);
    return full;
  } catch (err) {
    if (DEBUG) console.error('[DEBUG LLM] streamOpenAI error:', err);
    throw err;
  }
}

async function streamAnthropic({ apiKey, model, system, turns, imageDataUrl, maxTokens, onToken }) {
  if (DEBUG) console.log('[DEBUG LLM] streamAnthropic called', { model, hasImage: !!imageDataUrl, maxTokens });
  const Anthropic = require('@anthropic-ai/sdk');
  const client = new Anthropic({ apiKey });
  const messages = turns.map((t, i) => {
    const last = i === turns.length - 1;
    if (last && imageDataUrl && t.role === 'user') {
      const img = stripDataUrl(imageDataUrl);
      const content = [];
      if (img) content.push({ type: 'image', source: { type: 'base64', media_type: img.mime, data: img.b64 } });
      content.push({ type: 'text', text: t.text });
      return { role: 'user', content };
    }
    return { role: t.role, content: t.text };
  });
  if (DEBUG) console.log('[DEBUG LLM] streamAnthropic sending request to Anthropic SDK with messages count:', messages.length);
  try {
    const stream = await client.messages.create({ model, max_tokens: maxTokens, system, messages, stream: true });
    let full = '';
    for await (const ev of stream) {
      if (ev.type === 'content_block_delta' && ev.delta && ev.delta.type === 'text_delta') { full += ev.delta.text; onToken(ev.delta.text); }
    }
    if (DEBUG) console.log('[DEBUG LLM] streamAnthropic finished successfully, total length:', full.length);
    return full;
  } catch (err) {
    if (DEBUG) console.error('[DEBUG LLM] streamAnthropic error:', err);
    throw err;
  }
}

// LLM models tried in order on quota/overload errors
const GEMINI_LLM_MODELS = [
  'gemini-3.5-flash',
  'gemini-3.1-flash-lite',
  'gemini-3.5-flash-lite',
  'gemini-3.1-flash-image',
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite',
];

async function streamGemini({ apiKey, model, system, turns, imageDataUrl, maxTokens, onToken }) {
  if (DEBUG) console.log('[DEBUG LLM] streamGemini called', { model, hasImage: !!imageDataUrl, maxTokens });
  const { GoogleGenAI } = require('@google/genai');
  const ai = new GoogleGenAI({ apiKey });

  const contents = turns.map((t, i) => {
    const last = i === turns.length - 1;
    const parts = [{ text: t.text || '' }];
    if (last && imageDataUrl && t.role === 'user') {
      const img = stripDataUrl(imageDataUrl);
      if (img) parts.push({ inlineData: { mimeType: img.mime, data: img.b64 } });
    }
    return { role: t.role === 'assistant' ? 'model' : 'user', parts };
  });

  // Build the list: try the configured model first, then fallbacks
  const modelsToTry = [model, ...GEMINI_LLM_MODELS.filter(m => m !== model)];
  let lastErr = null;

  for (const tryModel of modelsToTry) {
    try {
      if (DEBUG) console.log('[DEBUG LLM] streamGemini trying model:', tryModel);
      const responseStream = await ai.models.generateContentStream({
        model: tryModel,
        contents,
        config: { systemInstruction: system }
      });
      let full = '';
      for await (const chunk of responseStream) {
        const t = chunk.text; // string getter in v2.x
        if (t) { full += t; onToken(t); }
      }
      if (DEBUG) console.log('[DEBUG LLM] streamGemini finished with', tryModel, '— length:', full.length);
      return full;
    } catch (err) {
      const msg = (err && err.message) || '';
      const is429 = err.status === 429 || msg.includes('429') || msg.includes('Too Many');
      const is503 = err.status === 503 || msg.includes('503') || msg.includes('UNAVAILABLE') || msg.includes('high demand');
      const is404 = err.status === 404 || msg.includes('404') || msg.includes('not found');
      if (is429 || is503 || is404) {
        if (DEBUG) console.log('[DEBUG LLM] model', tryModel, 'failed (', err.status || msg.slice(0, 40), ') — trying next model...');
        lastErr = err;
        continue; // try next model
      }
      if (DEBUG) console.error('[DEBUG LLM] streamGemini fatal error:', err);
      throw err;
    }
  }
  if (DEBUG) console.error('[DEBUG LLM] all Gemini models exhausted');
  throw lastErr || new Error('All Gemini models are currently unavailable (quota/overload)');
}


function createLLM(settings) {
  const provider = settings.provider;
  const keys = settings.apiKeys || {};
  const apiKey = keys[provider];
  const tier = settings.smart ? 'smart' : 'fast';
  const model = (settings.models[provider] || {})[tier];
  
  // Set to 4096 (effectively unlimited for a single response) 
  // since some SDKs like Anthropic require a maxTokens value.
  const maxTokens = 4096;

  if (DEBUG) console.log('[DEBUG LLM] createLLM initialized:', { provider, model, isKeyPresent: !!apiKey, ready: !!apiKey && !!model });

  return {
    provider, model, apiKey,
    ready: !!apiKey && !!model,
    async stream(params) {
      if (DEBUG) console.log('[DEBUG LLM] stream() invoked for provider:', provider);
      const args = { apiKey, model, maxTokens, ...params };
      if (provider === 'openai') return streamOpenAI(args);
      if (provider === 'nvidia') return streamOpenAI({ ...args, baseURL: 'https://integrate.api.nvidia.com/v1' });
      if (provider === 'groq')   return streamOpenAI({ ...args, baseURL: 'https://api.groq.com/openai/v1' });
      if (provider === 'anthropic') return streamAnthropic(args);
      if (provider === 'gemini') return streamGemini(args);
      throw new Error('unknown provider: ' + provider);
    }
  };
}

module.exports = { createLLM };
