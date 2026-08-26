// Generates opencode provider entries from the llama.cpp router preset file.
//
// The router (llama-server --models-preset) serves one OpenAI-compatible
// endpoint and swaps models on demand. Every [section] in the INI file becomes
// a model id on that endpoint, so the INI is the single source of truth and
// this plugin mirrors it into opencode's config at startup.
//
// Overrides (both are exported by `local-llm env`):
//   LOCAL_LLM_PRESET            path to the preset file
//   LOCAL_LLM_OPENAI_BASE_URL   endpoint the router listens on
//
// Installed by `local-llm integrate opencode`.

import fs from "node:fs"
import os from "node:os"
import path from "node:path"

const PROVIDER_ID = "llamacpp"

const INI_PATH =
  process.env.LOCAL_LLM_PRESET ?? path.join(os.homedir(), ".config", "local-llm", "models.ini")

const BASE_URL = process.env.LOCAL_LLM_OPENAI_BASE_URL ?? "http://127.0.0.1:5678/v1"

// Minimal INI reader: [section] headers plus `key = value` pairs. Comments start
// with # or ; at the beginning of a line or after whitespace, which keeps the
// # characters that can legally appear inside a filesystem path intact.
function parseIni(text) {
  const sections = new Map()
  let current = null

  for (const rawLine of text.split("\n")) {
    const line = rawLine.replace(/(^|\s)[#;].*$/, "").trim()
    if (line === "") continue

    const header = line.match(/^\[(.+)\]$/)
    if (header) {
      current = header[1].trim()
      if (!sections.has(current)) sections.set(current, {})
      continue
    }

    const separator = line.indexOf("=")
    if (separator === -1) continue
    if (current === null) continue // top-level keys such as `version = 1`

    const key = line.slice(0, separator).trim()
    const value = line.slice(separator + 1).trim()
    sections.get(current)[key] = value
  }

  return sections
}

function toInt(value, fallback) {
  const parsed = Number.parseInt(value ?? "", 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

function buildModels(sections) {
  const defaults = sections.get("*") ?? {}
  const models = {}

  for (const [id, raw] of sections) {
    if (id === "*") continue
    const entry = { ...defaults, ...raw }
    if (!entry.model) continue // not a servable model section

    const context = toInt(entry.c ?? entry["ctx-size"] ?? entry["context-size"], 32768)
    const output = toInt(entry["n-predict"], Math.min(8192, context))

    models[id] = {
      name: id,
      // A vision projector means the model can take images.
      attachment: Boolean(entry.mmproj),
      // Sections that declare a reasoning format emit a thinking channel.
      reasoning: Boolean(entry["reasoning-format"]),
      tool_call: true,
      temperature: true,
      limit: { context, output },
    }
  }

  return models
}

export default async () => ({
  config: async (config) => {
    let text
    try {
      text = fs.readFileSync(INI_PATH, "utf8")
    } catch {
      // No preset file on this machine: leave the config untouched rather than
      // breaking startup.
      return
    }

    const models = buildModels(parseIni(text))
    if (Object.keys(models).length === 0) return

    config.provider ??= {}
    const provider = (config.provider[PROVIDER_ID] ??= {})

    provider.npm ??= "@ai-sdk/openai-compatible"
    provider.name ??= "llama.cpp (local)"
    provider.options = {
      baseURL: BASE_URL,
      apiKey: process.env.LOCAL_LLM_API_KEY ?? "not-needed",
      ...provider.options,
    }

    // Anything hand-written in opencode.json wins over the generated entry.
    provider.models = { ...models, ...provider.models }
  },
})
