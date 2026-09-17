import test from 'node:test'
import assert from 'node:assert/strict'
import { ModelCatalog } from '../src/adapter/catalog.ts'
import {
  DEFAULT_BODY_IDLE_MS,
  PROVIDER_ID,
  RESPONSES_BODY_IDLE_MS,
  RESPONSES_THINKING_MAP,
  ZenAdapter,
  apiForModel,
  isResponsesOnlyModel,
} from '../src/adapter/zen-adapter.ts'

/**
 * The exact method surface dsh-llm touches on a registered adapter. A missing
 * member throws inside registerAdapter and silently drops the provider from
 * the model selector (regression: providerRetryPolicy, index.js:1208).
 */
test('ZenAdapter implements the full dsh-llm adapter surface', () => {
  const adapter = new ZenAdapter(new ModelCatalog())
  for (const method of ['providerInfo', 'providerRetryPolicy', 'listModels', 'resolveModel', 'prepareCall', 'stream']) {
    assert.equal(typeof (adapter as unknown as Record<string, unknown>)[method], 'function', `missing method: ${method}`)
  }
})

test('providerInfo preserves the route id and names the provider', () => {
  const adapter = new ZenAdapter(new ModelCatalog())
  assert.deepEqual(adapter.providerInfo('opencode2dsh'), { id: 'opencode2dsh', name: PROVIDER_ID })
})

test('providerRetryPolicy defers to the host default', () => {
  const adapter = new ZenAdapter(new ModelCatalog())
  assert.equal(adapter.providerRetryPolicy('opencode2dsh'), undefined)
})

test('resolveModel declares text-only input and finite limits', () => {
  const adapter = new ZenAdapter(new ModelCatalog())
  const resolved = adapter.resolveModel('opencode2dsh', 'big-pickle')
  assert.deepEqual(resolved.inputModalities, ['text'])
  assert.equal(resolved.context.contextWindow > 0, true)
  assert.equal(resolved.defaultMaxTokens > 0, true)
  assert.equal(resolved.provider, 'opencode2dsh')
  assert.equal(resolved.id, 'big-pickle')
})

test('prepareCall returns the resolved model and a stream dispatcher', async () => {
  const adapter = new ZenAdapter(new ModelCatalog())
  const call = await adapter.prepareCall('opencode2dsh', 'big-pickle')
  assert.equal(call.model.id, 'big-pickle')
  assert.equal(typeof call.stream, 'function')
})

test('listModels mirrors the catalog without duplicates', () => {
  const adapter = new ZenAdapter({
    list: () => ['big-pickle', 'big-pickle', 'mimo-v2.5-free'],
    decision: () => ({ allowed: true, source: 'test', known: true }),
  })
  const models = adapter.listModels('opencode2dsh')
  assert.deepEqual(models.map((m) => m.id), ['big-pickle', 'mimo-v2.5-free'])
})

test('isResponsesOnlyModel and apiForModel correctly classify muse-spark-* models', () => {
  assert.equal(isResponsesOnlyModel('muse-spark-050'), true)
  assert.equal(isResponsesOnlyModel('muse-spark-v1'), true)
  assert.equal(isResponsesOnlyModel('muse-spark-deepseek'), true)
  assert.equal(apiForModel('muse-spark-050'), 'openai-responses')
  assert.equal(apiForModel('muse-spark-v1'), 'openai-responses')

  assert.equal(isResponsesOnlyModel('big-pickle'), false)
  assert.equal(isResponsesOnlyModel('qwen-free'), false)
  assert.equal(isResponsesOnlyModel('deepseek-r1'), false)
  assert.equal(apiForModel('big-pickle'), 'openai-completions')
  assert.equal(apiForModel('qwen-free'), 'openai-completions')

  assert.equal(RESPONSES_BODY_IDLE_MS, 300_000)
  assert.equal(DEFAULT_BODY_IDLE_MS, 120_000)
})

test('streamSimple routes muse-spark-* to openai-responses with reasoning', async () => {
  let capturedModel: any
  const fakeProvider = {
    streamSimple: (model: unknown) => {
      capturedModel = model
      return (async function* () {
        yield { type: 'start', partial: { content: [] } }
        yield { type: 'thinking_delta', contentIndex: 0, delta: 'pondering' }
        yield { type: 'text_delta', contentIndex: 1, delta: 'spark output' }
        yield {
          type: 'done',
          message: {
            api: (model as any).api,
            provider: 'opencode2dsh',
            model: (model as any).id,
            content: [{ type: 'thinking', thinking: 'pondering' }, { type: 'text', text: 'spark output' }],
            usage: { input: 5, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: 15 },
            stopReason: 'stop',
          },
        }
      })()
    },
  }
  const adapter = new ZenAdapter(new ModelCatalog(), { providerOverride: fakeProvider })
  const stream = adapter.stream({
    provider: 'opencode2dsh',
    model: 'muse-spark-050',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
  })
  const chunks = []
  for await (const chunk of stream) chunks.push(chunk)
  assert.equal(capturedModel.id, 'muse-spark-050')
  assert.equal(capturedModel.api, 'openai-responses')
  assert.equal(capturedModel.reasoning, true)
  assert.ok(chunks.some((c) => c.type === 'reasoning-delta'))
  assert.ok(chunks.some((c) => c.type === 'text-delta'))
})

test('streamSimple routes standard models to openai-completions without reasoning', async () => {
  let capturedModel: any
  const fakeProvider = {
    streamSimple: (model: unknown) => {
      capturedModel = model
      return (async function* () {
        yield { type: 'start', partial: { content: [] } }
        yield { type: 'text_delta', contentIndex: 0, delta: 'regular output' }
        yield {
          type: 'done',
          message: {
            api: (model as any).api,
            provider: 'opencode2dsh',
            model: (model as any).id,
            content: [{ type: 'text', text: 'regular output' }],
            usage: { input: 5, output: 5, cacheRead: 0, cacheWrite: 0, totalTokens: 10 },
            stopReason: 'stop',
          },
        }
      })()
    },
  }
  const adapter = new ZenAdapter(new ModelCatalog(), { providerOverride: fakeProvider })
  const stream = adapter.stream({
    provider: 'opencode2dsh',
    model: 'big-pickle',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
  })
  const chunks = []
  for await (const chunk of stream) chunks.push(chunk)
  assert.equal(capturedModel.id, 'big-pickle')
  assert.equal(capturedModel.api, 'openai-completions')
  assert.equal(capturedModel.reasoning, false)
})

test('muse-spark-* model defines thinkingLevelMap with off: null and valid effort levels', async () => {
  let capturedModel: any
  const fakeProvider = {
    streamSimple: (model: unknown) => {
      capturedModel = model
      return (async function* () {
        yield { type: 'start', partial: { content: [] } }
        yield { type: 'done', message: { content: [], usage: {}, stopReason: 'stop' } }
      })()
    },
  }
  const adapter = new ZenAdapter(new ModelCatalog(), { providerOverride: fakeProvider })
  const stream = adapter.stream({
    provider: 'opencode2dsh',
    model: 'muse-spark-1.3-contributor',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
  })
  for await (const _ of stream) {}
  assert.equal(capturedModel.reasoning, true)
  assert.deepEqual(capturedModel.thinkingLevelMap, RESPONSES_THINKING_MAP)
  assert.equal(capturedModel.thinkingLevelMap.off, null)
  assert.equal(capturedModel.thinkingLevelMap.minimal, 'minimal')
  assert.equal(capturedModel.thinkingLevelMap.low, 'low')
  assert.equal(capturedModel.thinkingLevelMap.medium, 'medium')
  assert.equal(capturedModel.thinkingLevelMap.high, 'high')
  assert.equal(capturedModel.thinkingLevelMap.xhigh, 'xhigh')
  assert.equal(capturedModel.thinkingLevelMap.max, 'max')
})

test('reasoning effort "none" or "off" is sanitized to "minimal" for muse-spark-*', async () => {
  let capturedOptions: any
  const fakeProvider = {
    streamSimple: (_model: unknown, _context: unknown, options: unknown) => {
      capturedOptions = options
      return (async function* () {
        yield { type: 'start', partial: { content: [] } }
        yield { type: 'done', message: { content: [], usage: {}, stopReason: 'stop' } }
      })()
    },
  }
  const adapter = new ZenAdapter(new ModelCatalog(), { providerOverride: fakeProvider })

  // Test reasoningEffort: 'none'
  const streamNone = adapter.stream({
    provider: 'opencode2dsh',
    model: 'muse-spark-1.3-contributor',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    reasoningEffort: 'none',
  })
  for await (const _ of streamNone) {}
  assert.equal(capturedOptions.reasoning, 'minimal')

  // Test reasoningEffort: 'off'
  const streamOff = adapter.stream({
    provider: 'opencode2dsh',
    model: 'muse-spark-1.3-contributor',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    reasoningEffort: 'off',
  })
  for await (const _ of streamOff) {}
  assert.equal(capturedOptions.reasoning, 'minimal')

  // Test valid effort: 'high'
  const streamHigh = adapter.stream({
    provider: 'opencode2dsh',
    model: 'muse-spark-1.3-contributor',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    reasoningEffort: 'high',
  })
  for await (const _ of streamHigh) {}
  assert.equal(capturedOptions.reasoning, 'high')

  // Test onPayload callback sanitizes any remaining effort: 'none'
  const payloadWithEffortNone = { reasoning: { effort: 'none' }, reasoning_effort: 'none' }
  const sanitized = capturedOptions.onPayload(payloadWithEffortNone)
  assert.equal(sanitized.reasoning.effort, 'minimal')
  assert.equal(sanitized.reasoning_effort, 'minimal')
})

