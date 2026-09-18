import test from 'node:test'
import assert from 'node:assert/strict'
import { ModelCatalog } from '../src/adapter/catalog.ts'
import { apiForModel, isResponsesOnlyModel, PROVIDER_ID, reasoningEfforts, reasoningEffortWire, RESPONSES_BODY_IDLE_MS, ZenAdapter } from '../src/adapter/zen-adapter.ts'

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
    reasoningCapability: () => ({ reasoning: true, effortValues: [] }),
  })
  const models = adapter.listModels('opencode2dsh')
  assert.deepEqual(models.map((m) => m.id), ['big-pickle', 'mimo-v2.5-free'])
})

test('reasoningEfforts: declared ladder wins, none folds into off, default ladder otherwise', () => {
  // no capability / non-reasoning model: advertise nothing
  assert.equal(reasoningEfforts(undefined), undefined)
  assert.equal(reasoningEfforts({ reasoning: false, effortValues: ['low'] }), undefined)
  // reasoning without a declared ladder: the standard five the gateway accepts
  assert.deepEqual(
    reasoningEfforts({ reasoning: true, effortValues: [] })?.map((e) => e.id),
    ['off', 'minimal', 'low', 'medium', 'high'],
  )
  // declared ladder (muse-spark shape) is offered verbatim, ladder-ordered
  assert.deepEqual(
    reasoningEfforts({ reasoning: true, effortValues: ['xhigh', 'low', 'medium'] })?.map((e) => e.id),
    ['low', 'medium', 'xhigh'],
  )
  // metadata `none` folds into our `off`; out-of-vocabulary values drop
  assert.deepEqual(
    reasoningEfforts({ reasoning: true, effortValues: ['none', 'high', 'banana'] })?.map((e) => e.id),
    ['off', 'high'],
  )
  // selector labels are the capitalized level names
  assert.deepEqual(reasoningEfforts({ reasoning: true, effortValues: [] })?.[0], { id: 'off', name: 'Off' })
})

test('reasoningEffortWire maps picker ids to the gateway spelling', () => {
  // no selection: inject nothing (provider default keeps always-think models thinking)
  assert.equal(reasoningEffortWire(undefined), undefined)
  // off must SEND none — a mere omission never disables Zen's thinking models
  assert.equal(reasoningEffortWire('off'), 'none')
  // ladder levels pass through verbatim
  assert.equal(reasoningEffortWire('low'), 'low')
  assert.equal(reasoningEffortWire('xhigh'), 'xhigh')
  // unknown ids were never advertised; inject nothing rather than risk the 400
  assert.equal(reasoningEffortWire('banana'), undefined)
})

test('resolveModel advertises the thinking-level picker for reasoning models only', () => {
  const adapter = new ZenAdapter({
    list: () => ['big-pickle', 'ghost'],
    decision: () => ({ allowed: true, source: 'test', known: true }),
    reasoningCapability: (model: string) =>
      model === 'big-pickle' ? { reasoning: true, effortValues: ['low', 'high'] } : undefined,
  })
  assert.deepEqual(
    adapter.resolveModel('opencode2dsh', 'big-pickle').reasoning?.efforts.map((e) => e.id),
    ['low', 'high'],
  )
  // unknown metadata: no reasoning field — dsh-llm then offers only the default
  assert.equal(adapter.resolveModel('opencode2dsh', 'ghost').reasoning, undefined)
})

/** Scripted provider that records the streamSimple options it receives. */
function capturingProvider() {
  const captured: Array<{ onPayload?: (payload: unknown) => unknown }> = []
  const provider = {
    streamSimple(_model: unknown, _context: unknown, options: { onPayload?: (payload: unknown) => unknown }): AsyncIterable<{ type: string }> {
      captured.push(options)
      return (async function* () {
        yield { type: 'start' }
        yield { type: 'text_delta', delta: 'hi' }
        yield { type: 'done', message: { stopReason: 'stop', content: [], usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2 } } }
      })()
    },
  }
  return { provider, captured }
}

/** A body that already satisfies the free-lane agent-shape gate. */
const gateBody = {
  model: 'big-pickle',
  messages: [{ role: 'user', content: 'hi' }],
  stream: true,
  tools: ['bash', 'read'].map((name) => ({ type: 'function', function: { name, description: 'd', parameters: {} } })),
}

async function runStream(catalogReasoning: boolean, effort?: string): Promise<Array<{ onPayload?: (payload: unknown) => unknown }>> {
  const { provider, captured } = capturingProvider()
  const adapter = new ZenAdapter(
    {
      list: () => ['big-pickle'],
      decision: () => ({ allowed: true, source: 'test', known: true }),
      reasoningCapability: () => ({ reasoning: catalogReasoning, effortValues: [] }),
    },
    { providerOverride: provider },
  )
  const options = { provider: 'opencode2dsh', model: 'big-pickle', messages: [], temperature: 0, maxTokens: 16 }
  const stream = adapter.stream({ ...options, ...(effort !== undefined ? { reasoningEffort: effort } : {}) } as never)
  for await (const chunk of stream) void chunk
  return captured
}

test('stream injects the selected reasoning_effort into the outgoing body', async () => {
  // off -> wire none (the only spelling that stops the always-think models)
  const offOptions = (await runStream(true, 'off'))[0]!
  assert.deepEqual(offOptions.onPayload?.({ ...gateBody }), { ...gateBody, reasoning_effort: 'none' })

  // ladder levels ride verbatim
  const lowOptions = (await runStream(true, 'low'))[0]!
  assert.deepEqual(lowOptions.onPayload?.({ ...gateBody }), { ...gateBody, reasoning_effort: 'low' })

  // no selection: the onPayload stays the plain gate shaper (a gate-satisfied
  // body needs no rewrite -> undefined, and no effort field is ever added)
  const defaultOptions = (await runStream(true))[0]!
  assert.equal(defaultOptions.onPayload?.({ ...gateBody }), undefined)
})

test('stream keeps the free-lane gate rewrite alongside the effort injection', async () => {
  // a body missing the gate tools gets them AND the effort in one rewrite
  const offOptions = (await runStream(true, 'off'))[0]!
  const shaped = offOptions.onPayload?.({ model: 'big-pickle', messages: [], stream: true }) as Record<string, unknown>
  assert.equal(shaped.reasoning_effort, 'none')
  assert.deepEqual(
    (shaped.tools as Array<{ function: { name: string } }>).map((t) => t.function.name).sort(),
    ['bash', 'read'],
  )
  assert.equal(shaped.tool_choice, 'none')

  // non-chat payloads pass through untouched even with an effort selected
  assert.equal(offOptions.onPayload?.(null), undefined)
})

test('responses-only models (muse-spark-*) route to openai-responses api', async () => {
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
    model: 'muse-spark-1.3-contributor',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
  })
  const chunks = []
  for await (const chunk of stream) chunks.push(chunk)
  assert.equal(capturedModel.id, 'muse-spark-1.3-contributor')
  assert.equal(capturedModel.api, 'openai-responses')
})

test('extended stream body idle watchdog for responses-only models', () => {
  assert.equal(RESPONSES_BODY_IDLE_MS, 300_000)
  assert.equal(isResponsesOnlyModel('muse-spark-1.3-contributor'), true)
  assert.equal(isResponsesOnlyModel('big-pickle'), false)
  assert.equal(apiForModel('muse-spark-1.3-contributor'), 'openai-responses')
  assert.equal(apiForModel('big-pickle'), 'openai-completions')
})

test('responses-only models route reasoning effort to reasoning.effort and never send root reasoning_effort or effort "none"', async () => {
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

  // Case 1: user selects Minimal -> reasoning.effort = 'minimal', no root reasoning_effort
  const streamMinimal = adapter.stream({
    provider: 'opencode2dsh',
    model: 'muse-spark-1.3-contributor',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    reasoningEffort: 'minimal',
  })
  for await (const _ of streamMinimal) {}
  const payloadMinimal = capturedOptions.onPayload({ stream: true })
  assert.equal(payloadMinimal.reasoning_effort, undefined)
  assert.deepEqual(payloadMinimal.reasoning, { effort: 'minimal' })

  // Case 2: user selects High -> reasoning.effort = 'high', no root reasoning_effort
  const streamHigh = adapter.stream({
    provider: 'opencode2dsh',
    model: 'muse-spark-1.3-contributor',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    reasoningEffort: 'high',
  })
  for await (const _ of streamHigh) {}
  const payloadHigh = capturedOptions.onPayload({ stream: true })
  assert.equal(payloadHigh.reasoning_effort, undefined)
  assert.deepEqual(payloadHigh.reasoning, { effort: 'high' })

  // Case 3: user selects Off -> clamped to 'minimal' because upstream rejects 'none'
  const streamOff = adapter.stream({
    provider: 'opencode2dsh',
    model: 'muse-spark-1.3-contributor',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
    reasoningEffort: 'off',
  })
  for await (const _ of streamOff) {}
  const payloadOff = capturedOptions.onPayload({ stream: true })
  assert.equal(payloadOff.reasoning_effort, undefined)
  assert.deepEqual(payloadOff.reasoning, { effort: 'minimal' })

  // Case 4: user selects default (no reasoningEffort) -> pi-ai's reasoning.effort: 'none' is stripped!
  const streamDefault = adapter.stream({
    provider: 'opencode2dsh',
    model: 'muse-spark-1.3-contributor',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'test' }] }],
  })
  for await (const _ of streamDefault) {}
  const payloadDefault = capturedOptions.onPayload({ stream: true, reasoning: { effort: 'none' } })
  assert.equal(payloadDefault.reasoning_effort, undefined)
  assert.equal(payloadDefault.reasoning, undefined)
})
