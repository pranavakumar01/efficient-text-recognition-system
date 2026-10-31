import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, Images, Layers, Play, Trophy, Zap } from 'lucide-react'
import { api, demoImageAsFile } from '../api/client.js'
import { useAsync } from '../hooks/useAsync.js'
import { PageHeader } from '../components/Layout.jsx'
import { FileDrop } from '../components/FileDrop.jsx'
import { DEFAULT_RECOGNIZE_OPTIONS } from '../components/PreprocessControls.jsx'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  ErrorState,
  Loading,
  ProvenanceBadge,
} from '../components/ui.jsx'
import { cx, errorTone, ms, percent, rate } from '../lib/format.js'

/**
 * Run one image through several models with identical preprocessing.
 *
 * The models differ only in the recognition step, which is what makes the
 * comparison fair — and what makes the SimCLR effect visible on a single
 * image rather than only in an aggregate table.
 */
export default function Compare() {
  const [file, setFile] = useState(null)
  const [selected, setSelected] = useState([])
  const [groundTruth, setGroundTruth] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const abort = useRef(null)

  const models = useAsync(() => api.models(), [])
  const demos = useAsync(() => api.demoImages(), [])

  // Preselect the two experiment arms: that is the comparison this project
  // exists to make.
  useEffect(() => {
    if (!models.data?.models || selected.length) return
    const available = models.data.models.filter((m) => m.available)
    const arms = available.filter((m) => ['control', 'treatment'].includes(m.experiment_arm))
    setSelected((arms.length >= 2 ? arms : available.slice(0, 2)).map((m) => m.id))
  }, [models.data, selected.length])

  const toggle = (id) =>
    setSelected((current) =>
      current.includes(id) ? current.filter((x) => x !== id) : [...current, id].slice(0, 6),
    )

  const run = async () => {
    if (!file || selected.length === 0) return
    abort.current?.abort()
    abort.current = new AbortController()
    setBusy(true)
    setError(null)
    try {
      const data = await api.compare({
        file,
        modelIds: selected,
        options: DEFAULT_RECOGNIZE_OPTIONS,
        groundTruth: groundTruth.trim() || undefined,
        signal: abort.current.signal,
      })
      setResult(data)
    } catch (caught) {
      if (caught.name !== 'AbortError') setError(caught)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  const useDemo = async (filename) => {
    try {
      setFile(await demoImageAsFile(filename))
      const demo = demos.data?.images?.find((d) => d.file === filename)
      setGroundTruth(demo?.text || '')
      setResult(null)
    } catch (caught) {
      setError(caught)
    }
  }

  return (
    <div>
      <PageHeader
        title="Compare models"
        description="Run the same image through several models with identical preprocessing, and see exactly where they differ."
      />

      <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
        <div className="space-y-6">
          <Card>
            <CardHeader title="Input image" icon={Images} />
            <div className="p-5">
              <FileDrop file={file} onChange={(f) => { setFile(f); setResult(null) }} disabled={busy} />
              {demos.data?.images?.length > 0 && (
                <div className="mt-4 flex flex-wrap gap-2">
                  {demos.data.images.slice(0, 6).map((demo) => (
                    <button
                      key={demo.file}
                      onClick={() => useDemo(demo.file)}
                      disabled={busy}
                      className="overflow-hidden rounded-lg border border-slate-200 transition-colors hover:border-indigo-400 disabled:opacity-50 dark:border-slate-700"
                      title={demo.text}
                    >
                      <img
                        src={api.demoImageUrl(demo.file)}
                        alt={demo.text}
                        className="h-10 w-auto max-w-[140px] object-contain"
                      />
                    </button>
                  ))}
                </div>
              )}
              <div className="mt-5">
                <label className="label" htmlFor="gt">
                  Ground truth (optional)
                </label>
                <input
                  id="gt"
                  className="input"
                  placeholder="Enables CER and WER for every model"
                  value={groundTruth}
                  onChange={(e) => setGroundTruth(e.target.value)}
                  disabled={busy}
                />
              </div>
            </div>
          </Card>

          <Card>
            <CardHeader title="Models to compare" subtitle="Up to six" icon={Layers} />
            {models.loading ? (
              <Loading />
            ) : (
              <div className="divide-y divide-slate-100 dark:divide-slate-800">
                {models.data?.models?.map((model) => (
                  <label
                    key={model.id}
                    className={cx(
                      'flex cursor-pointer items-start gap-3 px-5 py-3',
                      !model.available && 'cursor-not-allowed opacity-50',
                    )}
                  >
                    <input
                      type="checkbox"
                      className="mt-0.5 h-4 w-4 rounded accent-indigo-600"
                      checked={selected.includes(model.id)}
                      disabled={!model.available || busy}
                      onChange={() => toggle(model.id)}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className="truncate text-sm font-medium">{model.name}</span>
                        <ProvenanceBadge provenance={model.provenance} />
                      </span>
                      <span className="mt-0.5 block text-[11px] leading-relaxed text-slate-500">
                        {model.unavailable_reason || model.description}
                      </span>
                    </span>
                  </label>
                ))}
              </div>
            )}
            <div className="border-t border-slate-200 p-4 dark:border-slate-800">
              <Button
                className="w-full"
                icon={Play}
                onClick={run}
                loading={busy}
                disabled={!file || selected.length === 0}
              >
                Compare {selected.length} model{selected.length === 1 ? '' : 's'}
              </Button>
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          {error && <ErrorState error={error} onRetry={run} />}

          {busy && (
            <Card>
              <Loading label="Running every model…" />
            </Card>
          )}

          {!result && !busy && !error && (
            <Card>
              <EmptyState icon={Layers} title="No comparison yet">
                Pick an image and at least one model, then run the comparison. A model that fails
                — Transformer OCR with no network, for instance — is reported on its own and the
                others still return.
              </EmptyState>
            </Card>
          )}

          {result && (
            <>
              <div className="grid gap-4 sm:grid-cols-2">
                <Card className="px-4 py-3.5">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Zap className="h-3.5 w-3.5" /> Fastest
                  </div>
                  <p className="mt-1.5 truncate text-lg font-semibold">
                    {result.results.find((r) => r.model_id === result.fastest_model_id)?.model_name ||
                      '—'}
                  </p>
                </Card>
                <Card className="px-4 py-3.5">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Trophy className="h-3.5 w-3.5" /> Most accurate
                  </div>
                  <p className="mt-1.5 truncate text-lg font-semibold">
                    {result.most_accurate_model_id
                      ? result.results.find((r) => r.model_id === result.most_accurate_model_id)
                          ?.model_name
                      : 'Add a ground truth'}
                  </p>
                </Card>
              </div>

              {result.ground_truth && (
                <Card>
                  <CardHeader title="Ground truth" />
                  <p className="px-5 py-4 font-mono text-sm">{result.ground_truth}</p>
                </Card>
              )}

              {result.results.map((item) => (
                <Card key={item.model_id}>
                  <CardHeader
                    title={item.model_name}
                    subtitle={
                      item.error
                        ? 'This model could not run'
                        : `${ms(item.total_ms)} · ${item.line_count} line${item.line_count === 1 ? '' : 's'} · ${percent(item.confidence)} confidence`
                    }
                    action={<ProvenanceBadge provenance={item.provenance} />}
                  />
                  {item.error ? (
                    <div className="p-5">
                      <Alert tone="warning" title="Not available">
                        <p className="whitespace-pre-wrap">{item.error}</p>
                      </Alert>
                    </div>
                  ) : (
                    <>
                      <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words px-5 py-4 font-mono text-sm">
                        {item.text || <span className="text-slate-400">(no text recognised)</span>}
                      </pre>
                      {item.cer !== null && item.cer !== undefined && (
                        <div className="flex gap-6 border-t border-slate-200 px-5 py-3 text-xs dark:border-slate-800">
                          <span>
                            CER{' '}
                            <span className={cx('nums font-semibold', errorTone(item.cer))}>
                              {rate(item.cer)}
                            </span>
                          </span>
                          <span>
                            WER{' '}
                            <span className={cx('nums font-semibold', errorTone(item.wer))}>
                              {rate(item.wer)}
                            </span>
                          </span>
                          <span>
                            Exact match{' '}
                            <Badge tone={item.exact_match ? 'emerald' : 'slate'}>
                              {item.exact_match ? 'yes' : 'no'}
                            </Badge>
                          </span>
                        </div>
                      )}
                    </>
                  )}
                </Card>
              ))}

              {result.results.some((r) => r.error) && (
                <Alert tone="info" icon={AlertTriangle}>
                  One or more models could not run. The comparison above shows the models that
                  did; see each card for the reason.
                </Alert>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
