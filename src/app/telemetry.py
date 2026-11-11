"""Observability in one file: OTel providers, structured logs, and a LangChain callback.

Signals -> OTLP -> Collector -> Tempo (traces) / Prometheus (metrics) / Loki (logs).
Set OTEL_EXPORTER_OTLP_ENDPOINT to turn export on; without it everything still logs to stdout.
Prompts and answers are never logged or put on spans: only sizes, names and token counts.
"""
from collections import deque
import copy
import json
import logging
import os
import re
import sys
import time
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import metrics, trace
from opentelemetry.trace import Span, StatusCode

SERVICE = os.getenv("OTEL_SERVICE_NAME", "extraction-agent")
PROVIDER = "groq"

# Illustrative USD per 1M tokens (input, output) - override with MODEL_PRICES='{"model": [in, out]}'
DEFAULT_PRICES = {"openai/gpt-oss-120b": (0.15, 0.60), "openai/gpt-oss-20b": (0.075, 0.30)}
PRICES = {**DEFAULT_PRICES, **{k: tuple(v) for k, v in json.loads(os.getenv("MODEL_PRICES", "{}")).items()}}

_STD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}
_providers: list[Any] = []


# ---------- providers ----------
DEBUG = os.getenv("ASSESSMENT_DEBUG") == "1"   # exposes /debug/telemetry (used by the autograder and handy while developing)
DEBUG_READER: Any = None                        # in-memory metric reader, only when DEBUG
LOG_BUFFER: deque = deque(maxlen=500)           # last log events as dicts, only when DEBUG


def setup_telemetry() -> None:
    """Call once, before LangChain is used. Without an OTLP endpoint nothing is exported (DEBUG still records in memory)."""
    global DEBUG_READER
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").removeprefix("http://").removeprefix("https://")
    if not endpoint and not DEBUG:
        return
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({
        "service.name": SERVICE,
        "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
        "deployment.environment": os.getenv("DEPLOY_ENV", "demo"),
    })
    # second-scale buckets: the SDK default buckets are millisecond-scale and would flatten LLM latencies
    seconds = ExplicitBucketHistogramAggregation([0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60])
    readers = []
    if DEBUG:
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        DEBUG_READER = InMemoryMetricReader()
        readers.append(DEBUG_READER)
    if endpoint:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        tp = TracerProvider(resource=resource)
        tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
        trace.set_tracer_provider(tp)
        readers.append(PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=True),
            export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "10000")),
        ))
        lp = LoggerProvider(resource=resource)
        lp.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, insecure=True)))
        set_logger_provider(lp)
        _providers.extend([tp, lp])
    mp = MeterProvider(resource=resource, metric_readers=readers, views=[View(instrument_name="*duration", aggregation=seconds)])
    metrics.set_meter_provider(mp)
    _providers.append(mp)


def metrics_snapshot() -> list[dict]:
    """Every metric recorded so far: [{name, points: [{attributes, value|count}]}] (needs ASSESSMENT_DEBUG=1)."""
    data = DEBUG_READER.get_metrics_data() if DEBUG_READER else None
    out = []
    for rm in (data.resource_metrics if data else []):
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                pts = [{"attributes": dict(p.attributes), **({"value": p.value} if hasattr(p, "value") else {"count": p.count, "sum": p.sum})}
                       for p in m.data.data_points]
                out.append({"name": m.name, "points": pts})
    return out


def shutdown_telemetry() -> None:
    """Flush batches on exit so the last requests are not lost."""
    for p in _providers:
        p.shutdown()


# ---------- logging ----------
class JsonFormatter(logging.Formatter):
    """One JSON object per line on stdout; extras (structlog key/values) become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname.lower(),
               "logger": record.name, "event": record.getMessage()}
        out.update({k: v for k, v in vars(record).items() if k not in _STD_ATTRS and k != "level"})
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            out["trace_id"], out["span_id"] = format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def _rename_reserved(_, __, event_dict: dict) -> dict:
    """LogRecord forbids extras named like its own attributes (filename, name, module, ...): `filename=` would raise
    KeyError at INFO level. Rename them (filename -> filename_) so no call site can crash logging."""
    return {(f"{k}_" if k in _STD_ATTRS else k): v for k, v in event_dict.items()}


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # request_id, thread_id bound by middleware
            structlog.stdlib.add_log_level,
            _rename_reserved,
            structlog.processors.format_exc_info,
            structlog.stdlib.render_to_log_kwargs,    # event -> message, key/values -> LogRecord extras
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [stdout]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    if _providers:  # OTLP log export (carries trace_id/span_id from the active span)
        from opentelemetry._logs import get_logger_provider
        from opentelemetry.instrumentation.logging.handler import LoggingHandler
        class FullLineHandler(LoggingHandler):
            """Ship the SAME JSON line as stdout as the OTLP body. By default the body is only the event name and every
            field hides in structured metadata, so Grafana's Logs panel shows 'ingest_failed' and nothing else."""

            def emit(self, record: logging.LogRecord) -> None:
                rec = copy.copy(record)
                rec.msg, rec.args = JsonFormatter().format(record), ()
                super().emit(rec)  # extras still become attributes, so `| level="error"` filters keep working

        otlp = FullLineHandler(logger_provider=get_logger_provider())
        # never ship the exporter's own errors through the exporter (feedback loop when the collector is down)
        otlp.addFilter(lambda r: not r.name.startswith("opentelemetry"))
        root.addHandler(otlp)
    if DEBUG:  # keep recent events as dicts so /debug/telemetry can show them
        class Ring(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                LOG_BUFFER.append(json.loads(JsonFormatter().format(record)))

        root.addHandler(Ring())
    for noisy in ("uvicorn.access", "httpx", "httpcore"):  # our middleware / spans already cover these
        logging.getLogger(noisy).setLevel(logging.WARNING)


log = structlog.get_logger("agent")


# ---------- metrics ----------
_meter = metrics.get_meter("agent")
HTTP_REQUESTS = _meter.create_counter("http.requests", unit="1", description="API hits")
HTTP_DURATION = _meter.create_histogram("http.request.duration", unit="s", description="API latency")
LLM_REQUESTS = _meter.create_counter("llm.requests", unit="1", description="LLM calls")
LLM_TOKENS = _meter.create_counter("llm.tokens", unit="1", description="Tokens used (token_type=input|output)")
LLM_COST = _meter.create_counter("llm.cost.usd", unit="1", description="Estimated spend")
LLM_DURATION = _meter.create_histogram("llm.duration", unit="s", description="LLM call latency")
TOOL_CALLS = _meter.create_counter("tool.calls", unit="1", description="Tool calls")
TOOL_DURATION = _meter.create_histogram("tool.duration", unit="s", description="Tool latency")
INGEST_DOCS = _meter.create_counter("ingest.documents", unit="1", description="Documents ingested (status=ok|error)")
INGEST_DURATION = _meter.create_histogram("ingest.duration", unit="s", description="Ingestion pipeline latency")
OCR_PAGES = _meter.create_counter("ocr.pages", unit="1", description="Pages read with OCR")
KB_SEARCHES = _meter.create_counter("kb.searches", unit="1", description="Knowledge-base searches (found=true|false)")
RUN_STEPS = _meter.create_histogram("run.steps", unit="1", description="Tool calls per /agent request")


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


# ---------- LangChain callback ----------
class TelemetryCallback(BaseCallbackHandler):
    """One handler for every model and tool call, passed via `config={"callbacks": [...]}`.

    Why not langsmith[otel]? Its bridge exports traces only. We need token/cost *metrics*
    per model, so we read `usage_metadata` ourselves and emit spans + metrics together.
    """

    run_inline = True  # keep event order inside async code

    def __init__(self) -> None:
        self._tracer = trace.get_tracer("agent")
        self._open: dict[Any, tuple[Span, float, dict]] = {}
        self._parent: dict[Any, Any] = {}

    # -- plumbing
    def _span_for(self, parent_run_id):
        """Nearest ancestor run that has a span, else the active (HTTP) span."""
        while parent_run_id is not None and parent_run_id not in self._open:
            parent_run_id = self._parent.get(parent_run_id)
        parent = self._open[parent_run_id][0] if parent_run_id is not None else trace.get_current_span()
        return trace.set_span_in_context(parent)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
        self._parent[run_id] = parent_run_id  # chains get no span (too noisy); only remembered

    def _start(self, run_id, parent_run_id, name, attrs):
        span = self._tracer.start_span(name, context=self._span_for(parent_run_id), attributes=attrs)
        self._open[run_id] = (span, time.perf_counter(), attrs)

    def _finish(self, run_id, error: BaseException | None = None):
        span, t0, attrs = self._open.pop(run_id)
        self._parent.pop(run_id, None)
        if error:
            span.record_exception(error)
            span.set_status(StatusCode.ERROR, type(error).__name__)
        span.end()
        return time.perf_counter() - t0, attrs

    # -- LLM
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kw):
        model = (metadata or {}).get("ls_model_name") or kw.get("invocation_params", {}).get("model", "unknown")
        attrs = {"gen_ai.system": PROVIDER, "gen_ai.request.model": model, "gen_ai.operation.name": "chat"}
        self._parent[run_id] = parent_run_id
        self._start(run_id, parent_run_id, f"chat {model}", attrs)

    def on_llm_end(self, response, *, run_id, **kw):
        span, _, attrs = self._open[run_id]
        usage = {}
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or usage
        tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        span.set_attributes({"gen_ai.usage.input_tokens": tin, "gen_ai.usage.output_tokens": tout})
        elapsed, attrs = self._finish(run_id)
        LLM_REQUESTS.add(1, {**attrs, "status": "ok"})
        LLM_DURATION.record(elapsed, attrs)
        LLM_TOKENS.add(tin, {**attrs, "token_type": "input"})
        LLM_TOKENS.add(tout, {**attrs, "token_type": "output"})
        LLM_COST.add(cost_usd(attrs["gen_ai.request.model"], tin, tout), attrs)
        log.info("llm_call", model=attrs["gen_ai.request.model"], tokens_in=tin, tokens_out=tout,
                 duration_ms=round(elapsed * 1000))

    def on_llm_error(self, error, *, run_id, **kw):
        elapsed, attrs = self._finish(run_id, error)
        LLM_REQUESTS.add(1, {**attrs, "status": "error", "error_type": type(error).__name__})
        LLM_DURATION.record(elapsed, attrs)
        log.error("llm_error", model=attrs["gen_ai.request.model"], error_type=type(error).__name__,
                  error=str(error)[:1000])

    # -- tools (local @tool and MCP tools look the same here)
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        name = serialized.get("name", "tool")
        self._parent[run_id] = parent_run_id
        self._start(run_id, parent_run_id, f"tool {name}", {"tool.name": name})
        log.info("tool_start", tool=name, input_chars=len(str(input_str)))

    def on_tool_end(self, output, *, run_id, **kw):
        elapsed, attrs = self._finish(run_id)
        text = str(getattr(output, "content", output))
        failed = getattr(output, "status", None) == "error" or text.startswith("ERROR[")  # handled ToolException
        code = re.match(r"ERROR\[([A-Z_]+)\]", text)
        TOOL_CALLS.add(1, {**attrs, "status": "error" if failed else "ok", **({"error_code": code.group(1)} if code else {})})
        TOOL_DURATION.record(elapsed, attrs)
        log.info("tool_end", tool=attrs["tool.name"], output_chars=len(str(getattr(output, "content", output))),
                 duration_ms=round(elapsed * 1000))

    def on_tool_error(self, error, *, run_id, **kw):
        elapsed, attrs = self._finish(run_id, error)
        TOOL_CALLS.add(1, {**attrs, "status": "error"})
        TOOL_DURATION.record(elapsed, attrs)
        log.error("tool_error", tool=attrs["tool.name"], error_type=type(error).__name__, error=str(error)[:1000])


def run_config(**extra) -> dict:
    """Per-request LangChain config: fresh handler (own span bookkeeping) + any extras (thread_id...)."""
    cfg = {"callbacks": [TelemetryCallback()], "recursion_limit": int(os.getenv("AGENT_MAX_STEPS", "25"))}
    if "thread_id" in extra:
        cfg["configurable"] = {"thread_id": extra.pop("thread_id")}
    return {**cfg, **extra}
