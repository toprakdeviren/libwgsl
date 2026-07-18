/**
 * TypeScript contract for `bindings/wgsl.js` — the single supported
 * JS/TS surface over libwgsl's wasm build.
 *
 * Struct field offsets live in `WGSLAbi` (filled from `wgsl_abi_layout()`);
 * do not hardcode wasm32 layouts in application code.
 */

export interface StructLayout {
  size: number;
  [field: string]: number;
}

export interface WGSLAbi {
  abiVersion: number;
  ptrSize: number;
  diagnostic: {
    size: number;
    severity: number;
    line: number;
    column: number;
    endLine: number;
    endColumn: number;
    message: number;
    rule: number;
  };
  lexToken: {
    size: number;
    kind: number;
    offset: number;
    length: number;
    line: number;
    column: number;
  };
  hover: {
    size: number;
    present: number;
    typeRepr: number;
    signature: number;
    doc: number;
    symbolOffset: number;
    symbolLength: number;
  };
  definition: {
    size: number;
    present: number;
    offset: number;
    length: number;
  };
  interp: {
    stepCap: number;
    bufferValuesCap: number;
    maxLanes: number;
  };
}

export interface Diagnostic {
  severity: number;
  line: number;
  column: number;
  endLine: number;
  endColumn: number;
  message: string;
  rule: string;
}

export interface HoverInfo {
  present: boolean;
  typeRepr?: string;
  signature?: string;
  doc?: string;
  symbolOffset?: number;
  symbolLength?: number;
}

export interface DefinitionInfo {
  present: boolean;
  offset?: number;
  length?: number;
}

export interface LexToken {
  kind: number;
  offset: number;
  length: number;
  line: number;
  column: number;
}

export interface FormatRangeResult {
  text: string;
  editStart: number;
  editEnd: number;
}

export interface OptimizeFinding {
  pass?: string;
  kind?: string;
  message?: string;
  offset?: number;
  length?: number;
  savings_hint?: number;
}

export interface OptimizeReport {
  passes?: string[];
  summary?: Record<string, unknown>;
  findings?: OptimizeFinding[];
}

export interface MLAnalyzeReport {
  version?: number;
  kind?: string;
  entry?: string;
  is_compute?: boolean;
  patterns?: string[];
  workgroup_hints?: Record<string, unknown>;
  signals?: Record<string, unknown>;
  bindings?: Record<string, unknown>[];
  shape_hints?: Record<string, unknown>[];
  advice?: string[];
}

/** Shape of `wgsl_module_json` (see docs/libwgsl.md §5). */
export interface ModuleSummary {
  spec_pin: string;
  entry_points: EntryPoint[];
  pipelines?: Pipeline[];
  resources: Resource[];
  structs: StructDecl[];
  overrides: OverrideDecl[];
  enable: string[];
  requires: string[];
  diagnostics: { severity: string; rule: string }[];
}

export interface EntryPoint {
  name: string;
  stage: 'vertex' | 'fragment' | 'compute';
  workgroup_size?: [number, number, number];
  bindings: { group: number; binding: number }[];
  features: string[];
  bind_group_layouts: BindGroupLayout[];
  workgroup_bytes?: number;
  inputs: IoSlot[];
  outputs: IoSlot[];
}

export interface BindGroupLayout {
  group: number;
  entries: BindGroupEntry[];
}

export interface BindGroupEntry {
  binding: number;
  visibility: string | string[];
  buffer?: { type: string; hasDynamicOffset: boolean; minBindingSize: number };
  sampler?: { type: string };
  texture?: { sampleType: string; viewDimension: string; multisampled?: boolean };
  storageTexture?: { access: string; format: string; viewDimension: string };
  externalTexture?: Record<string, never>;
}

export interface IoSlot {
  name?: string;
  type: string;
  location?: number;
  builtin?: string;
  interpolate?: { type: string; sampling?: string };
  blend_src?: number;
  invariant?: boolean;
}

export interface Pipeline {
  kind: 'compute' | 'render';
  entry_point?: string;
  vertex?: string;
  fragment?: string;
  inferred?: boolean;
  workgroup_size?: [number, number, number];
  workgroup_bytes?: number;
  bind_group_layouts: BindGroupLayout[];
  vertex_attributes?: {
    shaderLocation: number;
    format?: string;
    type: string;
  }[];
  color_targets?: { location: number; type: string }[];
}

export interface Resource {
  name: string;
  group: number;
  binding: number;
  address_space: string;
  access: string;
  type: string;
  size?: number;
  align?: number;
  layout?: { name: string; offset: number }[];
  handle?: Record<string, unknown>;
}

export interface StructDecl {
  name: string;
  size: number;
  align: number;
  members: {
    name: string;
    type: string;
    offset: number;
    size: number;
    align: number;
  }[];
}

export interface OverrideDecl {
  name: string;
  type: string;
  id: number | null;
}

/** Shape of `wgsl_interp` JSON (see include/wgsl.h). */
export interface InterpTrace {
  ok: boolean;
  entry: string;
  gid?: [number, number, number];
  workgroup_size?: [number, number, number];
  lanes?: number;
  steps?: InterpStep[];
  buffers?: { name: string; type: string; values: string[] }[];
  anomalies?: { nan: number; inf: number };
  cost?: {
    flops: number;
    iops: number;
    bytes_loaded: number;
    bytes_stored: number;
    mix?: Record<string, number>;
  };
  roofline?: Record<string, unknown>;
  memory?: Record<string, unknown>;
  cost_band?: Record<string, unknown>;
  races?: { count: number; samples: { line: number; kind: string }[] };
  divergence?: Record<string, unknown>;
  return?: string | null;
  steps_executed?: number;
  truncated?: boolean;
  warnings?: { line: number; message: string }[];
  error?: string;
}

export interface InterpStep {
  line: number;
  op: string;
  name?: string;
  value?: string;
  anomaly?: string;
  active?: number;
  epoch?: number;
}

/** Shape of `wgsl_debug_oracle` JSON. */
export interface DebugOracleReport {
  ok: boolean;
  verdict: 'pass' | 'fail';
  wedge: 'correctness-oracle+explainer';
  entry: string;
  gid?: [number, number, number];
  issues: DebugOracleIssue[];
  summary: {
    issue_count: number;
    race_count: number;
    nan: number;
    inf: number;
    truncated: boolean;
    race_truncated: boolean;
  };
  why: string;
  source: 'wgsl_interp';
}

export interface DebugOracleIssue {
  kind: string;
  severity: 'error';
  line?: number;
  race_kind?: string;
  nan?: number;
  inf?: number;
  summary: string;
  why: string;
  ci: 'fail';
}

export interface WGSLClient {
  module: unknown;
  abi: WGSLAbi;

  init(): void;
  shutdown(): void;
  specPin(): string;
  unicodeVersion(): string;

  check(src: string): number;
  free(r: number): void;
  ok(r: number): boolean;
  error(r: number): string;
  moduleJson(r: number): string;
  moduleJsonLen(r: number): number;
  moduleJsonParsed(src: string): ModuleSummary | null;

  diagnosticCount(r: number): number;
  diagnostic(r: number, index: number): Diagnostic | null;
  readDiagnostic(dptr: number): Diagnostic;

  hoverAt(r: number, offset: number): HoverInfo;
  definitionAt(r: number, offset: number): DefinitionInfo;
  semanticTokens(src: string): LexToken[];
  format(src: string): string;
  formatRange(src: string, startOffset: number, endOffset: number): FormatRangeResult;
  mlAnalyze(src: string): MLAnalyzeReport;
  optimize(src: string): OptimizeReport;
  optimizeApply(src: string): string;
  toMsl(src: string): string;

  interp(
    src: string,
    entry: string,
    gx?: number,
    gy?: number,
    gz?: number,
    defaultLen?: number,
    seeds?: string,
  ): InterpTrace;
  debugOracle(
    src: string,
    entry: string,
    gx?: number,
    gy?: number,
    gz?: number,
    defaultLen?: number,
    seeds?: string,
  ): DebugOracleReport;
  freeString(p: number): void;
}

export function makeClient(M: unknown): WGSLClient;
export function load(wasmJsPath?: string): Promise<WGSLClient>;
export const DEFAULT_WASM_JS: string;
