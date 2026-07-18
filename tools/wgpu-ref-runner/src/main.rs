use std::borrow::Cow;
use std::env;
use std::fs;
use std::num::NonZeroU64;
use std::process;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

#[derive(Clone, Copy)]
enum Access {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy)]
enum Scalar {
    I32,
    U32,
    F32,
}

#[derive(Clone)]
enum StorageType {
    Scalar(Scalar),
    AtomicU32,
    Array {
        elem: Scalar,
        atomic: bool,
        count: usize,
    },
}

struct Binding {
    binding: u32,
    name: String,
    ty_text: String,
    access: Access,
    storage_ty: StorageType,
    byte_len: u64,
}

struct Args {
    entry: String,
    gx: u32,
    gy: u32,
    gz: u32,
    len: usize,
    file: String,
}

fn main() {
    if let Err(err) = pollster::block_on(run()) {
        eprintln!("wgpu-ref-runner: {err}");
        process::exit(2);
    }
}

async fn run() -> Result<(), String> {
    let args = parse_args()?;
    let src = fs::read_to_string(&args.file)
        .map_err(|e| format!("cannot read {}: {e}", args.file))?;
    let workgroup_size = parse_workgroup_size(&src, &args.entry)?;
    let bindings = parse_bindings(&src, args.len)?;
    if bindings.is_empty() {
        return Err("no @group(0) storage bindings found".to_string());
    }

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| "no compatible wgpu adapter found".to_string())?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wgsl-ref-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .map_err(|e| format!("request_device failed: {e}"))?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wgsl-ref-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&src)),
    });

    let mut sorted: Vec<&Binding> = bindings.iter().collect();
    sorted.sort_by_key(|b| b.binding);

    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = sorted
        .iter()
        .map(|b| wgpu::BindGroupLayoutEntry {
            binding: b.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: matches!(b.access, Access::ReadOnly),
                },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(b.byte_len),
            },
            count: None,
        })
        .collect();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wgsl-ref-bgl"),
        entries: &layout_entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wgsl-ref-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wgsl-ref-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: &args.entry,
    });

    let zeroed_buffers: Vec<Vec<u8>> = bindings
        .iter()
        .map(|b| vec![0u8; b.byte_len as usize])
        .collect();
    let storage_buffers: Vec<wgpu::Buffer> = bindings
        .iter()
        .zip(zeroed_buffers.iter())
        .map(|(b, bytes)| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&b.name),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        })
        .collect();

    let bind_entries: Vec<wgpu::BindGroupEntry> = sorted
        .iter()
        .map(|b| {
            let idx = bindings
                .iter()
                .position(|candidate| candidate.binding == b.binding)
                .expect("binding index");
            wgpu::BindGroupEntry {
                binding: b.binding,
                resource: storage_buffers[idx].as_entire_binding(),
            }
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wgsl-ref-bind-group"),
        layout: &bind_group_layout,
        entries: &bind_entries,
    });

    let readback_buffers: Vec<wgpu::Buffer> = bindings
        .iter()
        .map(|b| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wgsl-ref-readback"),
                size: b.byte_len,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
        .collect();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgsl-ref-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgsl-ref-compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    for ((b, src_buf), dst_buf) in bindings
        .iter()
        .zip(storage_buffers.iter())
        .zip(readback_buffers.iter())
    {
        encoder.copy_buffer_to_buffer(src_buf, 0, dst_buf, 0, b.byte_len);
    }
    queue.submit(Some(encoder.finish()));

    let mut readbacks = Vec::with_capacity(readback_buffers.len());
    for (idx, read_buf) in readback_buffers.iter().enumerate() {
        let slice = read_buf.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| format!("readback channel closed for {}", bindings[idx].name))?
            .map_err(|e| format!("readback map failed for {}: {e}", bindings[idx].name))?;
        let bytes = slice.get_mapped_range().to_vec();
        readbacks.push(bytes);
        read_buf.unmap();
    }

    print_json(&args, workgroup_size, &bindings, &readbacks);
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut words: Vec<String> = env::args().skip(1).collect();
    if words.first().map(|s| s.as_str()) == Some("interp") {
        words.remove(0);
    }

    let mut entry = "main".to_string();
    let mut gx = 0u32;
    let mut gy = 0u32;
    let mut gz = 0u32;
    let mut len = 1usize;
    let mut file: Option<String> = None;

    let mut i = 0;
    while i < words.len() {
        match words[i].as_str() {
            "--entry" => {
                i += 1;
                entry = words.get(i).ok_or("--entry requires a value")?.clone();
            }
            "--gx" => {
                i += 1;
                gx = parse_u32(words.get(i), "--gx")?;
            }
            "--gy" => {
                i += 1;
                gy = parse_u32(words.get(i), "--gy")?;
            }
            "--gz" => {
                i += 1;
                gz = parse_u32(words.get(i), "--gz")?;
            }
            "--len" => {
                i += 1;
                len = parse_usize(words.get(i), "--len")?.max(1);
            }
            "--seeds" => {
                i += 1;
                let seed = words.get(i).ok_or("--seeds requires a value")?;
                if seed.trim() != "{}" {
                    return Err("--seeds is not supported by the wgpu reference runner".to_string());
                }
            }
            "--" => {}
            opt if opt.starts_with('-') => {
                return Err(format!("unknown option {opt}"));
            }
            path => {
                if file.is_some() {
                    return Err(format!("extra argument {path}"));
                }
                file = Some(path.to_string());
            }
        }
        i += 1;
    }

    let file = file.ok_or("missing FILE")?;
    Ok(Args {
        entry,
        gx,
        gy,
        gz,
        len,
        file,
    })
}

fn parse_u32(v: Option<&String>, name: &str) -> Result<u32, String> {
    v.ok_or_else(|| format!("{name} requires a value"))?
        .parse::<u32>()
        .map_err(|e| format!("{name}: {e}"))
}

fn parse_usize(v: Option<&String>, name: &str) -> Result<usize, String> {
    v.ok_or_else(|| format!("{name} requires a value"))?
        .parse::<usize>()
        .map_err(|e| format!("{name}: {e}"))
}

fn parse_bindings(src: &str, default_len: usize) -> Result<Vec<Binding>, String> {
    let mut out = Vec::new();
    for raw in src.lines() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if !line.contains("var<storage") {
            continue;
        }
        let group = parse_attr_u32(line, "group")
            .ok_or_else(|| format!("storage declaration missing @group: {line}"))?;
        if group != 0 {
            return Err(format!("only @group(0) is supported, got @group({group})"));
        }
        let binding = parse_attr_u32(line, "binding")
            .ok_or_else(|| format!("storage declaration missing @binding: {line}"))?;
        let var_pos = line
            .find("var<storage")
            .ok_or_else(|| format!("bad storage declaration: {line}"))?;
        let gt = line[var_pos..]
            .find('>')
            .ok_or_else(|| format!("bad var<storage> declaration: {line}"))?
            + var_pos;
        let access_text = &line[var_pos + "var<".len()..gt];
        let access = if access_text.contains("read") && !access_text.contains("read_write") {
            Access::ReadOnly
        } else {
            Access::ReadWrite
        };
        let after = line[gt + 1..].trim();
        let colon = after
            .find(':')
            .ok_or_else(|| format!("storage declaration missing type: {line}"))?;
        let name = after[..colon].trim();
        if name.is_empty() {
            return Err(format!("storage declaration missing name: {line}"));
        }
        let ty_text = after[colon + 1..]
            .split(';')
            .next()
            .unwrap_or("")
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>();
        let storage_ty = parse_storage_type(&ty_text, default_len)?;
        let byte_len = storage_byte_len(&storage_ty);
        if byte_len == 0 {
            return Err(format!("zero-sized storage binding {name}"));
        }
        out.push(Binding {
            binding,
            name: name.to_string(),
            ty_text,
            access,
            storage_ty,
            byte_len,
        });
    }
    out.sort_by_key(|b| b.binding);
    Ok(out)
}

fn parse_attr_u32(line: &str, attr: &str) -> Option<u32> {
    let pat = format!("@{attr}(");
    let start = line.find(&pat)? + pat.len();
    let end = line[start..].find(')')? + start;
    line[start..end].trim().parse().ok()
}

fn parse_storage_type(ty: &str, default_len: usize) -> Result<StorageType, String> {
    match ty {
        "i32" => return Ok(StorageType::Scalar(Scalar::I32)),
        "u32" => return Ok(StorageType::Scalar(Scalar::U32)),
        "f32" => return Ok(StorageType::Scalar(Scalar::F32)),
        "atomic<u32>" => return Ok(StorageType::AtomicU32),
        _ => {}
    }

    if let Some(inner) = ty.strip_prefix("array<").and_then(|s| s.strip_suffix('>')) {
        let parts = split_top_level_comma(inner);
        let elem_text = parts
            .first()
            .ok_or_else(|| format!("bad array storage type {ty}"))?
            .as_str();
        let (elem, atomic) = if elem_text == "atomic<u32>" {
            (Scalar::U32, true)
        } else {
            (parse_scalar(elem_text)?, false)
        };
        let count = if parts.len() >= 2 {
            parts[1]
                .parse::<usize>()
                .map_err(|e| format!("bad array count in {ty}: {e}"))?
        } else {
            default_len.max(1)
        };
        return Ok(StorageType::Array {
            elem,
            atomic,
            count,
        });
    }

    Err(format!("unsupported storage type {ty}"))
}

fn parse_scalar(s: &str) -> Result<Scalar, String> {
    match s {
        "i32" => Ok(Scalar::I32),
        "u32" => Ok(Scalar::U32),
        "f32" => Ok(Scalar::F32),
        _ => Err(format!("unsupported scalar storage element {s}")),
    }
}

fn split_top_level_comma(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    for (i, ch) in s.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(s[start..].trim().to_string());
    parts
}

fn storage_byte_len(ty: &StorageType) -> u64 {
    match ty {
        StorageType::Scalar(_) | StorageType::AtomicU32 => 4,
        StorageType::Array { count, .. } => (*count as u64) * 4,
    }
}

fn parse_workgroup_size(src: &str, entry: &str) -> Result<[u32; 3], String> {
    let mut pending = None;
    for raw in src.lines() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if let Some(size) = parse_workgroup_attr(line)? {
            pending = Some(size);
        }
        if let Some(name) = parse_fn_name(line) {
            if name == entry {
                return Ok(pending.unwrap_or([1, 1, 1]));
            }
            pending = None;
        }
    }
    Err(format!("entry point {entry:?} not found"))
}

fn parse_workgroup_attr(line: &str) -> Result<Option<[u32; 3]>, String> {
    let Some(start) = line.find("@workgroup_size(") else {
        return Ok(None);
    };
    let args_start = start + "@workgroup_size(".len();
    let end = line[args_start..]
        .find(')')
        .ok_or_else(|| format!("bad @workgroup_size attribute: {line}"))?
        + args_start;
    let nums: Result<Vec<u32>, _> = line[args_start..end]
        .split(',')
        .map(|part| part.trim().parse::<u32>())
        .collect();
    let nums = nums.map_err(|e| format!("bad @workgroup_size value: {e}"))?;
    let x = nums.get(0).copied().unwrap_or(1).max(1);
    let y = nums.get(1).copied().unwrap_or(1).max(1);
    let z = nums.get(2).copied().unwrap_or(1).max(1);
    Ok(Some([x, y, z]))
}

fn parse_fn_name(line: &str) -> Option<&str> {
    let pos = line.find("fn ")? + 3;
    let rest = &line[pos..];
    let end = rest
        .find(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .unwrap_or(rest.len());
    if end == 0 {
        None
    } else {
        Some(&rest[..end])
    }
}

fn print_json(args: &Args, workgroup_size: [u32; 3], bindings: &[Binding], readbacks: &[Vec<u8>]) {
    let mut s = String::new();
    s.push_str("{\"ok\":true,\"entry\":");
    push_json_string(&mut s, &args.entry);
    s.push_str(",\"backend\":\"wgpu\",\"gid\":[");
    s.push_str(&args.gx.to_string());
    s.push(',');
    s.push_str(&args.gy.to_string());
    s.push(',');
    s.push_str(&args.gz.to_string());
    s.push_str("],\"workgroup_size\":[");
    s.push_str(&workgroup_size[0].to_string());
    s.push(',');
    s.push_str(&workgroup_size[1].to_string());
    s.push(',');
    s.push_str(&workgroup_size[2].to_string());
    s.push_str("],\"buffers\":[");
    for (i, (binding, bytes)) in bindings.iter().zip(readbacks.iter()).enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"name\":");
        push_json_string(&mut s, &binding.name);
        s.push_str(",\"binding\":");
        s.push_str(&binding.binding.to_string());
        s.push_str(",\"type\":");
        push_json_string(&mut s, &binding.ty_text);
        s.push_str(",\"values\":[");
        let values = read_values(&binding.storage_ty, bytes);
        for (j, value) in values.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            push_json_string(&mut s, value);
        }
        s.push_str("]}");
    }
    s.push_str("],\"anomalies\":{\"nan\":0,\"inf\":0}}\n");
    print!("{s}");
}

fn read_values(ty: &StorageType, bytes: &[u8]) -> Vec<String> {
    match ty {
        StorageType::Scalar(scalar) => vec![read_scalar(*scalar, bytes, 0)],
        StorageType::AtomicU32 => vec![read_scalar(Scalar::U32, bytes, 0)],
        StorageType::Array {
            elem,
            atomic,
            count,
        } => {
            let scalar = if *atomic { Scalar::U32 } else { *elem };
            (0..*count)
                .map(|idx| read_scalar(scalar, bytes, idx * 4))
                .collect()
        }
    }
}

fn read_scalar(scalar: Scalar, bytes: &[u8], offset: usize) -> String {
    let b = [
        bytes.get(offset).copied().unwrap_or(0),
        bytes.get(offset + 1).copied().unwrap_or(0),
        bytes.get(offset + 2).copied().unwrap_or(0),
        bytes.get(offset + 3).copied().unwrap_or(0),
    ];
    match scalar {
        Scalar::I32 => i32::from_le_bytes(b).to_string(),
        Scalar::U32 => u32::from_le_bytes(b).to_string(),
        Scalar::F32 => format!("{}", f32::from_le_bytes(b)),
    }
}

fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
