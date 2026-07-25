<#
.SYNOPSIS
    Start the local CPU-only embeddings sidecar (llama-server) for Pass 2d catalog matching.

.DESCRIPTION
    Serves the Jina v5 Q8_0 GGUF text-matching model over an OpenAI-compatible
    /v1/embeddings endpoint on 127.0.0.1:8081. The RealtorVision analyzer's
    embeddings backend (EMBEDDINGS_BACKEND=openai_compatible) talks to this server;
    it does NOT fall back to the in-process v3 model, so this MUST be started BEFORE
    running the analyzer / catalog auditor / model comparison tools.

.PREREQUISITES
    Install llama.cpp (provides llama-server), build b9637 or newer:

        winget install llama.cpp

    Or point LLAMA_SERVER_EXE at a specific llama-server.exe:

        $env:LLAMA_SERVER_EXE = "C:\path\to\llama-server.exe"

.NOTES
    On first launch, llama.cpp downloads the model (~639 MB) into the Hugging Face
    cache; subsequent launches reuse the cached weights and start quickly.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start-embeddings-server.ps1
#>

$ErrorActionPreference = "Stop"

$ModelId  = "jinaai/jina-embeddings-v5-text-small-text-matching-GGUF:Q8_0"
$HostAddr = "127.0.0.1"
$Port     = 8081
$CtxSize  = 2048
$MinBuild = 9637

# --- Resolve the llama-server executable ------------------------------------
$exe = $env:LLAMA_SERVER_EXE
if ([string]::IsNullOrWhiteSpace($exe)) {
    $cmd = Get-Command "llama-server" -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        Write-Error "llama-server not found on PATH. Install it (winget install llama.cpp) or set LLAMA_SERVER_EXE."
        exit 1
    }
    $exe = $cmd.Source
}
if (-not (Test-Path $exe)) {
    Write-Error "llama-server executable not found at: $exe"
    exit 1
}
Write-Host "Using llama-server: $exe"

# --- Require build b9637 or newer -------------------------------------------
# llama-server prints --version to stderr; in Windows PowerShell 5.1 a 2>&1
# redirect wraps those lines in ErrorRecords, which $ErrorActionPreference=Stop
# would turn into a throw. Relax the preference just for this probe.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $versionText = (& $exe --version 2>&1 | Out-String)
} catch {
    $versionText = ""
} finally {
    $ErrorActionPreference = $prevEap
}
$m = [regex]::Match($versionText, 'version:\s*(\d+)')
if ($m.Success) {
    $build = [int]$m.Groups[1].Value
    if ($build -lt $MinBuild) {
        Write-Error "llama-server build b$build is too old; b$MinBuild or newer is required. Upgrade with: winget upgrade llama.cpp"
        exit 1
    }
    Write-Host "llama-server build b$build (>= b$MinBuild) OK"
} else {
    Write-Warning "Could not parse llama-server build from --version output; expected b$MinBuild or newer. Continuing."
}

# --- Launch (blocking; Ctrl+C to stop) --------------------------------------
Write-Host "Starting embeddings server for $ModelId on http://${HostAddr}:${Port}/v1 ..."
Write-Host "(First launch downloads ~639 MB into the Hugging Face cache.)"

& $exe `
    -hf $ModelId `
    --embedding `
    --pooling last `
    --n-gpu-layers 0 `
    --host $HostAddr `
    --port $Port `
    --ctx-size $CtxSize `
    --alias $ModelId
