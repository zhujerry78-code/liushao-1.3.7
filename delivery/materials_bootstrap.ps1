param([ValidateSet('prepare','migrate','start','verify','rebuild')][string]$Action='prepare')
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$prefix = $root.TrimEnd('\') + '\'

function Get-Sha256([string]$path) {
    # Use .NET directly: inherited PSModulePath can hide Get-FileHash in
    # Windows PowerShell launched from a newer PowerShell installation.
    $stream = [IO.File]::OpenRead($path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}

function Get-LocalPath([string]$relative) {
    if ([string]::IsNullOrWhiteSpace($relative) -or $relative.Contains(':') -or
        $relative.StartsWith('/') -or $relative.StartsWith('\')) { throw 'Invalid relative material path.' }
    foreach ($part in ($relative -split '[/\\]')) {
        if ($part -eq '' -or $part -eq '.' -or $part -eq '..' -or $part.EndsWith(' ') -or $part.EndsWith('.')) {
            throw 'Unsafe relative material path.'
        }
        if ($part.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
            $part -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Unsafe Windows material name.' }
    }
    $result = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $result.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Material path escaped its root.' }
    $cursor = $result
    while ($cursor.Length -ge $root.Length) {
        if (Test-Path -LiteralPath $cursor) {
            if (((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Linked material paths are not supported.'
            }
        }
        if ($cursor -eq $root) { break }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $result
}

try {
    $manifestPath = Join-Path $root 'MATERIAL_MANIFEST.json'
    if ((Get-Item -LiteralPath $manifestPath).Length -gt 4MB) { throw 'Material manifest exceeds 4 MiB.' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.format -ne 1) { throw 'Unsupported material format.' }
    foreach ($entry in $manifest.files.PSObject.Properties) {
        $file = Get-LocalPath $entry.Name
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw ('Material missing: ' + $entry.Name) }
        if ((Get-Item -LiteralPath $file).Length -ne $entry.Value.bytes -or
            (Get-Sha256 $file) -ne $entry.Value.sha256) {
            throw ('Material changed: ' + $entry.Name)
        }
    }
    $archivePath = Get-LocalPath $manifest.runtime_archive
    $runtimeDirectory = Get-LocalPath $manifest.runtime_directory
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (-not (Test-Path -LiteralPath $runtimeDirectory)) {
        $stageName = $manifest.runtime_directory + '-' + [Guid]::NewGuid().ToString('N') + '.tmp'
        $stage = Get-LocalPath $stageName
        $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
        try {
            $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
            $total = [long]0
            foreach ($entry in $archive.Entries) {
                $name = $entry.FullName.TrimEnd('/')
                $null = Get-LocalPath ($stageName + '/' + $name)
                if (-not $seen.Add($name) -or (($entry.ExternalAttributes -shr 16) -band 61440) -eq 40960) {
                    throw 'Archive contains duplicate or linked entries.'
                }
                $total += $entry.Length
                if ($total -gt 2GB) { throw 'Archive expands beyond 2 GiB.' }
            }
            $null = New-Item -ItemType Directory -Path $stage
            foreach ($entry in $archive.Entries) {
                $target = Get-LocalPath ($stageName + '/' + $entry.FullName.TrimEnd('/'))
                if ($entry.FullName.EndsWith('/')) {
                    $null = New-Item -ItemType Directory -Path $target -Force
                } else {
                    $null = New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($target)) -Force
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $false)
                }
            }
        } finally { $archive.Dispose() }
        # Both absolute paths have been checked to stay inside this bundle root.
        $stage = Get-LocalPath $stageName
        $runtimeDirectory = Get-LocalPath $manifest.runtime_directory
        [IO.Directory]::Move($stage, $runtimeDirectory)
    }
    $package = Get-LocalPath ($manifest.runtime_directory + '/' + $manifest.runtime_root)
    # Verify the expanded interpreter and helpers from the already checked ZIP
    # before executing any expanded Python program, including on repeated use.
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        $entry = $archive.GetEntry($manifest.runtime_root + '/PACKAGE_MANIFEST.json')
        if ($null -eq $entry -or $entry.Length -gt 4MB) { throw 'Invalid runtime manifest.' }
        $stream = $entry.Open()
        $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
        try { $runtimeManifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
        foreach ($entry in $runtimeManifest.files.PSObject.Properties) {
            $file = Get-LocalPath ($manifest.runtime_directory + '/' + $manifest.runtime_root + '/' + $entry.Name)
            if (-not (Test-Path -LiteralPath $file -PathType Leaf) -or
                (Get-Item -LiteralPath $file).Length -ne $entry.Value.bytes -or
                (Get-Sha256 $file) -ne $entry.Value.sha256) {
                throw ('Expanded runtime changed: ' + $entry.Name)
            }
        }
    } finally { $archive.Dispose() }
    $python = Join-Path $package 'runtime\python\python.exe'
    $entryPoint = Get-LocalPath 'tools/material_entry.py'
    $env:PYTHONUTF8 = '1'
    & $python -I -X utf8 $entryPoint --root $root --action $Action
    exit $LASTEXITCODE
} catch {
    Write-Host ('Material operation failed: ' + $_.Exception.Message)
    exit 1
}
