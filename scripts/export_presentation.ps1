param([string]$Revision = 'v9')
$ErrorActionPreference = 'Stop'
if ($Revision -notmatch '^v[0-9]+$') { throw 'Revision must be v followed by digits.' }
$projectRoot = Split-Path $PSScriptRoot -Parent
$deckPath = Join-Path $projectRoot "outputs/presentation/gridpulse-defense-$Revision.pptx"
$pdfPath = Join-Path $projectRoot "outputs/presentation/gridpulse-defense-$Revision.pdf"
$renderPath = Join-Path $projectRoot "tmp/ppt-render-$Revision"
if (-not (Test-Path -LiteralPath $deckPath)) { throw 'Deck is missing.' }
New-Item -ItemType Directory -Path $renderPath -Force | Out-Null
$ownedPowerPoint = $null
$ownedPresentation = $null
try {
    $ownedPowerPoint = New-Object -ComObject PowerPoint.Application
    $ownedPresentation = $ownedPowerPoint.Presentations.Open($deckPath, -1, 0, 0)
    $ownedPresentation.SaveAs($pdfPath, 32)
    for ($number = 1; $number -le $ownedPresentation.Slides.Count; $number++) {
        $ownedPresentation.Slides.Item($number).Export((Join-Path $renderPath "slide-$number.png"), 'PNG', 1600, 900)
    }
    Write-Output "Exported $($ownedPresentation.Slides.Count) slides and PDF."
} finally {
    if ($null -ne $ownedPresentation) {
        $ownedPresentation.Close()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($ownedPresentation)
    }
    if ($null -ne $ownedPowerPoint) {
        $ownedPowerPoint.Quit()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($ownedPowerPoint)
    }
}
