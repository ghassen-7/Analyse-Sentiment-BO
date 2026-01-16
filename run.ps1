# run.ps1 - Execution E2E sous PowerShell
param()

$venvPath = Join-Path -Path (Get-Location) -ChildPath ".venv"
$activate = Join-Path -Path $venvPath -ChildPath "Scripts\Activate.ps1"

if (-Not (Test-Path $activate)) {
    Write-Host ".venv introuvable. Création d'un nouvel environnement Python 3.10..."
    py -3.10 -m venv .venv
}

Write-Host "Activation de l'environnement virtuel..."
. $activate

Write-Host "Installation des dépendances (requirements.txt)..."
pip install --upgrade pip; pip install -r requirements.txt

Write-Host "Création des dossiers reports et figs si manquants..."
New-Item -ItemType Directory -Path .\reports -Force | Out-Null
New-Item -ItemType Directory -Path .\figs -Force | Out-Null
New-Item -ItemType Directory -Path .\data\processed -Force | Out-Null

Write-Host "Lancement du training..."
python .\src\train\tp_min_models.py

Write-Host "Lancement des visualisations..."
python .\src\viz\viz_clean.py

Write-Host "Terminé."
