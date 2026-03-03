#!/usr/bin/env bash
# setup_docling.sh — Установка Docling в изолированное окружение
# Запускать из папки docling_eval/
# Требует: Python 3.10+, CUDA 11.8 driver (535)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv_docling"

echo "═══════════════════════════════════════════════"
echo " Docling Setup — GTX 1050 Max-Q (CUDA 11.8)"
echo "═══════════════════════════════════════════════"

# 1. Активация venv
if [ ! -d "$VENV" ]; then
    echo "🔧 Создание venv..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
echo "✅ venv активирован: $VENV"

# 2. pip upgrade
pip install --upgrade pip --quiet

# 3. PyTorch + CUDA 11.8
echo ""
echo "📦 Установка PyTorch cu118..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

# 4. Docling
echo "📦 Установка Docling..."
pip install docling --quiet

# 5. Проверка
echo ""
echo "─────────────────────────────────────────────"
echo "🔍 Проверка установки:"
python3 -c "
import torch, docling
print(f'  torch      : {torch.__version__}')
print(f'  CUDA OK    : {torch.cuda.is_available()}')
print(f'  GPU name   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"—\"}')
print(f'  docling    : {docling.__version__}')
"

echo ""
echo "✅ Установка завершена!"
echo "   Запустить тест: python run_docling_test.py"
