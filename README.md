# ProteinMPNN + HBDesigner: inference package

Пакет содержит три модели ProteinMPNN — `exp_063`, `exp_082`, `exp_085` — и pH-aware HBDesigner. Во всех командах pH передаётся как непрерывное входное свойство.

## Состав

- `proteinmpnn/weights/exp_063.pt`
- `proteinmpnn/weights/exp_082.pt`
- `proteinmpnn/weights/exp_085.pt`
- `proteinmpnn/run_proteinmpnn.py`
- `proteinmpnn/parse_multiple_chains.py`
- `hbdesigner/weights/hbd_best.pt`
- `hbdesigner/config/hbd_best.yaml`
- исходный inference-код HBDesigner и необходимые packing-веса

## ProteinMPNN

Требования: Python 3.10+, PyTorch, NumPy.

Сначала положите один или несколько PDB в каталог `inputs/` и преобразуйте их в JSONL:

```bash
python proteinmpnn/parse_multiple_chains.py \
  --input_path inputs \
  --output_path parsed.jsonl
```

Запуск `exp_063` при pH 6.5:

```bash
python proteinmpnn/run_proteinmpnn.py \
  --jsonl_path parsed.jsonl \
  --model proteinmpnn/weights/exp_063.pt \
  --ph 6.5 \
  --out_folder outputs/exp_063_ph65 \
  --num_seq_per_target 16 \
  --sampling_temp 0.1
```

Для двух остальных моделей замените имя файла на `exp_082.pt` или `exp_085.pt`. Чтобы проектировать только выбранные цепи, добавьте, например, `--chains_to_design A,C`. Если параметр не задан, проектируются все белковые цепи.

Модель использует 27 состояний: 20 стандартных аминокислот, шесть альтернативных состояний `c/d/e/h/k/r` и `X`. Для каждого запуска создаются два FASTA:

- `*_states.fasta` сохраняет предсказанные состояния;
- `*_20aa.fasta` сворачивает строчные состояния в стандартные `C/D/E/H/K/R`.

Аргумент `--ph` обязателен. Допустимый диапазон — 0–14; рекомендуемый рабочий диапазон — 2–11.

## HBDesigner

HBDesigner требует своё окружение с PyTorch, PyTorch Geometric, PyRosetta и зависимостями из `hbdesigner/env_gpu.yaml`. Установка кода:

```bash
python -m pip install -e hbdesigner
```

Пример запуска при pH 6.5:

```bash
run_hbdesigner \
  --pdb inputs/example.pdb \
  --out_dir outputs/hbd_ph65 \
  --design_model_ckpt hbdesigner/weights/hbd_best.pt \
  --design_model_config hbdesigner/config/hbd_best.yaml \
  --packing_model_ckpt hbdesigner/model_weights/pack.pt \
  --ph 6.5 \
  --predicted_protonation_mode free \
  --rosetta_acid_tautomer P1 \
  --n_samples 100 \
  --top_k 5 \
  --n_res 3
```

`--ph` задаёт целевой pH для нейросети. Режим `--predicted_protonation_mode free` передаёт предсказанное состояние в Rosetta как начальное и разрешает перепаковку. Для ASP/GLU модель различает протонированное и депротонированное состояние, но не различает кислородные таутомеры P1/P2; политика задаётся явно через `--rosetta_acid_tautomer P1` или `P2`.

## Загрузка пакета

Полный архив опубликован в GitHub Releases как `ProteinMPNN_HBDesigner_pH_v1.tar.gz`.

Проверка архива:

```bash
sha256sum ProteinMPNN_HBDesigner_pH_v1.tar.gz
# 0d921c36ea24db2f438e2791a9135cc3c7454f38d17e9c3e179ee7a4a7ba8f66
```

После распаковки:

```bash
sha256sum -c SHA256SUMS
```

## ProteinMPNN training source

The standalone ProteinMPNN training project is available in [`proteinmpnn-training/`](proteinmpnn-training/README.md). It includes pH-conditioned training code for `exp_063`, `exp_082`, and `exp_085`, model and data-loading modules, checkpoint export, portable launch examples, and English documentation describing the training recipe and file locations.
