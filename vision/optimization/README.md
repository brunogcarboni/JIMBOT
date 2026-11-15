# Quantização do NanoDet para NCNN int8

O script `quantize_ncnn.py` converte o checkpoint da trena em um par NCNN int8. A calibração roda no PC, onde estão o PyTorch e o `model_best.ckpt`. O arquivo gerado é o mesmo em x86 e em aarch64. A Raspberry só entra na inferência, com a libncnn compilada para ela.

A versão das tools que geram o `.param` e o `.bin` tem que ser a mesma libncnn da placa. O clone usado nesta máquina é o commit `24ea07e87f544a3fbc59980b57885a92c75f2d24` (`20261020-61-g24ea07e8`).

## O que o script faz

1. Carrega `run/model_best.ckpt` com a arquitetura de `run/trena.yml` (NanoDet-Plus, ShuffleNetV2 1.0x, entrada 320×320).
2. Exporta a rede com o PNNX, direto do módulo PyTorch, em fp32. O `channel_shuffle` do ShuffleNet vira `ShuffleChannel`.
3. Roda `ncnnoptimize` e grava a rede fp32 otimizada.
4. Calibra as ativações com até 200 fotos da trena (`ncnn2table`, método KL).
5. Gera o modelo int8 com `ncnn2int8`.

O caminho pelo ONNX e pelo `onnx2ncnn` não serve para este modelo. O export em opset 11 quebra o simplificador, e o `onnx2ncnn` transforma o shuffle de canais em camadas `Shape`. O NCNN não executa esse grafo: o `ncnnoptimize` avisa `network graph not ready` e o `ncnn2table` recusa o arquivo.

## Arquivos em `optimization/out`

| Arquivo | Papel |
| --- | --- |
| `trena.pt` | TorchScript intermediário do PNNX |
| `trena.param` e `trena.bin` | Rede fp32, antes do optimize |
| `trena-opt.param` e `trena-opt.bin` | Rede fp32 otimizada, entrada da calibração |
| `trena.table` | Estatística da calibração |
| `trena-int8.param` e `trena-int8.bin` | Modelo quantizado. Os dois vão juntos para a placa |

A entrada do modelo int8 se chama `in0` e a saída `out0`. O pré-processamento é BGR 320×320, com `mean=[103.53,116.28,123.675]` e `norm=[0.017429,0.017507,0.017125]`, lidos de `trena.yml`.

## Executar no PC

A venv do projeto fica em `vision/.venv`. Se o `activate` ainda apontar para um caminho antigo, `python` cai no pyenv e o `torch` some. Recarregue a venv depois de qualquer mudança de pasta:

```bash
cd vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pnnx
```

O PyTorch não está no `requirements.txt`. Nesta máquina a venv já tem o `torch` 2.4.1 com CUDA. O clone do NanoDet precisa existir em `third_party/nanodet`.

O NCNN é compilado no PC. Neste i5-8300H o `-j` sem número mata o `cc1plus` por falta de RAM (`Killed signal terminated program cc1plus`): a CPU não tem AVX-512, FMA4 nem XOP, e cada variante dessas ocupa muita memória. O build que funcionou desliga essas variantes e usa dois processos:

```bash
git clone --recursive https://github.com/Tencent/ncnn.git
cd ncnn
git checkout 24ea07e87f544a3fbc59980b57885a92c75f2d24
cd ..
cmake -S ncnn -B ncnn/build -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF \
    -DNCNN_AVX512=OFF -DNCNN_AVX512VNNI=OFF -DNCNN_AVX512BF16=OFF \
    -DNCNN_AVX512FP16=OFF -DNCNN_AVXVNNI=OFF -DNCNN_AVXVNNIINT8=OFF \
    -DNCNN_AVXNECONVERT=OFF -DNCNN_FMA4=OFF -DNCNN_XOP=OFF \
    -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF
cmake --build ncnn/build -j2
```

As fotos de calibração são frames da trena, de preferência da mesma câmera. O padrão é `get_data/dataset/images`.

```bash
source .venv/bin/activate
python optimization/quantize_ncnn.py --ncnn-dir ncnn --images get_data/dataset/images
```

`--ncnn-dir` aponta para o clone quando `onnx2ncnn` não está no `PATH`. O script procura `ncnnoptimize`, `ncnn2table` e `ncnn2int8` em `ncnn/build/tools`.

## Inferência na Raspberry

O `run/main.py --mode fp32` não usa PyTorch. Ele abre `trena-opt.param` e `trena-opt.bin`, procura primeiro em `run/` e depois em `optimization/out/`, e lê o pré-processamento de `run/trena.yml`. `--mode int8` faz o mesmo com `trena-int8.param` e `trena-int8.bin`. O modo padrão, `--mode float`, continua no `model_best.ckpt`.

A placa desta seção é o Raspberry Pi 3, com Raspberry Pi OS de 64 bits. `uname -m` tem que imprimir `aarch64`.

### 1. Swap e pacotes

O Pi 3 tem 1 GB de RAM. Compilar o NCNN nessa memória precisa de swap, no mesmo esquema do setup do PyTorch em `run/README.md`:

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

O OpenCV do pip nessa placa pode morrer com `Illegal instruction`. Use o OpenCV do sistema e uma venv que o enxerga:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev python3-opencv git cmake g++ libgl1 libglib2.0-0
```

A partir da pasta `vision` do repositório, já copiada para a placa, a venv fica em `run/`, de onde o `main.py` roda:

```bash
python3 -m venv --system-site-packages run/.venv
source run/.venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pyyaml
```

Confira o OpenCV antes de seguir:

```bash
python -c "import cv2; print(cv2.__version__)"
```

### 2. NCNN na mesma versão do PC

O módulo Python tem que nascer do mesmo commit que quantizou o modelo. O submódulo do pybind11 entra no clone recursivo.

```bash
git clone --recursive https://github.com/Tencent/ncnn.git
cd ncnn
git checkout 24ea07e87f544a3fbc59980b57885a92c75f2d24
cd ..
```

No Pi não existem as flags AVX do PC. As tools também ficam de fora: o conversor de Caffe pede o `protoc`, e a inferência não usa essas ferramentas. O CMake não executa o symlink `run/.venv/bin/python`; `readlink -f` entrega o interpretador do sistema em que essa venv se apoia (`/usr/bin/python3.13`). Um processo só evita o compilador morrer por falta de RAM. Rode a partir de `vision`:

```bash
cmake -S ncnn -B ncnn/build -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_PYTHON=ON \
    -DPython_EXECUTABLE="$(readlink -f run/.venv/bin/python)" \
    -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF
cmake --build ncnn/build --target pyncnn -j1
```

O alvo `pyncnn` gera `ncnn/python/ncnn/ncnn.cpython-*.so`. Copie esse arquivo para o `site-packages` da venv, para o `import ncnn` achar `Net` quando o `main.py` roda de dentro de `run/`:

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp ncnn/python/ncnn/ncnn.cpython-*.so "$SITE/"
python -c "import ncnn; print(ncnn.Net)"
```

### 3. Modelos e câmera

Copie do PC o par fp32 otimizado. Os dois arquivos podem ficar em `run/` ou em `optimization/out/`:

```bash
scp trena-opt.param trena-opt.bin PI:vision/run/
```

`run/trena.yml` já vai com o repositório. Sem ele o detector não lê a média, o desvio e o tamanho de entrada.

```bash
cd run
python main.py --mode fp32
```

`q` na janela encerra. A legenda mostra `fp32` ao lado do fps. A câmera USB costuma ser o índice 0 (`CAMERA_INDEX` em `run/config.py`).

No PC, esse fp32 reproduz o score do PyTorch. O int8, nas mesmas fotos, ficou com score médio 0,20 e não passou do corte `SCORE_MIN` 0,5, então `--mode int8` não desenha caixa.
