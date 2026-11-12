# Inferência e rumo

`main.py` abre a câmera, passa cada frame no NanoDet-Plus-m 320 e imprime o rumo da trena. A janela mostra a caixa e os últimos 50 centros. O terminal só escreve o rótulo quando ele muda. `q` na janela encerra.

A caixa grande, a partir de `NEAR_ENTER` da largura do quadro, deixa o comando em `parado`, mesmo com a trena fora do centro. Abaixo de `NEAR_EXIT` a navegação volta: centro `reto`, direita `reto-direita`, esquerda `reto-esquerda`. Se a trena sai do quadro, o giro continua para o lado da saída (`buscando-esquerda` ou `buscando-direita`) até uma detecção nova. Sem histórico, o rótulo é `perdido`. Alguns frames vazios não disparam a busca.

Os limiares, o índice da câmera e o score mínimo (`SCORE_MIN`) estão em `config.py`. A arquitetura é `trena.yml`, nesta pasta. O peso `model_best.ckpt` não entra no Git: o link para baixar está em `model_weight.txt`. Salve o arquivo baixado como `model_best.ckpt` aqui, ao lado do `main.py`, antes de rodar.

## Setup no Raspberry Pi 3

O sistema precisa ser o Raspberry Pi OS de 64 bits (Bookworm ou Trixie). O Pi 3 roda essa imagem. `uname -m` tem que imprimir `aarch64`. No sistema de 32 bits não existe wheel do PyTorch.

O Pi 3 tem 1 GB de RAM. O pacote do PyTorch não cabe nessa memória durante a instalação. Aumente a swap antes de instalar:

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

Pacotes de sistema e a venv, a partir da pasta `vision` do repositório.

O wheel oficial do PyTorch, a partir da 2.4, inclui a instrução `ldaddal`. O Cortex-A53 do Pi 3 não tem essa instrução, e `import torch` morre com `Illegal instruction`. O 2.3.1 era a última versão oficial sem essa instrução, mas o wheel só existe para Python 3.8–3.12. No Trixie o sistema é Python 3.13, e o pip só oferece da 2.6.0+cpu para cima. O pacote do apt é a build do Debian, sem exigir `ldaddal`. No Bookworm ele entra como 1.13.1; no Trixie, como 2.6.0.

A venv precisa de `--system-site-packages` para enxergar o torch do apt. Sem essa flag o `import torch` falha mesmo com o pacote instalado. Se a `.venv` já foi criada sem a flag, apague a pasta e crie de novo.

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev git libgl1 libglib2.0-0 python3-torch python3-torchvision
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Não instale `torch` pelo pip nesta placa. O wheel substitui o pacote do apt e o `import` volta a morrer com `Illegal instruction`.

No torch 2.6 do Trixie, `torch.load` recusa o checkpoint da trena. O `main.py` sobe com a variável abaixo. No 1.13 do Bookworm ela não faz diferença.

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

O clone do NanoDet fica em `third_party/nanodet` e não entra no Git. A instalação é sem as dependências do projeto, porque o `requirements.txt` de lá prende `torch<2` e substituiria o torch do apt.

```bash
git clone --depth 1 https://github.com/RangiLyu/nanodet.git third_party/nanodet
pip install -e third_party/nanodet --no-deps
sed -i 's/from torch._six import string_classes/string_classes = (str, bytes)/' third_party/nanodet/nanodet/data/collate.py
```

O `torch._six` saiu no PyTorch 2. O `collate.py` do NanoDet ainda importa esse módulo, e a inferência quebra sem essa troca.

Confira se a CPU aceitou o torch e suba a câmera. A primeira carga baixa os pesos do ShuffleNetV2 (cerca de 9 MB) e precisa de rede. O checkpoint da trena substitui esses pesos em seguida.

```bash
python -c "import torch; print(torch.__version__); print(torch.zeros(2))"
cd run
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python main.py
```

A câmera USB costuma ser o índice 0, o valor de `CAMERA_INDEX`. A janela do OpenCV abre na sessão gráfica do Pi. Se `import cv2` também morrer com `Illegal instruction`, o wheel do OpenCV veio com instruções novas: apague a venv, crie outra com `python3 -m venv --system-site-packages .venv` e use o OpenCV do sistema (`sudo apt install python3-opencv`) no lugar da linha `opencv-python` do `requirements.txt`.
