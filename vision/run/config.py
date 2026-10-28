"""Limiares do rumo. A caixa é fração da largura do quadro."""

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MODEL_PATH = "model_best.ckpt"
CONFIG_PATH = "trena.yml"
SCORE_MIN = 0.5

# nx = (cx - largura/2) / (largura/2). Dentro da zona, o rumo é reto.
CENTER_DEADZONE = 0.18

# Acima de NEAR_ENTER a trena está perto e o comando fica parado.
# Só volta a navegar abaixo de NEAR_EXIT.
NEAR_ENTER = 0.35
NEAR_EXIT = 0.28

HISTORY = 50
SMOOTH_N = 5
LOST_AFTER = 6
EXIT_DX_PX = 12
EXIT_EDGE_NX = 0.45
