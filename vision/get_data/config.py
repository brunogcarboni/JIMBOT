"""Limiares do rastreamento da trena. Ajuste olhando a janela de debug."""

# Webcam
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# A cor só propõe. Pele e plástico vermelho caem no mesmo matiz;
# o que separa é o excesso de R sobre G e B, não a saturação sozinha.
HUE_MAX = 15
HUE_MIN_WRAP = 165
EXCESS_RED_MIN = 0.45
BLUR_KSIZE = 5
DETAIL_BLUR_KSIZE = 3
MORPH_KSIZE = 3

# Forma da carcaça (escala invariante).
MIN_AREA_PX = 120
MIN_AREA_RATIO = 0.0004
CIRCULARITY_MIN = 0.45
CIRCULARITY_MAX = 0.98
ASPECT_MAX = 1.45
SOLIDITY_MIN = 0.72

# Grade e barra só são exigidas quando a caixa é grande o bastante
# para a textura existir em pixels. Longe, a frequência entra no score.
CANNY_LOW = 25
CANNY_HIGH = 80
GRID_DENSITY_MIN = 0.04
BAR_RATIO_MIN = 1.05
STRUCTURE_MIN_SIDE = 110
CENTER_BAND_RATIO = 0.14
INNER_ERODE_RATIO = 0.12
BAR_KERNEL_WIDTH_RATIO = 0.06
BAR_KERNEL_HEIGHT_RATIO = 0.35
FREQ_MIN = 0.25

# Pontuação. A proximidade do último centroide só desempata.
WEIGHT_CIRCULARITY = 0.25
WEIGHT_GRID = 0.20
WEIGHT_BAR = 0.15
WEIGHT_SOLIDITY = 0.15
WEIGHT_FREQ = 0.25
GRID_DENSITY_REF = 0.12
BAR_RATIO_REF = 2.0
FREQ_REF = 6.0
PROXIMITY_PENALTY = 0.15

# Rastro
HISTORY = 100
SMOOTH_N = 5
LOST_AFTER = 6
EXIT_DX_PX = 12
EXIT_EDGE_NX = 0.45

# Coleta para o NanoDet. img_path = dataset/images, ann_path = dataset/annotations/instances.json
DATASET_DIR = "dataset"
DATASET_TARGET = 250
DATASET_FRAME_INTERVAL = 15

# Comando. nx = (cx - largura/2) / (largura/2), em [-1, 1].
CENTER_DEADZONE = 0.18
# Fração da largura do quadro ocupada pela caixa.
NEAR_ENTER = 0.35
NEAR_EXIT = 0.28
