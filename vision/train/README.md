# Treino

O detector é o NanoDet-Plus-m 320, com backbone ShuffleNetV2 1.0x. É um modelo leve, feito para celular e placa, e cabe no Raspberry Pi, o hardware escolhido para o processamento do robô.

O notebook `train_nanodet.ipynb` lê o COCO em `../get_data/dataset`, separa 70% treino, 15% validação e 15% teste, aplica aumento de escala, posição, espelho, inclinação e luz, e treina a partir dos pesos do COCO. A cabeça de 80 classes é descartada. A classe treinada é a trena.
