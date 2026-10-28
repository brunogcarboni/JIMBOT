# Coleta do dataset

Este pipeline filma o objeto com a webcam e grava um dataset COCO para o NanoDet. A cor só propõe regiões. A forma compacta e a textura interna confirmam qual mancha é o objeto. Cada detecção salva o frame cru e a caixa, espera 15 frames e segue, até 250 imagens.

Rode de dentro desta pasta:

```bash
python main.py
```

As fotos ficam em `dataset/images` e o rótulo em `dataset/annotations/instances.json`. A janela mostra a proposta de cor, a forma aceita e o contador `n/250`. `q` encerra antes. Uma nova execução continua de onde parou.

Os limiares estão em `config.py`. Os valores atuais foram ajustados para a trena vermelha, um disco com grade no centro.

## Parâmetros que mudam com o objeto

`EXCESS_RED_MIN`, `HUE_MAX` e `HUE_MIN_WRAP` definem o quão vermelho um pixel precisa ser para entrar na proposta. Subir `EXCESS_RED_MIN` corta pele, madeira e outros vermelhos fracos, e pode perder o objeto debaixo de lâmpada fluorescente. Descer inclui mais fundo. Para um objeto que não é vermelho, estes intervalos deixam de servir: o matiz e o canal dominante precisam ser os da cor nova.

`CIRCULARITY_MIN`, `ASPECT_MAX` e `SOLIDITY_MIN` descrevem a forma. A trena passa porque é quase um disco, cheia, sem buracos. Um objeto alongado pede `CIRCULARITY_MIN` menor e `ASPECT_MAX` maior. `SOLIDITY_MIN` mais baixo aceita um contorno mais recortado. Subir esses cortes deixa a captura mais exigente e perde o objeto quando a mão tapa a borda.

`MIN_AREA_PX` é o menor blob, em pixels, que ainda conta. Descer permite o objeto longe. Subir ignora ruído pequeno e também o objeto distante.

`FREQ_MIN` exige textura dentro da mancha. A grade da trena produz essa energia. Um objeto liso pede um valor menor. Subir demais rejeita o objeto quando ele ocupa poucos pixels.

`DATASET_TARGET` é quantas imagens gravar. `DATASET_FRAME_INTERVAL` é quantos frames esperar entre duas gravações. Um intervalo maior espalha as fotos no tempo. Um menor repete quase o mesmo quadro.
