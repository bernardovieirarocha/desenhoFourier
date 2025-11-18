# Fourier Draw

Converte qualquer imagem em uma animação usando Série de Fourier. O resultado é uma bela visualização de epicíclos (círculos em rotação) que desenham o contorno da imagem.

## Como funciona

A transformada de Fourier decompõe o contorno de uma imagem em uma soma de ondas senoidais. Cada onda é representada como um círculo rotativo em uma determinada frequência. Quando você anima esses círculos rotacionando juntos, eles desenham o padrão original.

## Requisitos

- Python 3.8+
- opencv-python
- matplotlib
- numpy
- scipy
- cairosvg (para converter SVG para PNG)

## Instalação

```bash
git clone https://github.com/staghado/fourier-draw.git
cd fourier-draw

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Uso

Primeiro, prepare sua imagem. Se for um SVG, converta para PNG:

```bash
cairosvg logo.svg -o logo.png
```

Depois, execute o script:

```bash
python3 fourier-draw-smooth.py -i logo.png -o animacao.gif
```

## Parâmetros

- `-i, --input`: Caminho da imagem de entrada (obrigatório)
- `-o, --output`: Caminho do arquivo de saída (obrigatório)
- `-f, --frames`: Número de quadros (padrão: 300)
- `-N`: Número de coeficientes de Fourier (padrão: 300)
- `-mc, --max-contours`: Máximo de contornos a usar
- `-ms, --min-size`: Tamanho mínimo de contorno para incluir

## Exemplos

```bash
# Animação rápida com menos detalhes
python3 fourier-draw-smooth.py -i logo.png -o rápida.gif -f 100 -N 100

# Animação detalhada com mais coeficientes
python3 fourier-draw-smooth.py -i logo.png -o detalhada.gif -f 500 -N 500

# Removendo ruído pequeno
python3 fourier-draw-smooth.py -i logo.png -o limpa.gif -ms 20

# Limitando a apenas os maiores contornos
python3 fourier-draw-smooth.py -i logo.png -o simples.gif -mc 5
```

## Qualidade da Animação

A qualidade depende de vários fatores:

- **Número de coeficientes (-N)**: Mais coeficientes = mais detalhes, mas mais lento
- **Número de quadros (-f)**: Mais quadros = animação mais suave
- **Tamanho mínimo de contorno (-ms)**: Aumentar remove ruído e detalhes muito pequenos
- **Máximo de contornos (-mc)**: Limitar pode melhorar o desempenho

## Notas Técnicas

O script usa um algoritmo guloso de vizinho mais próximo para ordenar os contornos. Isso minimiza grandes saltos entre contornos, resultando em uma animação mais limpa sem cruzamentos desnecessários.

A escolha do tamanho mínimo de contorno é importante: muito pequeno deixa ruído, muito grande perde detalhes.

## Referências

Baseado na série de Fourier e epicíclos. Conceito popularizado por:
- [3Blue1Brown - But what is a Fourier series?](https://www.youtube.com/watch?v=r6sGWTCMz2k)

## Autor

Feito com interesse em matemática, visualização e desenho criativo.
