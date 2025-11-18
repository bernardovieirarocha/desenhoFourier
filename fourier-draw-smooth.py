#!/usr/bin/env python3
# coding: utf-8

# DESENHO COM TRANSFORMADA DE FOURIER 

import argparse
from itertools import permutations
from math import tau

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad_vec

# CONFIGURAÇÃO DE PARÂMETROS

ap = argparse.ArgumentParser(
    description="Cria uma animação de desenho usando Transformada de Fourier"
)
ap.add_argument("-i", "--input", required=True, type=str,
    help="Caminho da imagem de entrada")
ap.add_argument("-o", "--output", required=True, type=str,
    help="Caminho do arquivo de animação de saída")
ap.add_argument("-f", "--frames", required=False, default=300, type=int,
    help="Número de quadros da animação (padrão: 300)")
ap.add_argument("-N", "--N", required=False, default=300, type=int,
    help="Número de coeficientes de Fourier (padrão: 300)")
ap.add_argument("-mc", "--max-contours", required=False, default=None, type=int,
    help="Máximo de contornos a usar (padrão: todos)")
ap.add_argument("-ms", "--min-size", required=False, default=10, type=int,
    help="Tamanho mínimo de contorno para incluir (padrão: 10)")
args = vars(ap.parse_args())

# Validação: precisamos de mais coeficientes que quadros
assert args["N"] >= args["frames"], \
    "Erro: coeficientes >= quadros. Aumente -N ou diminua -f"

print("Carregando imagem...")

# PROCESSAMENTO DA IMAGEM

# Carrega a imagem
img = cv2.imread(args["input"])

# Converte para escala de cinza
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar desfoque para suavizar e remover ruído de alta frequência
# Isso evita linhas serrilhadas nos bordos
blurred = cv2.GaussianBlur(imgray, (7, 7), 0)

# Converte para imagem binária (preto e branco)
# THRESH_OTSU calcula automaticamente o melhor valor de limiar
(T, thresh) = cv2.threshold(blurred, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# EXTRAÇÃO DE CONTORNOS

# Encontra os contornos (bordos) na imagem
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filtra contornos muito pequenos (ruído)
min_size = args["min_size"]
contours = [c for c in contours if len(c) >= min_size]

# Ordena por tamanho (maior primeiro)
contours = sorted(contours, key=len, reverse=True)

print(f"✓ Encontrados {len(contours)} contornos (tamanho mín: {min_size})")

def get_contour_center(contour):
    """Calcula o ponto central de um contorno"""
    coords = np.array(contour)
    return np.mean(coords, axis=0)

def get_contour_endpoints(contour):
    """Retorna os dois extremos de um contorno (início e fim)"""
    return contour[0], contour[-1]

def distance(p1, p2):
    """Calcula a distância Euclidiana entre dois pontos"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def connect_contours_optimal(contours, max_contours=None):
    """
    Conecta múltiplos contornos usando um algoritmo guloso otimizado.
    
    Este algoritmo evita que a linha "volte pra trás" ao conectar os contornos,
    resultando em uma animação muito mais limpa e bonita.
    
    Estratégia:
    1. Começa com o maior contorno
    2. Para cada próximo contorno, encontra qual está mais perto
    3. Reverte contornos quando necessário
    4. Adiciona pontos de transição suave entre contornos
    
    Argumentos:
        contours: lista de contornos do OpenCV
        max_contours: limita quantos contornos usar (None = todos)
    
    Retorna:
        Lista de tuplas (x, y) representando o contorno conectado
    """
    
    if max_contours:
        contours = contours[:max_contours]
    
    # Converte todos os contornos para listas de coordenadas
    all_verts = []
    for contour in contours:
        verts = [tuple(coord[0]) for coord in contour]
        if len(verts) > 1:
            all_verts.append(verts)
    
    if not all_verts:
        raise ValueError("Nenhum contorno disponível!")
    
    if len(all_verts) == 1:
        return all_verts[0]
    
    # Começa com o maior contorno
    ordered_indices = [0]
    remaining_indices = list(range(1, len(all_verts)))
    
    # Ponto atual é o fim do primeiro contorno
    current_point = all_verts[0][-1]
    
    # Conecta os contornos de forma gulosa (nearest neighbor)
    while remaining_indices:
        best_idx_pos = 0
        best_idx = remaining_indices[0]
        best_dist = float('inf')
        best_reversed = False
        
        # Procura o contorno mais próximo
        for idx_pos, idx in enumerate(remaining_indices):
            contour = all_verts[idx]
            
            # Calcula distância até o início e até o fim do contorno
            dist_to_start = distance(current_point, contour[0])
            dist_to_end = distance(current_point, contour[-1])
            
            # Prefere conectar no ponto mais próximo
            if dist_to_start <= dist_to_end:
                if dist_to_start < best_dist:
                    best_dist = dist_to_start
                    best_idx_pos = idx_pos
                    best_idx = idx
                    best_reversed = False
            else:
                if dist_to_end < best_dist:
                    best_dist = dist_to_end
                    best_idx_pos = idx_pos
                    best_idx = idx
                    best_reversed = True
        
        # Pega o contorno selecionado
        next_contour = all_verts[best_idx]
        if best_reversed:
            # Se for mais perto pelo fim, reverte o contorno
            next_contour = next_contour[::-1]
        
        # Cria uma ponte suave entre contornos (evita saltos grandes)
        next_start = next_contour[0]
        if best_dist > 2:
            # Interpola pontos entre o fim de um contorno e o início do próximo
            n_interp = max(2, int(best_dist / 5))
            for t in np.linspace(0, 1, n_interp)[:-1]:
                bridge_point = (
                    int(current_point[0] * (1 - t) + next_start[0] * t),
                    int(current_point[1] * (1 - t) + next_start[1] * t)
                )
                all_verts.append([bridge_point])
        
        # Atualiza o estado para o próximo contorno
        ordered_indices.append(best_idx)
        remaining_indices.pop(best_idx_pos)
        current_point = next_contour[-1]
    
    # Reconstrói a lista ordenada e achata em um único array
    result = []
    for idx in ordered_indices:
        result.extend(all_verts[idx])
    
    return result

# Conecta todos os contornos de forma otimizada
print("Conectando contornos...")
verts = connect_contours_optimal(contours, max_contours=args["max_contours"])
print(f"Total de pontos: {len(verts)}")

# Extrai coordenadas x e y
xs, ys = zip(*verts)

# Centraliza o contorno na origem (0, 0)
xs = np.asarray(xs) - np.mean(xs)
ys = - np.asarray(ys) + np.mean(ys)

# Cria um vetor de tempo parametrizado de 0 a 2π
# Isso mapeia cada ponto do contorno para um ângulo
t_list = np.linspace(0, tau, len(xs))

# CÁLCULO DOS COEFICIENTES DE FOURIER

# A Série de Fourier decompõe qualquer contorno contínuo em uma soma infinita
# de ondas senoidais. Aqui, calculamos numericamente os "coeficientes" c_n
# que definem a amplitude e fase de cada onda.

def f(t, t_list, xs, ys):
    """
    Interpola os pontos do contorno de forma contínua.
    
    Para um valor de tempo 't' (entre 0 e 2π), retorna a coordenada complexa
    (x + iy) do ponto correspondente no contorno.
    """
    return np.interp(t, t_list, xs + 1j*ys)


def compute_cn(f, n):
    """
    Calcula o n-ésimo coeficiente de Fourier usando integração numérica.
    
    Cada coeficiente c_n representa a "força" de um círculo específico que
    vai girar numa determinada frequência.
    
    Argumentos:
        f: função de interpolação do contorno
        n: índice do coeficiente (frequência)
    
    Retorna:
        Número complexo representando o coeficiente c_n
    """
    coef = 1/tau*quad_vec(
        lambda t: f(t, t_list, xs, ys)*np.exp(-n*t*1j), 
        0, 
        tau, 
        limit=100,  # Número de iterações da integração numérica
        full_output=False)[0]
    return coef

def get_circle_coords(center, r, N=50):
    """
    Gera as coordenadas de um círculo para visualização.
    
    Argumentos:
        center: coordenada (x, y) do centro
        r: raio do círculo
        N: número de pontos para desenhar o círculo (padrão: 50)
    
    Retorna:
        Duas listas: x e y com as coordenadas dos pontos no círculo
    """
    theta = np.linspace(0, tau, N)
    x, y = center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)
    return x, y

def get_next_pos(c, fr, t, drawing_time = 1):
    """
    Calcula a posição de um epicíclo em um determinado tempo.
    
    Um epicíclo é simplesmente um vetor que gira. Este vetor representa
    um círculo no nosso desenho.
    
    Argumentos:
        c: coeficiente complexo do epicíclo (amplitude)
        fr: frequência (quantas voltas por segundo)
        t: tempo atual
        drawing_time: tempo total do desenho (padrão: 1 segundo)
    
    Retorna:
        Vetor complexo representando o vetor rotacionado
    """
    angle = (fr * tau * t) / drawing_time
    return c * np.exp(1j*angle)


print("Calculando coeficientes de Fourier...")

N = args["N"]
# Calcula os coeficientes para frequências de -N até +N
# A frequência 0 é o centro médio, frequências positivas giram clockwise,
# frequências negativas giram counter-clockwise
coefs = [ (compute_cn(f, 0), 0) ] + \
        [ (compute_cn(f, j), j) for i in range(1, N+1) for j in (i, -i) ]

print(f"✓ {len(coefs)} coeficientes calculados")

# CRIAÇÃO DA ANIMAÇÃO

print("Gerando animação...")

# Cria uma figura em branco para a animação
fig, ax = plt.subplots()

# Cria elementos gráficos para visualizar os epicíclos
circles = [ax.plot([], [], 'b-')[0] for i in range(-N, N+1)]
circle_lines = [ax.plot([], [], 'g-')[0] for i in range(-N, N+1)]
drawing, = ax.plot([], [], 'r-', linewidth=2)  # Em vermelho, o desenho final

# Define os limites da janela (quadrado de -500 a 500)
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)

# Remove os eixos para um visual mais limpo
ax.set_axis_off()
ax.set_aspect('equal')  # Garante que os círculos pareçam circulares
fig.set_size_inches(15, 15)

# Arrays para armazenar o caminho do desenho
draw_x, draw_y = [], []

def animate(i, coefs, time): 
    """
    Função chamada para cada quadro da animação.
    
    Desenha todos os epicíclos e calcula o ponto final do último círculo.
    """
    # Tempo atual (entre 0 e 1)
    t = time[i]
    
    # Rotaciona todos os epicíclos para este instante de tempo
    coefs = [ (get_next_pos(c, fr, t=t), fr) for c, fr in coefs ]
    
    # Começa no centro
    center = (0, 0)
    
    # Desenha cada círculo (epicíclo)
    for i, elts in enumerate(coefs) :
        c, _ = elts
        r = np.linalg.norm(c)  # Raio é a magnitude do vetor complexo
        
        # Gera os pontos do círculo
        x, y = get_circle_coords(center=center, r=r, N=80)
        
        # Desenha o círculo (em azul)
        circle_lines[i].set_data([center[0], center[0]+np.real(c)], 
                                  [center[1], center[1]+np.imag(c)])
        circles[i].set_data(x, y) 
        
        # Atualiza o centro para o fim deste círculo
        center = (center[0] + np.real(c), center[1] + np.imag(c))
    
    # O ponto final é o centro do último círculo
    # Este ponto desenha o padrão final
    draw_x.append(center[0])
    draw_y.append(center[1])

    # Atualiza o gráfico do desenho final
    drawing.set_data(draw_x, draw_y)

# EXECUÇÃO DA ANIMAÇÃO

# Tempo total da animação (em segundos)
drawing_time = 1

# Número de quadros
frames = args["frames"]

# Vetor de tempos para cada quadro
time = np.linspace(0, drawing_time, num=frames)

# Cria a animação chamando a função 'animate' para cada quadro
anim = animation.FuncAnimation(
    fig, animate, 
    frames=frames, 
    interval=5,  # Intervalo entre quadros em milissegundos
    fargs=(coefs, time)
)

# SALVAMENTO DO ARQUIVO

print(f"Salvando em: {args['output']}")
anim.save(args["output"], fps=15)

print("Pronto! Animação criada com sucesso!")
