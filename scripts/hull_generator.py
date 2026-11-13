# -*- coding: utf-8 -*-
"""
Параметрический генератор корпуса контейнеровоза типа "Emma Maersk" (PS-class).
Моделируется ТОЛЬКО корпус: обшивка, днище, палуба, транцевая плита,
бульбообразный нос. Надстройка, мостик, люковые закрытия и палубные
устройства не моделируются (по условию тестового задания).

Масштаб вывода: 1:100 (единицы OBJ - метры, длина модели 3.9771 м).

КАК УСТРОЕНА ГЕОМЕТРИЯ
----------------------
Корпус задаётся семейством ватерлиний. Каждая ватерлиния (уровень z) - это
кривая полушироты y(x), состоящая из трёх участков:
    * кормовая ветвь  - кубическая кривая Безье от транца до цилиндрической вставки,
    * цилиндрическая вставка - постоянная полуширота b(z),
    * носовая ветвь   - кубическая кривая Безье от вставки до точки форштевня.
Безье выбран потому, что даёт C1-стык со вставкой (касательная горизонтальна)
и прямой контроль формы: "полнота" ветви и "острота" оконечности - отдельные
параметры. Управляющие точки x монотонны, поэтому x(t) монотонна и поверхность
не складывается.

Транец - настоящий: ниже высоты Z_TR полуширота транца равна нулю, т.е.
подводная кормовая оконечность плавно сходится к диаметральной плоскости
(замкнутая корма), а транцевая плита существует только выше Z_TR.

Поверхность собирается из 4 патчей с общими вершинами на стыках:
    1) обшивка (лофт ватерлиний), 2) днище (z=0), 3) палуба (z=D),
    4) транцевая плита (z = Z_TR..D).
После сварки дублей получается замкнутый (watertight) квадовый меш.

Запуск (headless):
    blender -b --factory-startup -P hull_generator.py -- --out <папка> [--render] [--report]
"""

import bpy
import bmesh
import math
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# 1. ГЛАВНЫЕ РАЗМЕРЕНИЯ (натуральная величина, метры)
# ----------------------------------------------------------------------------
LOA = 397.71     # длина наибольшая (по бульбу)                    м
B = 56.40        # ширина                                           м
D = 30.20        # высота борта (ОЛ -> палуба)                      м
T = 16.00        # осадка (расчётная/летняя)                        м
LBP = 376.00     # длина между перпендикулярами (справочно)         м
SCALE = 0.01     # 1:100

Z_TR = 13.5      # высота нижней кромки транца (knuckle), м

# --- фальшборт по периметру палубы -----------------------------------------
BULWARK_H = 1.20      # высота фальшборта над палубой, м
BULWARK_RAKE = 0.10   # завал внутрь к планширю, м
BULWARK_TOP = 0.22    # ширина планширя, м
BULWARK_SINK = 0.03   # утопление в палубу, чтобы не было щели, м

# --- комингсы грузовых люков (палубная «начинка» контейнеровоза) -----------
HATCH_H = 1.20        # высота комингса над палубой, м
HATCH_WALL = 0.28     # толщина стенки комингса, м
HATCH_BAYS = 10       # число люков по длине
HATCH_LEN = 23.0      # длина люка, м
HATCH_GAP = 2.6       # мéжлюковый промежуток, м
HATCH_X0_FRAC = 0.165 # начало грузовой палубы (доля LOA)
HATCH_MARGIN = 1.4    # отступ от фальшборта, м
HATCH_WALKWAY = 2.4   # проход по ДП между люками, м

# --- внутренний набор (начинка корпуса) ------------------------------------
DB_HEIGHT = 2.6       # высота второго дна (внутреннее дно над ОЛ), м
DB_THICK = 0.30       # толщина настила второго дна, м
BH_THICK = 0.25       # толщина поперечной переборки, м
BH_COUNT = 10         # число поперечных переборок по грузовой части

# ----------------------------------------------------------------------------
# 2. ПАРАМЕТРЫ ФОРМЫ.  Контрольные точки: (список z, список значений)
# ----------------------------------------------------------------------------
PARAMS = {
    # максимальная полуширота ватерлинии на уровне z (скула -> борт)
    "b":      ([0.0, 2.0, 4.0, 6.0, 8.0, 16.0, D],
               [18.4, 23.4, 26.2, 27.60, 28.20, 28.20, 28.20]),
    # профиль носовой оконечности: линия форштевня и бульба
    "xf":     ([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.5, 13.0, 14.0, 16.0, 18.0, 22.0, D],
               [320.0, 348.0, 366.0, 379.0, 388.0, 394.0, LOA, 396.0, 392.0, 388.0,
                387.6, 387.9, 388.4]),
    # профиль кормовой оконечности (линия транца / подзора)
    "xa":     ([0.0, 4.0, 8.0, Z_TR, 17.0, 20.0, 24.0, D],
               [14.0, 11.0, 8.0, 6.0, 3.5, 2.2, 1.0, 0.0]),
    # полуширота транца / b  (ниже Z_TR транец отсутствует)
    "r":      ([0.0, Z_TR - 0.1, Z_TR, 15.5, 18.0, 22.0, D],
               [0.0, 0.0, 0.0, 0.20, 0.44, 0.70, 0.84]),
    # начало носового заострения (доля LOA)
    "ff":     ([0.0, 8.0, 16.0, D], [0.590, 0.605, 0.620, 0.630]),
    # начало кормового заострения (доля LOA)
    "fa":     ([0.0, 8.0, 16.0, D], [0.255, 0.270, 0.276, 0.256]),
    # кормовая ветвь: x-плечо Безье у транца и подъём полушироты у транца
    "k1":     ([0.0, 8.0, 16.0, D], [0.58, 0.56, 0.52, 0.42]),
    "m1":     ([0.0, 8.0, 16.0, D], [0.04, 0.05, 0.06, 0.26]),
    # носовая ветвь: x-плечо у вставки, x-плечо у форштевня,
    # полнота в конце ветви (c) - управляет формой и остротой носа
    "a1":     ([0.0, 8.0, 12.0, 16.0, D], [0.26, 0.30, 0.35, 0.41, 0.47]),
    "a2":     ([0.0, 8.0, 16.0, D], [0.14, 0.13, 0.13, 0.12]),
    "c":      ([0.0, 6.0, 9.0, 12.0, 14.0, 16.0, D],
               [0.32, 0.31, 0.30, 0.33, 0.40, 0.49, 0.56]),
}
K2, M2 = 0.32, 0.02          # кормовая ветвь: плечо у вставки и "прижим" к b

# плотность сетки
NZ = 112        # строк (уровней z)
NA = 44         # колонок в кормовой ветви
NM = 60         # колонок в цилиндрической вставке
NF = 108        # колонок в носовой ветви
NL = 21         # поперечных точек на палубе / днище / транце

# цвета (линейный sRGB)
COL_AFT = (0.150, 0.035, 0.030)   # противообрастающая окраска (ниже ВЛ)
COL_BOOT = (0.018, 0.018, 0.020)  # полоса переменной ватерлинии
COL_SIDE = (0.032, 0.045, 0.065)  # надводный борт
COL_DECK = (0.028, 0.036, 0.030)  # палуба (тёмный серо-зелёный)
COL_INTERNAL = (0.085, 0.090, 0.100)  # внутренний набор (серый металл)

BOOT_LO, BOOT_HI = T - 0.45, T + 0.15


# ----------------------------------------------------------------------------
# 3. ИНТЕРПОЛЯЦИЯ (монотонный кубический сплайн Fritsch-Carlson / PCHIP)
# ----------------------------------------------------------------------------
def pchip_factory(xk, yk):
    """f(x) - монотонная кубическая интерполяция точек (xk, yk), без выбросов."""
    xk = np.asarray(xk, float)
    yk = np.asarray(yk, float)
    n = len(xk)
    h = np.diff(xk)
    delta = np.diff(yk) / h
    m = np.zeros(n)

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    def _end(h0, h1, d0, d1):
        s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if s * d0 <= 0.0:
            s = 0.0
        elif d0 * d1 < 0.0 and abs(s) > abs(3.0 * d0):
            s = 3.0 * d0
        return s

    if n > 2:
        m[0] = _end(h[0], h[1], delta[0], delta[1])
        m[-1] = _end(h[-1], h[-2], delta[-1], delta[-2])
    else:
        m[0] = m[-1] = delta[0]

    def f(x):
        x = np.clip(np.asarray(x, float), xk[0], xk[-1])
        i = np.clip(np.searchsorted(xk, x) - 1, 0, n - 2)
        t = (x - xk[i]) / h[i]
        t2, t3 = t * t, t * t * t
        return (yk[i] * (2 * t3 - 3 * t2 + 1) + h[i] * m[i] * (t3 - 2 * t2 + t)
                + yk[i + 1] * (-2 * t3 + 3 * t2) + h[i] * m[i + 1] * (t3 - t2))

    return f


F = {k: pchip_factory(v[0], v[1]) for k, v in PARAMS.items()}

# седловатость палубы (sheer), м: подъём палубы к носу и корме
SHEER = pchip_factory(
    [0.0, 0.10 * LOA, 0.22 * LOA, 0.40 * LOA, 0.62 * LOA, 0.78 * LOA, 0.90 * LOA, LOA],
    [0.45, 0.25, 0.08, 0.00, 0.08, 0.42, 0.85, 1.30])
SHEER_P = 3.0      # степень: погибь затрагивает только верхнюю часть борта


def warp_z(x, z):
    """Седловатость: z -> z + sheer(x) * (z/D)^p (z=0 и ДП не меняются)."""
    return z + SHEER(x) * np.clip(z / D, 0.0, 1.0) ** SHEER_P


# ----------------------------------------------------------------------------
# 4. ТЕОРЕТИЧЕСКИЙ ЧЕРТЁЖ
# ----------------------------------------------------------------------------
def level_geom(z):
    """Параметры ватерлинии на уровне z."""
    zb = float(np.clip(z, 0.0, D))
    bb = float(F["b"](zb))
    xf = float(F["xf"](zb))
    xa = float(F["xa"](zb))
    wa = float(F["r"](zb)) * bb
    xa2 = float(np.clip(float(F["fa"](zb)) * LOA, xa + 4.0, xf - 8.0))
    xf2 = float(np.clip(float(F["ff"](zb)) * LOA, xa2 + 6.0, xf - 4.0))
    return dict(bb=bb, xf=xf, xa=xa, wa=wa, xa2=xa2, xf2=xf2,
                k1=float(F["k1"](zb)), m1=float(F["m1"](zb)),
                a1=float(F["a1"](zb)), a2=float(F["a2"](zb)), c=float(F["c"](zb)))


def bezier(P, t):
    """Кубическая кривая Безье. P: (4,2) контрольных точек, t: параметр 0..1."""
    t = np.asarray(t, float)
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * (1.0 - t) ** 2 * t
    b2 = 3.0 * (1.0 - t) * t ** 2
    b3 = t ** 3
    return (P[0][None, :] * b0[:, None] + P[1][None, :] * b1[:, None]
            + P[2][None, :] * b2[:, None] + P[3][None, :] * b3[:, None])


def cosspace(a, b, n):
    """n точек от a до b со сгущением к краям."""
    t = np.linspace(0.0, 1.0, n)
    return a + (b - a) * (1.0 - np.cos(np.pi * t)) / 2.0


def level_columns(z):
    """Абсциссы и полушироты колонок на уровне z (корма -> вставка -> нос)."""
    g = level_geom(z)
    bb, xa, xa2, xf2, xf = g["bb"], g["xa"], g["xa2"], g["xf2"], g["xf"]
    wa = g["wa"]

    # кормовая ветвь
    P_run = np.array([[xa, wa],
                      [xa + g["k1"] * (xa2 - xa), wa + g["m1"] * (bb - wa)],
                      [xa2 - K2 * (xa2 - xa), bb - M2 * (bb - wa)],
                      [xa2, bb]])
    t_run = cosspace(0.0, 1.0, NA + 1)
    run = bezier(P_run, t_run)

    # цилиндрическая вставка (без повторения стыков)
    t_mid = np.linspace(0.0, 1.0, NM + 1)[1:-1]
    mid = np.column_stack([xa2 + (xf2 - xa2) * t_mid, np.full(NM - 1, bb)])

    # носовая ветвь
    P_ent = np.array([[xf2, bb],
                      [xf2 + g["a1"] * (xf - xf2), bb],
                      [xf - g["a2"] * (xf - xf2), g["c"] * bb],
                      [xf, 0.0]])
    t_ent = cosspace(0.0, 1.0, NF + 1)[1:]
    ent = bezier(P_ent, t_ent)

    pts = np.vstack([run, mid, ent])
    pts[-1, 1] = 0.0                      # точка форштевня
    return pts[:, 0], pts[:, 1]


def half_breadth(x, z):
    """Полуширота в точке (x, z) - для контрольных расчётов. Векторизация по x."""
    xs, ys = level_columns(z)
    xq = np.atleast_1d(np.asarray(x, float))
    yq = np.interp(xq, xs, ys, left=0.0, right=0.0)
    return yq


# ----------------------------------------------------------------------------
# 5. СЕТКА
# ----------------------------------------------------------------------------
def level_rows():
    """Уровни z: косинусное сгущение + принудительные строки (ВЛ, кромка транца)."""
    z = cosspace(0.0, D, NZ + 1)
    forced = [BOOT_LO, T, BOOT_HI, Z_TR, 11.5]
    z = np.unique(np.round(np.concatenate([z, forced]), 6))
    return z


def add_uv_grid(obj, nrows, ncols, uv_fn):
    """UV-развёртка для патча-сетки (квады созданы в порядке row-major).

    uv_fn(ri, cj) -> (u, v). Каждый патч получает свою прямоугольную область
    в 0..1, поэтому острова не перекрываются и годятся для текстурирования.
    """
    me = obj.data
    layer = me.uv_layers.new(name="UVMap")
    corners = ((0, 0), (0, 1), (1, 1), (1, 0))
    k = 0
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            poly = me.polygons[k]
            k += 1
            for li, (di, dj) in zip(poly.loop_indices, corners):
                layer.data[li].uv = uv_fn(i + di, j + dj)
    return layer


def mesh_from_arrays(name, verts, faces):
    """Меш из готовых списков вершин/граней (для накладных конструкций)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate(verbose=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def set_uv_from_verts(obj, uvs):
    """UV по таблице «вершина -> (u, v)» (на каждую петлю - по её вершине)."""
    me = obj.data
    layer = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            layer.data[li].uv = uvs[me.loops[li].vertex_index]


def build_bulwark(top_row, transom_top):
    """Фальшборт: замкнутая оболочка (труба) по всему периметру палубы.

    Сечение: наружная стенка с завалом внутрь -> планширь -> внутренняя стенка.
    Нижняя кромка утоплена в палубу, поэтому щелей нет.
    Возвращает (verts, faces, uvs) в натуральных метрах.
    """
    path = [(p[0], p[1], p[2]) for p in top_row]                    # борт: корма -> нос -> борт
    path += [(p[0], p[1], p[2]) for p in transom_top[1:-1]]         # транец: замыкаем контур
    N = len(path)
    pts = np.array([[p[0], p[1]] for p in path])
    zs = np.array([p[2] for p in path])
    c = np.array([pts[:, 0].mean(), 0.0])                            # центр палубы (для знака нормали)

    verts, uvs = [], []
    for i in range(N):
        p = pts[i]
        t = pts[(i + 1) % N] - pts[(i - 1) % N]
        n = np.array([-t[1], t[0]])
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-9 else np.array([0.0, 1.0])
        if np.dot(n, c - p) < 0.0:
            n = -n
        z0 = zs[i] - BULWARK_SINK
        a = np.array([p[0], p[1], z0])                                   # низ наружный
        b = np.array([p[0] + BULWARK_RAKE * n[0], p[1] + BULWARK_RAKE * n[1],
                      zs[i] + BULWARK_H])                                # верх наружный
        cc = np.array([p[0] + (BULWARK_RAKE + BULWARK_TOP) * n[0],
                       p[1] + (BULWARK_RAKE + BULWARK_TOP) * n[1],
                       zs[i] + BULWARK_H])                                # верх внутренний
        dd = np.array([p[0] + (BULWARK_RAKE + BULWARK_TOP) * n[0],
                       p[1] + (BULWARK_RAKE + BULWARK_TOP) * n[1], z0])   # низ внутренний
        verts.extend([a, b, cc, dd])
        for k in range(4):
            uvs.append((i / N, 0.155 + 0.030 * (k / 3.0)))

    faces = []
    for i in range(N):
        j = (i + 1) % N
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((i * 4 + k, j * 4 + k, j * 4 + k2, i * 4 + k2))
    return verts, faces, uvs


def build_coamings(top_row):
    """Комингсы грузовых люков: 2 ряда по 10 люков, замкнутые «рамки» на палубе.

    Это та самая палубная «начинка» контейнеровоза - то, что на фото читается
    как система панелей на палубе. Низ комингса садится на палубу по 4 углам,
    верх горизонтальный, стенка толщиной HATCH_WALL.
    """
    xs = np.array([p[0] for p in top_row])
    ws = np.abs(np.array([p[1] for p in top_row]))
    zs = np.array([p[2] for p in top_row])
    half = len(xs) // 2 + 1
    order = np.argsort(xs[:half])
    x_half, w_half, z_half = xs[:half][order], ws[:half][order], zs[:half][order]

    def deck_z(x):
        return float(np.interp(x, x_half, z_half))

    def deck_w(x):
        return float(np.interp(x, x_half, w_half))

    verts, uvs, faces = [], [], []
    n_hatch = 0
    x_start = HATCH_X0_FRAC * LOA
    for b in range(HATCH_BAYS):
        x0 = x_start + b * (HATCH_LEN + HATCH_GAP)
        x1 = x0 + HATCH_LEN
        if x1 > 0.80 * LOA:
            break
        w_out = min(deck_w(x0), deck_w(x1)) - HATCH_MARGIN
        w_in = HATCH_WALKWAY / 2.0
        if w_out - w_in < 4.0:
            break
        for sgn in (1.0, -1.0):
            y_a, y_b = sorted((sgn * w_in, sgn * w_out))
            cx, cy = (x0 + x1) / 2.0, (y_a + y_b) / 2.0
            outer = [(x0, y_a), (x1, y_a), (x1, y_b), (x0, y_b)]
            inner = [(px + HATCH_WALL * (1 if px < cx else -1),
                      py + HATCH_WALL * (1 if py < cy else -1)) for (px, py) in outer]
            z_bot = [deck_z(px) - 0.03 for (px, _) in outer]
            z_top = max(z_bot) + HATCH_H
            base = len(verts)
            u0 = n_hatch / float(2 * HATCH_BAYS)
            du = 1.0 / float(2 * HATCH_BAYS)
            n_hatch += 1
            for k, ((px, py), zz) in enumerate(zip(outer, z_bot)):
                verts.append(np.array([px, py, zz]))
                uvs.append((u0 + du * k / 4.0, 0.300))
            for k, ((px, py), zz) in enumerate(zip(inner, z_bot)):
                verts.append(np.array([px, py, zz]))
                uvs.append((u0 + du * k / 4.0, 0.315))
            for k, (px, py) in enumerate(outer):
                verts.append(np.array([px, py, z_top]))
                uvs.append((u0 + du * k / 4.0, 0.330))
            for k, (px, py) in enumerate(inner):
                verts.append(np.array([px, py, z_top]))
                uvs.append((u0 + du * k / 4.0, 0.345))
            e0, i0, e1, i1 = base, base + 4, base + 8, base + 12
            for k in range(4):
                k2 = (k + 1) % 4
                faces.append((e0 + k, e0 + k2, e1 + k2, e1 + k))      # наружная стенка
                faces.append((i0 + k, i0 + k2, i1 + k2, i1 + k))      # внутренняя стенка
                faces.append((e1 + k, e1 + k2, i1 + k2, i1 + k))      # верхняя полка
                faces.append((e0 + k, e0 + k2, i0 + k2, i0 + k))      # низ (на палубе)
    return verts, faces, uvs


def tube_from_rings(rings, close=False):
    """Замкнутая пластина постоянного сечения: 4 точки на сечение -> труба."""
    verts = [p for ring in rings for p in ring]
    faces = []
    n = len(rings)
    span = n if close else n - 1
    for i in range(span):
        j = (i + 1) % n
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((i * 4 + k, j * 4 + k, j * 4 + k2, i * 4 + k2))
    if not close:
        faces.append((0, 1, 2, 3))
        m = (n - 1) * 4
        faces.append((m + 3, m + 2, m + 1, m + 0))
    return verts, faces


def build_internals():
    """Внутренний набор корпуса: второе дно + поперечные водонепроницаемые переборки.

    Это «начинка», которой на фото не видно, но без неё корпус - пустая скорлупа.
    Всё ставится с отступом от обшивки (0,25-0,35 м), чтобы не было совпадающих граней.
    """
    verts, uvs, faces = [], [], []

    def add(rings, v_band):
        global_v = len(verts)
        tv, tf = tube_from_rings(rings)
        for k, p in enumerate(tv):
            verts.append(np.array(p))
            uvs.append(((global_v + k) % 97 / 97.0, v_band))
        for f in tf:
            faces.append(tuple(i + global_v for i in f))

    # --- двойное дно (настил второго дна) ---------------------------------
    rings = []
    for x in np.linspace(0.11 * LOA, 0.79 * LOA, 30):
        w = max(float(half_breadth(np.array([x]), DB_HEIGHT)[0]) - 0.35, 1.0)
        rings.append([(x, w, DB_HEIGHT - DB_THICK / 2.0), (x, w, DB_HEIGHT + DB_THICK / 2.0),
                      (x, -w, DB_HEIGHT + DB_THICK / 2.0), (x, -w, DB_HEIGHT - DB_THICK / 2.0)])
    add(rings, 0.292)

    # --- поперечные переборки ---------------------------------------------
    zs = np.linspace(DB_HEIGHT + DB_THICK, D - 0.2, 16)
    for i in range(BH_COUNT):
        x = HATCH_X0_FRAC * LOA + i * (HATCH_LEN + HATCH_GAP) - HATCH_GAP / 2.0
        if x < 0.12 * LOA or x > 0.79 * LOA:
            continue
        rings = []
        for z in zs:
            w = max(float(half_breadth(np.array([x]), z)[0]) - 0.25, 0.5)
            rings.append([(x - BH_THICK / 2.0, w, z), (x - BH_THICK / 2.0, -w, z),
                          (x + BH_THICK / 2.0, -w, z), (x + BH_THICK / 2.0, w, z)])
        add(rings, 0.296)
    return verts, faces, uvs


def mesh_from_grid(name, grid):
    """grid: (rows, cols, 3) -> объект с квадовой сеткой."""
    rows, cols, _ = grid.shape
    verts = grid.reshape(-1, 3).tolist()
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a = i * cols + j
            faces.append((a, a + 1, a + cols + 1, a + cols))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate(verbose=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def make_material(name, rgb, rough=0.5, metallic=0.0, coat=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    mat.diffuse_color = (*rgb, 1.0)
    return mat


# ----------------------------------------------------------------------------
# 6. ПОСТРОЕНИЕ КОРПУСА
# ----------------------------------------------------------------------------
def build_hull():
    z_rows = level_rows()
    nz = len(z_rows)
    s_lat = np.linspace(0.0, 1.0, NL)

    col_x, col_y = [], []
    for z in z_rows:
        xs, ys = level_columns(z)
        col_x.append(xs)
        col_y.append(ys)

    NU = len(col_x[0])
    NB = 2 * NU - 1

    # --- 1. обшивка: строка = уровень z, колонка = обход борт -> нос -> борт ---
    grid = np.zeros((nz, NB, 3))
    for i, z in enumerate(z_rows):
        row = np.zeros((NB, 3))
        row[:NU, 0] = col_x[i]
        row[:NU, 1] = col_y[i]
        row[NU:, 0] = col_x[i][-2::-1]
        row[NU:, 1] = -col_y[i][-2::-1]
        row[:, 2] = z
        grid[i] = row
    # седловатость палубы (общая для обшивки, палубы и транца)
    grid[:, :, 2] = warp_z(grid[:, :, 0], grid[:, :, 2])
    shell = mesh_from_grid("Hull_Shell", grid)
    # обшивка: u - вдоль обхода (корма -> нос -> борт), v - от киля к палубе
    add_uv_grid(shell, nz, NB, lambda ri, cj: (cj / (NB - 1), 0.36 + 0.64 * ri / (nz - 1)))

    bottom = grid[0]

    # --- 2. днищевая плита (z = 0) -----------------------------------------
    g0 = np.zeros((NB, NL, 3))
    g0[:, :, 0] = bottom[:, 0][:, None]
    g0[:, :, 1] = bottom[:, 1][:, None] * s_lat[None, :]
    g0[:, :, 2] = 0.0
    bottom_ob = mesh_from_grid("Hull_Bottom", g0)
    add_uv_grid(bottom_ob, NB, NL, lambda ri, cj: (0.52 + 0.48 * ri / (NB - 1), 0.145 * cj / (NL - 1)))

    # --- 3. палуба (по седловатой линии верхней кромки борта) ---------------
    top = grid[-1]
    gd = np.zeros((NB, NL, 3))
    gd[:, :, 0] = top[:, 0][:, None]
    gd[:, :, 1] = top[:, 1][:, None] * s_lat[None, :]
    gd[:, :, 2] = top[:, 2][:, None]
    deck_ob = mesh_from_grid("Hull_Deck", gd)
    add_uv_grid(deck_ob, NB, NL, lambda ri, cj: (0.48 * ri / (NB - 1), 0.145 * cj / (NL - 1)))

    # --- 4. транцевая плита (только выше кромки транца Z_TR) ----------------
    k_tr = int(np.searchsorted(z_rows, Z_TR - 1e-6))
    z_tr_rows = z_rows[k_tr:]
    s_tr = np.concatenate([-s_lat[::-1][:-1], s_lat])
    gt = np.zeros((len(z_tr_rows), len(s_tr), 3))
    for i, z in enumerate(z_tr_rows):
        gt[i, :, 0] = float(F["xa"](z))
        gt[i, :, 1] = float(F["r"](z)) * float(F["b"](z)) * s_tr
        gt[i, :, 2] = warp_z(float(F["xa"](z)), z)
    transom = mesh_from_grid("Hull_Transom", gt)
    add_uv_grid(transom, len(z_tr_rows), len(s_tr),
                lambda ri, cj: (cj / (len(s_tr) - 1), 0.19 + 0.10 * ri / (len(z_tr_rows) - 1)))

    # --- 5. фальшборт по периметру палубы -----------------------------------
    bv, bf, buv = build_bulwark(top, gt[-1])
    bulwark = mesh_from_arrays("Hull_Bulwark", bv, bf)
    set_uv_from_verts(bulwark, buv)

    # --- 6. комингсы грузовых люков (палубная «начинка») --------------------
    cv, cf, cuv = build_coamings(top)
    coamings = mesh_from_arrays("Deck_Hatches", cv, cf)
    set_uv_from_verts(coamings, cuv)

    # --- 7. внутренний набор (второе дно + переборки) -----------------------
    iv, if_, iuv = build_internals()
    internals = mesh_from_arrays("Hull_Internals", iv, if_)
    set_uv_from_verts(internals, iuv)

    # --- материалы ---------------------------------------------------------
    m_aft = make_material("Hull_Antifouling_Below_WL", COL_AFT, rough=0.62)
    m_boot = make_material("Hull_Boot_Top_Band", COL_BOOT, rough=0.35)
    m_side = make_material("Hull_Topsides", COL_SIDE, rough=0.30, coat=0.25)
    m_deck = make_material("Deck_Main", COL_DECK, rough=0.70)

    for mat in (m_aft, m_boot, m_side):
        shell.data.materials.append(mat)
    deck_ob.data.materials.append(m_deck)
    bottom_ob.data.materials.append(m_aft)
    for mat in (m_side, m_aft, m_boot):        # 0 - борт, 1 - подводная, 2 - КВЛ
        transom.data.materials.append(mat)
    m_internal = make_material("Internal_Steel", COL_INTERNAL, rough=0.55)
    bulwark.data.materials.append(m_side)
    internals.data.materials.append(m_internal)
    coamings.data.materials.append(m_deck)

    for poly in shell.data.polygons:
        zc = sum(shell.data.vertices[v].co.z for v in poly.vertices) / len(poly.vertices)
        poly.material_index = 0 if zc < BOOT_LO else (1 if zc <= BOOT_HI else 2)
    for poly in transom.data.polygons:
        zc = sum(transom.data.vertices[v].co.z for v in poly.vertices) / len(poly.vertices)
        poly.material_index = 1 if zc < BOOT_LO else (2 if zc <= BOOT_HI else 0)

    # --- объединение -------------------------------------------------------
    bpy.ops.object.select_all(action="DESELECT")
    for ob in (shell, bottom_ob, deck_ob, transom, bulwark, coamings, internals):
        ob.select_set(True)
    bpy.context.view_layer.objects.active = shell
    bpy.ops.object.join()
    hull = bpy.context.view_layer.objects.active
    hull.name = "EmmaMaersk_Hull"
    hull.data.name = "EmmaMaersk_Hull_Mesh"

    bm = bmesh.new()
    bm.from_mesh(hull.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-5, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(hull.data)
    bm.free()
    hull.data.update()

    # сглаживание с сохранением рёбер (палуба, транец, форштевень)
    me = hull.data
    bm = bmesh.new()
    bm.from_mesh(me)
    for e in bm.edges:
        if len(e.link_faces) == 2:
            e.smooth = e.calc_face_angle(0.0) < math.radians(35.0)
        else:
            e.smooth = False
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(me)
    bm.free()
    me.update()

    return hull


# ----------------------------------------------------------------------------
# 7. КОНТРОЛЬНЫЕ РАСЧЁТЫ (по аналитической форме, независимо от сетки)
# ----------------------------------------------------------------------------
def numerical_report(verbose=True):
    zs = np.linspace(0.0, D, 2401)
    A, Mx = [], []
    for z in zs:
        xs, ys = level_columns(z)
        A.append(2.0 * np.trapezoid(ys, xs))
        Mx.append(2.0 * np.trapezoid(ys * xs, xs))
    A = np.asarray(A)
    Mx = np.asarray(Mx)

    def cut(arr, zt):
        sel = zs <= zt + 1e-9
        return float(np.trapezoid(arr[sel], zs[sel]))

    V = cut(A, T)
    xfw = float(F["xf"](T))
    xaw = float(F["xa"](T))
    lwl = xfw - xaw
    lcb = cut(Mx, T) / V
    zz = np.linspace(0.0, T, 600)
    bb_zz = np.array([float(F["b"](z)) for z in zz])
    am = float(np.trapezoid(2.0 * bb_zz, zz))
    wpa = float(A[np.searchsorted(zs, T)])

    dy = dict(
        vol=V, disp=V * 1.025, lwl=lwl, wpa=wpa, am=am,
        lcb_from_ap=lcb, lcb_pct_from_mid=100.0 * (lcb - (xaw + xfw) / 2.0) / lwl,
        cb_lwl=V / (lwl * B * T), cb_lbp=V / (LBP * B * T), cb_loa=V / (LOA * B * T),
        cm=am / (B * T), cwp=wpa / (lwl * B),
    )
    dy["cp"] = dy["cb_lwl"] / dy["cm"]
    if verbose:
        for k, v in dy.items():
            print("  %-20s %12.4f" % (k, v))
        for zv in (0.0, 4.0, 8.0, 12.0, T, 20.0, 26.0, D):
            print("  WPA@z=%-5.1f %12.0f" % (zv, float(A[np.searchsorted(zs, zv)])))
    return dy


# ----------------------------------------------------------------------------
# 8. СЦЕНА, ЭКСПОРТ, РЕНДЕР
# ----------------------------------------------------------------------------
def setup_scene(hull):
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"

    hull["vessel"] = "Emma Maersk (PS-class container carrier)"
    hull["scope"] = "Hull only: shell, bottom, deck, transom, bulbous bow. No superstructure."
    hull["scale"] = "1:100"
    hull["LOA_m_real"] = LOA
    hull["beam_m_real"] = B
    hull["depth_m_real"] = D
    hull["draught_m_real"] = T
    hull["LOA_m_model"] = round(LOA * SCALE, 4)
    hull["beam_m_model"] = round(B * SCALE, 4)
    hull["depth_m_model"] = round(D * SCALE, 4)
    hull["draught_m_model"] = round(T * SCALE, 4)

    hull.location = (-LOA / 2.0, 0.0, 0.0)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    hull.scale = (SCALE, SCALE, SCALE)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def build_stage(hull):
    """Студия: пол, свет, камера - .blend открывается готовым к рендеру."""
    scene = bpy.context.scene
    coll = bpy.data.collections.new("Presentation")
    scene.collection.children.link(coll)
    L = LOA * SCALE

    world = bpy.data.worlds.new("Studio")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.075, 0.082, 0.095, 1.0)
    bg.inputs[1].default_value = 1.6
    scene.world = world

    bpy.ops.mesh.primitive_plane_add(size=L * 14.0, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Studio_Floor"
    floor.data.materials.append(make_material("Studio_Floor_Mat", (0.045, 0.048, 0.055), rough=0.35))
    for c in floor.users_collection:
        c.objects.unlink(floor)
    coll.objects.link(floor)

    def add_area(name, loc, size, energy, rot):
        d = bpy.data.lights.new(name, "AREA")
        d.shape = "RECTANGLE"
        d.size, d.size_y = size
        d.energy = energy
        ob = bpy.data.objects.new(name, d)
        ob.location = loc
        ob.rotation_euler = rot
        coll.objects.link(ob)
        return ob

    add_area("Key_Light", (-L * 1.5, -L * 1.4, L * 1.7), (L * 2.2, L * 1.3),
             5200, (math.radians(50), 0, math.radians(-46)))
    add_area("Fill_Light", (L * 2.2, -L * 1.2, L * 0.9), (L * 2.4, L * 1.8),
             1800, (math.radians(76), 0, math.radians(62)))
    add_area("Rim_Light", (L * 0.2, L * 2.2, L * 1.2), (L * 2.6, L * 0.9),
             3200, (math.radians(-110), 0, math.radians(4)))

    cam_data = bpy.data.cameras.new("Cam_3-4")
    cam_data.lens = 50
    cam = bpy.data.objects.new("Cam_3-4", cam_data)
    cam.location = (-L * 1.45, -L * 1.02, L * 0.50)
    cam.rotation_euler = (math.radians(74), 0, math.radians(-55))
    coll.objects.link(cam)
    scene.camera = cam
    return coll, cam


def setup_render(engine_pref="EEVEE"):
    scene = bpy.context.scene
    engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    chosen = None
    if engine_pref == "EEVEE":
        for cand in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            if cand in engines:
                chosen = cand
                break
    if chosen is None:
        chosen = "CYCLES" if "CYCLES" in engines else engines[0]
    scene.render.engine = chosen
    if chosen.startswith("CYCLES"):
        scene.cycles.samples = 96
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 6
        scene.cycles.device = "CPU"
    else:
        try:
            scene.eevee.taa_render_samples = 64
            scene.eevee.use_raytracing = True
        except Exception:
            pass
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -1.2
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass


def place_cam(cam, kind, L, target_z):
    """Ракурсы превью. L - длина модели (1:100)."""
    import mathutils

    cam.constraints.clear()
    cam.data.type = "PERSP"

    if kind == "quarter_fore":
        cam.location = (-L * 1.45, -L * 1.02, L * 0.50)
        cam.data.lens = 50
    elif kind == "quarter_aft":
        cam.location = (L * 1.40, L * 0.92, L * 0.46)
        cam.data.lens = 50
    elif kind == "bow":
        cam.location = (-L * 1.58, -L * 0.66, L * 0.44)
        cam.data.lens = 70
    elif kind == "stern":
        cam.location = (L * 1.52, L * 0.64, L * 0.40)
        cam.data.lens = 70
    elif kind == "deck":
        cam.location = (-L * 0.50, -L * 0.62, L * 0.92)
        cam.data.lens = 55
    elif kind == "side":
        cam.location = (0.0, -L * 3.0, target_z)
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = L * 1.06
        cam.rotation_euler = (math.radians(90), 0.0, 0.0)
        return
    elif kind == "top":
        cam.location = (0.0, 0.0, L * 2.0)
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = L * 1.06
        cam.rotation_euler = (0.0, 0.0, 0.0)
        return
    elif kind == "bow_side":
        cam.location = (L * 0.40, -L * 1.5, target_z)
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = L * 0.34
        cam.rotation_euler = (math.radians(90), 0.0, 0.0)
        return
    elif kind == "bow_top":
        cam.location = (L * 0.40, 0.0, L * 2.0)
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = L * 0.34
        cam.rotation_euler = (0.0, 0.0, 0.0)
        return
    elif kind == "stern_side":
        cam.location = (-L * 0.44, -L * 1.5, target_z)
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = L * 0.30
        cam.rotation_euler = (math.radians(90), 0.0, 0.0)
        return
    else:
        raise ValueError("unknown view: %s" % kind)

    look = mathutils.Vector((0.0, 0.0, target_z)) - cam.location
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()


VIEW_NAMES = {"quarter_fore": "01_quarter_fore", "quarter_aft": "02_quarter_aft", "side": "03_side_ortho", "top": "04_top_ortho", "bow": "05_bow", "stern": "06_stern", "bow_side": "07_bow_side_ortho", "bow_top": "08_bow_top_ortho", "stern_side": "09_stern_side_ortho", "deck": "13_deck_view"}

VIEW_RES = {
    "quarter_fore": (1600, 1000),
    "quarter_aft": (1600, 1000),
    "bow": (1600, 1000),
    "stern": (1600, 1000),
    "side": (2000, 620),
    "top": (2000, 820),
    "bow_side": (1400, 900),
    "bow_top": (1400, 900),
    "stern_side": (1400, 900),
    "deck": (1800, 1000),
}


def export_all(hull, out_dir):
    """Выгрузка ТОЛЬКО в OBJ - под импорт в 3ds Max (и любой другой пакет).

    Оба варианта: Z вверх (как в 3ds Max), X - в нос, Y - поперек,
    один объект, квады, раздельные нормали на острых рёбрах, группы сглаживания,
    материалы в .mtl (4 шт., назначены по граням).
    """
    obj_dir = out_dir
    os.makedirs(obj_dir, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    hull.select_set(True)
    bpy.context.view_layer.objects.active = hull

    common = dict(
        export_selected_objects=True,
        apply_modifiers=True,
        export_uv=True,
        export_normals=True,
        export_materials=True,
        export_triangulated_mesh=False,
        export_smooth_groups=True,
        export_object_groups=False,
        export_material_groups=False,
        export_vertex_groups=False,
        forward_axis="Y",
        up_axis="Z",
    )

    # 1) основной файл для 3ds Max: единицы - МИЛЛИМЕТРЫ, LOA = 3977,1
    bpy.ops.wm.obj_export(
        filepath=os.path.join(obj_dir, "Emma_Maersk_hull_1-100_mm.obj"),
        global_scale=1000.0, **common)

    # 2) вариант в метрах (сцена, где 1 unit = 1 м), LOA = 3,9771
    bpy.ops.wm.obj_export(
        filepath=os.path.join(obj_dir, "Emma_Maersk_hull_1-100_m.obj"),
        global_scale=1.0, **common)


CUTAWAYS = [
    # вырез четверти: срезаем носовую и левую части - видно трюм, переборки, второе дно
    ("14_section_quarter", dict(planes=[((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                                       ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))],
                                cam=(0.58, 0.52, 0.44), aim=(0.0, 0.0, 0.08), lens=45, world=3.4)),
    # продольный разрез по ДП: ортогональная «элевация» - видно второе дно и переборки
    ("15_section_longitudinal", dict(planes=[((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))],
                                     cam=(0.0, 1.5, 0.040), aim=None, lens=50, world=3.4,
                                     ortho_scale=1.06,
                                     rot=(math.pi / 2.0, 0.0, math.pi),
                                     res=(2000, 700))),
    # поперечный разрез: ортогонально «в торец» - классический разрез по миделю
    ("16_section_inside", dict(planes=[((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                                      ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))],
                               cam=(0.100, 0.055, 0.033), aim=(-0.60, -0.05, 0.05),
                               lens=22, world=3.4, res=(1800, 1000))),
]


def render_cutaways(hull, cam, out_dir):
    """Разрезы: бисекция плоскостями + толщина листа (Solidify) + подсветка нутра."""
    import mathutils

    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    L = LOA * SCALE
    base = hull.data.copy()
    # лампа внутрь выреза: без неё трюм чёрный
    lamp_data = bpy.data.lights.new("SectionLamp", "POINT")
    lamp_data.energy = 26.0
    lamp_data.shadow_soft_size = L * 0.02
    lamp = bpy.data.objects.new("SectionLamp", lamp_data)
    bpy.context.collection.objects.link(lamp)
    lamp.location = (L * 0.12, L * 0.02, L * 0.10)
    wnode = scene.world.node_tree.nodes.get("Background") if scene.world else None
    w_strength = wnode.inputs[1].default_value if wnode else None
    try:
        for name, cfd in CUTAWAYS:
            hull.data = base.copy()
            bm = bmesh.new()
            bm.from_mesh(hull.data)
            for co, no in cfd["planes"]:
                bmesh.ops.bisect_plane(
                    bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
                    plane_co=co, plane_no=no, clear_outer=True, clear_inner=False, dist=1e-6)
            bm.to_mesh(hull.data)
            bm.free()
            hull.data.update()
            sol = hull.modifiers.new("SectionPlate", "SOLIDIFY")
            sol.thickness = 0.0022          # ~220 мм в натуре: толщина листа видна на срезе
            sol.offset = 0.0
            if wnode is not None:
                wnode.inputs[1].default_value = cfd.get("world", w_strength or 1.6)
            cam.constraints.clear()
            cam.location = (L * cfd["cam"][0], L * cfd["cam"][1], L * cfd["cam"][2])
            if cfd.get("ortho_scale"):
                cam.data.type = "ORTHO"
                cam.data.ortho_scale = L * cfd["ortho_scale"]
                cam.rotation_euler = cfd["rot"]
            else:
                cam.data.type = "PERSP"
                cam.data.lens = cfd["lens"]
                cam.data.clip_start = 0.005
                look = mathutils.Vector(cfd["aim"]) - cam.location
                cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
            rx, ry = cfd.get("res", (1800, 1100))
            scene.render.resolution_x = rx
            scene.render.resolution_y = ry
            scene.render.filepath = os.path.join(out_dir, name + ".png")
            bpy.ops.render.render(write_still=True)
            print("RENDERED %s" % name, flush=True)
            hull.modifiers.clear()
    except Exception as exc:                              # noqa: BLE001
        print("RENDER_FAILED cutaway: %r" % (exc,), flush=True)
    finally:
        hull.modifiers.clear()
        hull.data = base
        bpy.data.objects.remove(lamp, do_unlink=True)
        if wnode is not None and w_strength is not None:
            wnode.inputs[1].default_value = w_strength


CLAY_VIEWS = {"10_clay_fore": "quarter_fore", "11_clay_bow": "bow"}


UV_MAP = [
    (0.00, 0.36, 1.00, 1.00, "Обшивка: борт + подводная часть",
     "u — вдоль обхода корпуса (корма → нос → борт), v — от киля к палубе"),
    (0.00, 0.19, 1.00, 0.29, "Транец", "u — по ширине транца, v — по высоте"),
    (0.00, 0.00, 0.48, 0.145, "Палуба", "план: u — по длине, v — по полушироте"),
    (0.52, 0.00, 1.00, 0.145, "Днище", "план: u — по длине, v — по полушироте"),
    (0.00, 0.155, 1.00, 0.185, "Фальшборт", "u — по периметру палубы, v — по высоте"),
    (0.00, 0.300, 1.00, 0.345, "Комингсы люков", "18 рамок, у каждой своя полоса по u"),
]


def export_uv_layout(hull, out_dir):
    """Компактная размеченная карта UV-развёртки (SVG, несколько КБ).

    Blender умеет выгружать «раскладку» сам, но сыпет контуром всех 68 тыс. граней
    (~11 МБ). Здесь острова и так прямоугольные, поэтому рисуем схему с подписями
    и линией КВЛ - так модельеру в 3ds Max сразу видно, где красить борт и днище.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "12_uv_layout.svg")
    W = 1000

    def X(u):
        return u * W

    def Y(v):
        return (1.0 - v) * W

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (W, W, W, W),
           '<rect width="100%" height="100%" fill="#12161c"/>',
           '<text x="16" y="30" fill="#8ab4f8" font-family="Segoe UI, Arial" font-size="24">'
           'Emma Maersk hull 1:100 — карта UV-развёртки (u / v в 0…1)</text>']
    for u0, v0, u1, v1, title, note in UV_MAP:
        x0, y0 = X(u0), Y(v1)
        w, h = X(u1) - X(u0), Y(v0) - Y(v1)
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#1b232d" '
                   'stroke="#6fa8dc" stroke-width="2"/>' % (x0, y0, w, h))
        out.append('<text x="%.1f" y="%.1f" fill="#d7e6f5" font-family="Segoe UI, Arial" '
                   'font-size="22" text-anchor="middle">%s</text>'
                   % (x0 + w / 2.0, y0 + (34 if note else h / 2.0), title))
        if note:
            out.append('<text x="%.1f" y="%.1f" fill="#93a4b8" font-family="Segoe UI, Arial" '
                       'font-size="18" text-anchor="middle">%s</text>'
                       % (x0 + w / 2.0, y0 + 62, note))
    v_wl = 0.36 + 0.64 * T / D
    out.append('<line x1="0" y1="%.1f" x2="%d" y2="%.1f" stroke="#e06666" stroke-width="2" '
               'stroke-dasharray="9 7"/>' % (Y(v_wl), W, Y(v_wl)))
    out.append('<text x="16" y="%.1f" fill="#e06666" font-family="Segoe UI, Arial" font-size="20">'
               'КВЛ: осадка %0.0f м → v = %.3f (ниже — противообрастающая окраска)</text>'
               % (Y(v_wl) - 12, T, v_wl))
    out.append('</svg>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("UV_LAYOUT %s (%d байт)" % (path, os.path.getsize(path)), flush=True)


def render_clay(hull, cam, out_dir):
    """Нейтральный «клэй»-рендер: оценивается только форма, без окраски."""
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    clay = make_material("Clay_Preview", (0.34, 0.345, 0.36), rough=0.62)
    saved = list(hull.data.materials)
    for i in range(len(saved)):
        hull.data.materials[i] = clay
    scene.view_settings.exposure = -0.4
    for name, view in CLAY_VIEWS.items():
        try:
            place_cam(cam, view, LOA * SCALE, D * SCALE * 0.42)
            scene.render.resolution_x = 1600
            scene.render.resolution_y = 1000
            scene.render.filepath = os.path.join(out_dir, f"{name}.png")
            bpy.ops.render.render(write_still=True)
            print("RENDERED %s" % name, flush=True)
        except Exception as exc:                      # noqa: BLE001
            print("RENDER_FAILED %s: %r" % (name, exc), flush=True)
    scene.view_settings.exposure = -1.2
    for i, m in enumerate(saved):
        hull.data.materials[i] = m


def render_previews(cam, out_dir, keys):
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    L = LOA * SCALE
    target_z = D * SCALE * 0.42
    for k in keys:
        try:
            place_cam(cam, k, L, target_z)
            rx, ry = VIEW_RES.get(k, (1600, 1000))
            scene.render.resolution_x = rx
            scene.render.resolution_y = ry
            scene.render.filepath = os.path.join(out_dir, VIEW_NAMES.get(k, k) + ".png")
            bpy.ops.render.render(write_still=True)
            print("RENDERED %s" % k, flush=True)
        except Exception as exc:                      # noqa: BLE001
            print("RENDER_FAILED %s: %r" % (k, exc), flush=True)


# ----------------------------------------------------------------------------
# 9. MAIN
# ----------------------------------------------------------------------------
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_dir = "."
    do_render = False
    do_report_only = False
    for i, a in enumerate(argv):
        if a == "--out":
            out_dir = argv[i + 1]
        elif a == "--render":
            do_render = True
        elif a == "--report":
            do_report_only = True

    bpy.ops.wm.read_factory_settings(use_empty=True)

    if "--dump" in argv:
        print("HALF-BREADTH TABLE  rows=z  cols=x")
        zt = [0.0, 4.0, 8.0, 12.0, T, 20.0, 24.0, D]
        xh = [10, 40, 80, 120, 160, 200, 240, 280, 300, 320, 340, 360, 375, 385, 392]
        print("   z\\x " + " ".join("%7d" % x for x in xh))
        for z in zt:
            row = [float(half_breadth(np.array([x]), z)[0]) for x in xh]
            print("%7.1f " % z + " ".join("%7.2f" % v for v in row))
        print()
        print("SECTION HALF-BREADTH  rows=x  cols=z")
        print("   x\\z " + " ".join("%7.1f" % z for z in zt))
        for x in (40, 80, 150, 220, 300, 340, 370, 385):
            row = [float(half_breadth(np.array([x]), z)[0]) for z in zt]
            print("%7d " % x + " ".join("%7.2f" % v for v in row))
        print()
        for z in (0.0, 8.0, 16.0, 24.0, D):
            print("LEVEL z=%.1f  xa=%.1f xa2=%.1f xf2=%.1f xf=%.1f wa=%.1f b=%.1f"
                  % (z, float(F["xa"](z)), level_geom(z)["xa2"], level_geom(z)["xf2"],
                     float(F["xf"](z)), level_geom(z)["wa"], float(F["b"](z))))
        print("DUMP_ONLY")
        return

    if do_report_only:
        numerical_report(verbose=True)
        print("REPORT_ONLY")
        return

    hull = build_hull()
    me = hull.data
    nq = sum(1 for p in me.polygons if len(p.vertices) == 4)
    nt = sum(1 for p in me.polygons if len(p.vertices) == 3)
    nn = sum(1 for p in me.polygons if len(p.vertices) > 4)
    print("MESH verts=%d faces=%d edges=%d" % (len(me.vertices), len(me.polygons), len(me.edges)))
    print("FACE_TYPES quads=%d tris=%d ngons=%d" % (nq, nt, nn))

    bm = bmesh.new()
    bm.from_mesh(me)
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    vol_full = bm.calc_volume(signed=False)
    bm.free()

    setup_scene(hull)
    rep = numerical_report(verbose=True)

    bm = bmesh.new()
    bm.from_mesh(hull.data)
    vol_model = bm.calc_volume(signed=False)
    bm.free()

    xs = [v.co.x for v in hull.data.vertices]
    ys = [v.co.y for v in hull.data.vertices]
    zs = [v.co.z for v in hull.data.vertices]
    print("BBOX_X %.4f..%.4f  (%.4f)" % (min(xs), max(xs), max(xs) - min(xs)))
    print("BBOX_Y %.4f..%.4f  (%.4f)" % (min(ys), max(ys), max(ys) - min(ys)))
    print("BBOX_Z %.4f..%.4f  (%.4f)" % (min(zs), max(zs), max(zs) - min(zs)))
    print("NON_MANIFOLD_EDGES %d" % non_manifold)
    print("VOLUME_WHOLE_HULL_full_scale_m3 %.0f" % vol_full)
    print("VOLUME_WHOLE_HULL_model_m3 %.6f" % vol_model)
    print("DISPLACEMENT_t %.0f" % rep["disp"])

    os.makedirs(out_dir, exist_ok=True)
    setup_render("EEVEE")
    coll, cam = build_stage(hull)
    if "--blend" in argv:      # рабочая сцена Blender (по умолчанию не выгружается)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "Emma_Maersk_Hull.blend"))
    export_all(hull, out_dir)
    export_uv_layout(hull, os.path.join(out_dir, "renders"))

    if do_render:
        render_previews(cam, os.path.join(out_dir, "renders"),
                        ["quarter_fore", "quarter_aft", "side", "top", "bow", "stern",
                         "bow_side", "bow_top", "stern_side", "deck"])
        render_clay(hull, cam, os.path.join(out_dir, "renders"))
        render_cutaways(hull, cam, os.path.join(out_dir, "renders"))

    print("DONE")


main()
