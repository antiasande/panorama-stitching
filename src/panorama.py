"""
Panorama Image Stitching con OpenCV (sin usar cv2.Stitcher).

Módulo con todo el pipeline de la práctica:

1. Detección y descripción de características (esquinas / blobs, binarios / coma flotante).
2. Emparejamiento con fuerza bruta o FLANN + test de razón de Lowe.
3. Estimación robusta de homografías / traslaciones con RANSAC.
4. Inserción de una imagen sobre un plano detectado (Parte 1).
5. Stitching planar de N imágenes ordenadas (Parte 2).
6. Stitching wide-view sobre cilindro o esfera, ordenación automática de
   imágenes desordenadas y varias estrategias de blending (Parte 3).

Compatible con OpenCV 4.x y 5.x (en la 5, BRISK, AKAZE y KAZE están en xfeatures2d).
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

# =============================================================================
# 1. Detectores y descriptores
# =============================================================================


def _crear(nombre, *args, **kwargs):
    """Crea un detector buscando primero en cv2 y luego en cv2.xfeatures2d (OpenCV 5)."""
    fabrica = getattr(cv2, f"{nombre}_create", None)
    if fabrica is None and hasattr(cv2, "xfeatures2d"):
        fabrica = getattr(cv2.xfeatures2d, f"{nombre}_create", None)
    if fabrica is None:
        raise RuntimeError(f"{nombre} no está disponible en esta versión de OpenCV")
    return fabrica(*args, **kwargs)


@dataclass
class Metodo:
    """Combinación detector + descriptor."""

    nombre: str
    tipo_detector: str      # "esquinas" o "blobs"
    tipo_descriptor: str    # "binario" o "float"
    detector: object
    descriptor: object
    tamano_kp: float = None  # tamaño fijo para keypoints que no lo traen (GFTT)

    def detectar(self, gris, max_kp):
        kps = self.detector.detect(gris, None)
        if max_kp and len(kps) > max_kp:
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[:max_kp]  # los más fuertes
        if self.tamano_kp:
            for k in kps:
                k.size = self.tamano_kp
        return kps

    def describir(self, gris, kps):
        return self.descriptor.compute(gris, kps)


def crear_metodo(nombre, nfeatures=4000):
    """
    Devuelve un :class:`Metodo`. Opciones:

    =========  ==========  ===========  ==========================================
    nombre     detector    descriptor   detalle
    =========  ==========  ===========  ==========================================
    orb        esquinas    binario      FAST + rBRIEF
    brisk      esquinas    binario      AGAST multiescala + BRISK
    gftt_sift  esquinas    float        Shi-Tomasi + descriptor SIFT
    sift       blobs       float        Diferencia de Gaussianas + SIFT
    kaze       blobs       float        Hessiano en espacio no lineal + KAZE
    akaze      blobs       binario      Hessiano en espacio no lineal + M-LDB
    =========  ==========  ===========  ==========================================
    """
    n = nombre.lower()
    if n == "orb":
        orb = cv2.ORB_create(nfeatures=nfeatures)
        return Metodo("ORB", "esquinas", "binario", orb, orb)
    if n == "brisk":
        b = _crear("BRISK")
        return Metodo("BRISK", "esquinas", "binario", b, b)
    if n == "gftt_sift":
        g = cv2.GFTTDetector_create(nfeatures, 0.01, 8, 3, True)  # Harris
        return Metodo("GFTT+SIFT", "esquinas", "float", g, cv2.SIFT_create(), tamano_kp=31)
    if n == "sift":
        s = cv2.SIFT_create(nfeatures=nfeatures)
        return Metodo("SIFT", "blobs", "float", s, s)
    if n == "kaze":
        k = _crear("KAZE")
        return Metodo("KAZE", "blobs", "float", k, k)
    if n == "akaze":
        a = _crear("AKAZE")
        return Metodo("AKAZE", "blobs", "binario", a, a)
    raise ValueError(f"Método desconocido: {nombre}")


METODOS = ["orb", "brisk", "gftt_sift", "sift", "kaze", "akaze"]


@dataclass
class Caracteristicas:
    kps: list
    desc: np.ndarray
    t_deteccion: float
    t_descripcion: float


def extraer(img_rgb, metodo, max_kp=4000):
    """Detecta y describe características midiendo por separado cada fase."""
    gris = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    t0 = time.perf_counter()
    kps = metodo.detectar(gris, max_kp)
    t1 = time.perf_counter()
    kps, desc = metodo.describir(gris, kps)
    t2 = time.perf_counter()
    return Caracteristicas(kps, desc, t1 - t0, t2 - t1)


# =============================================================================
# 2. Emparejamiento
# =============================================================================


def crear_matcher(tipo_descriptor, estrategia="bf"):
    """Fuerza bruta (BF) o FLANN; FLANN usa KD-tree para float y LSH para binarios."""
    if estrategia == "bf":
        norma = cv2.NORM_HAMMING if tipo_descriptor == "binario" else cv2.NORM_L2
        return cv2.BFMatcher(norma, crossCheck=False)
    if estrategia == "flann":
        if tipo_descriptor == "float":
            indice = dict(algorithm=1, trees=5)  # KD-tree
        else:
            indice = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)  # LSH
        return cv2.FlannBasedMatcher(indice, dict(checks=64))
    raise ValueError("estrategia debe ser 'bf' o 'flann'")


def emparejar(desc_a, desc_b, tipo_descriptor, estrategia="bf", ratio=0.75):
    """
    knnMatch con k=2 y test de razón de Lowe.

    Devuelve siempre (matches, tiempo), también cuando no hay descriptores.
    """
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return [], 0.0
    dtype = np.uint8 if tipo_descriptor == "binario" else np.float32
    desc_a, desc_b = desc_a.astype(dtype, copy=False), desc_b.astype(dtype, copy=False)

    matcher = crear_matcher(tipo_descriptor, estrategia)
    t0 = time.perf_counter()
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    t = time.perf_counter() - t0

    buenos = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    return buenos, t


def puntos(kps_a, kps_b, matches):
    pa = np.float32([kps_a[m.queryIdx].pt for m in matches])
    pb = np.float32([kps_b[m.trainIdx].pt for m in matches])
    return pa, pb


# =============================================================================
# 3. Estimación robusta
# =============================================================================


def homografia_ransac(kps_a, kps_b, matches, umbral=4.0):
    """Homografía que lleva puntos de A a B. Devuelve (H, mascara_inliers, n_inliers, tiempo)."""
    if len(matches) < 4:
        return None, None, 0, 0.0
    pa, pb = puntos(kps_a, kps_b, matches)
    t0 = time.perf_counter()
    H, mascara = cv2.findHomography(pa, pb, cv2.RANSAC, umbral)
    t = time.perf_counter() - t0
    n_in = int(mascara.sum()) if mascara is not None else 0
    return H, mascara, n_in, t


def error_reproyeccion(H, kps_a, kps_b, matches, mascara):
    """Error medio (px) al proyectar con H los inliers de A sobre B."""
    if H is None or mascara is None or mascara.sum() == 0:
        return np.nan
    pa, pb = puntos(kps_a, kps_b, matches)
    sel = mascara.ravel().astype(bool)
    proy = cv2.perspectiveTransform(pa[sel].reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.linalg.norm(proy - pb[sel], axis=1).mean())


def traslacion_ransac(kps_a, kps_b, matches, umbral=3.0, iteraciones=1000, semilla=0):
    """
    Traslación (dx, dy) de A a B estimada con RANSAC (1 punto por hipótesis)
    y refinada con la mediana de los inliers. Devuelve ((dx, dy), n_inliers).
    """
    if len(matches) < 6:
        return None, 0
    pa, pb = puntos(kps_a, kps_b, matches)
    d = pb - pa
    rng = np.random.default_rng(semilla)
    mejor, mejor_n = None, 0
    for idx in rng.integers(0, len(d), size=iteraciones):
        n = int((np.linalg.norm(d - d[idx], axis=1) < umbral).sum())
        if n > mejor_n:
            mejor, mejor_n = d[idx], n
    inl = d[np.linalg.norm(d - mejor, axis=1) < umbral]
    dx, dy = np.median(inl, axis=0)
    return (float(dx), float(dy)), mejor_n


def matriz_traslacion(dx, dy):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


# =============================================================================
# 4. Parte 1: detección de un plano e inserción de una imagen
# =============================================================================


def insertar_en_plano(modelo, escena, insertar, metodo, estrategia="bf", ratio=0.75, umbral=5.0):
    """
    Localiza el objeto plano `modelo` en `escena` y pega `insertar` con la perspectiva correcta.

    Devuelve un dict con keypoints, matches, homografía, imagen compuesta y métricas.
    """
    cv2.setRNGSeed(0)
    fm = extraer(modelo, metodo)
    fe = extraer(escena, metodo)
    matches, t_match = emparejar(fm.desc, fe.desc, metodo.tipo_descriptor, estrategia, ratio)
    H, mascara, n_in, t_ransac = homografia_ransac(fm.kps, fe.kps, matches, umbral)

    resultado = None
    if H is not None:
        h, w = modelo.shape[:2]
        ins = cv2.resize(insertar, (w, h), interpolation=cv2.INTER_AREA)
        H_, W_ = escena.shape[:2]
        warp = cv2.warpPerspective(ins, H, (W_, H_))
        # Se warpea también una máscara completa: así no se pierden los píxeles negros de la imagen insertada
        mascara_warp = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (W_, H_)) > 0
        resultado = escena.copy()
        resultado[mascara_warp] = warp[mascara_warp]

    return dict(
        modelo=fm, escena=fe, matches=matches, H=H, mascara=mascara, resultado=resultado,
        metricas=dict(
            Método=metodo.nombre, Detector=metodo.tipo_detector, Descriptor=metodo.tipo_descriptor,
            Matcher=estrategia.upper(), **{"KP modelo": len(fm.kps), "KP escena": len(fe.kps)},
            Matches=len(matches), Inliers=n_in,
            **{"% inliers": 100 * n_in / max(len(matches), 1),
               "t detección (ms)": 1000 * (fm.t_deteccion + fe.t_deteccion),
               "t descripción (ms)": 1000 * (fm.t_descripcion + fe.t_descripcion),
               "t matching (ms)": 1000 * t_match,
               "t RANSAC (ms)": 1000 * t_ransac},
        ),
    )


def contorno_proyectado(H, forma):
    """Esquinas de una imagen de tamaño `forma` proyectadas con H (para dibujar el plano detectado)."""
    h, w = forma[:2]
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(c, H).reshape(-1, 2)


# =============================================================================
# 5. Utilidades de lienzo y proyecciones
# =============================================================================


def lienzo(transformaciones, formas, max_pixeles=60e6):
    """Calcula el tamaño del lienzo y la traslación S que deja todas las coordenadas en positivo."""
    esquinas = []
    for H, (h, w) in zip(transformaciones, formas):
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        esquinas.append(cv2.perspectiveTransform(c, H).reshape(-1, 2))
    p = np.vstack(esquinas)
    (x0, y0), (x1, y1) = p.min(axis=0), p.max(axis=0)
    ancho, alto = int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0))
    if ancho * alto > max_pixeles:
        raise RuntimeError(f"Lienzo demasiado grande ({ancho}x{alto}): la proyección planar ha divergido")
    S = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    return ancho, alto, S


def proyeccion_cilindrica(img, f):
    """Proyecta la imagen sobre un cilindro de radio f (en píxeles) y la 'desenrolla'."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    theta = (xg - cx) / f
    altura = (yg - cy) / f
    map_x = f * np.tan(theta) + cx
    map_y = altura * f / np.cos(theta) + cy
    map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mascara = cv2.remap(np.full((h, w), 255, np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT)
    return out, mascara


def proyeccion_esferica(img, f):
    """Proyecta la imagen sobre una esfera de radio f (coordenadas de longitud / latitud)."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    theta = (xg - cx) / f
    phi = (yg - cy) / f
    map_x = f * np.tan(theta) + cx
    map_y = f * np.tan(phi) / np.cos(theta) + cy
    map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mascara = cv2.remap(np.full((h, w), 255, np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT)
    return out, mascara


def warpear(imgs, mascaras, transformaciones):
    """Warpea todas las imágenes (y sus máscaras) a un lienzo común."""
    ancho, alto, S = lienzo(transformaciones, [i.shape[:2] for i in imgs])
    wi, wm = [], []
    for img, m, T in zip(imgs, mascaras, transformaciones):
        wi.append(cv2.warpPerspective(img, S @ T, (ancho, alto)))
        wm.append(cv2.warpPerspective(m, S @ T, (ancho, alto), flags=cv2.INTER_NEAREST))
    return wi, wm


def recortar_contenido(img, mascaras):
    union = np.any(np.stack([m > 0 for m in mascaras]), axis=0)
    ys, xs = np.where(union)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def recortar_rectangulo_interior(img, mascaras, margen=10):
    """
    Recorta el mayor rectángulo horizontal sin bordes negros: se buscan las filas en las
    que la unión de máscaras cubre todo el ancho útil del panorama.
    """
    union = np.any(np.stack([m > 0 for m in mascaras]), axis=0)
    cols = np.where(union.any(axis=0))[0]
    x0, x1 = cols.min() + margen, cols.max() - margen
    filas_llenas = np.where(union[:, x0:x1 + 1].all(axis=1))[0]
    if len(filas_llenas) == 0:
        return recortar_contenido(img, mascaras)
    y0, y1 = filas_llenas.min(), filas_llenas.max()
    return img[y0:y1 + 1, x0:x1 + 1]


# =============================================================================
# 6. Blending
# =============================================================================


def _pesos_distancia(mascara):
    m = (mascara > 0).astype(np.uint8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)


def blend_superposicion(imgs, mascaras):
    """Sin blending: cada imagen pisa a las anteriores."""
    out = np.zeros_like(imgs[0])
    for img, m in zip(imgs, mascaras):
        out[m > 0] = img[m > 0]
    return out


def blend_media(imgs, mascaras):
    """Media aritmética de todas las imágenes que cubren cada píxel."""
    acc = np.zeros(imgs[0].shape, np.float32)
    n = np.zeros(imgs[0].shape[:2], np.float32)
    for img, m in zip(imgs, mascaras):
        w = (m > 0).astype(np.float32)
        acc += img * w[..., None]
        n += w
    return (acc / np.maximum(n, 1)[..., None]).astype(np.uint8)


def blend_feather(imgs, mascaras):
    """Media ponderada por la distancia al borde de cada imagen (feathering)."""
    acc = np.zeros(imgs[0].shape, np.float32)
    wsum = np.zeros(imgs[0].shape[:2], np.float32)
    for img, m in zip(imgs, mascaras):
        w = _pesos_distancia(m)
        acc += img * w[..., None]
        wsum += w
    return (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)


def blend_max_peso(imgs, mascaras):
    """Cada píxel se toma de la imagen en la que está más lejos del borde (sin mezcla)."""
    W = np.stack([_pesos_distancia(m) for m in mascaras])
    mejor = W.argmax(axis=0)
    out = np.zeros_like(imgs[0])
    for i, img in enumerate(imgs):
        sel = (mejor == i) & (W[i] > 0)
        out[sel] = img[sel]
    return out


def blend_multibanda(imgs, mascaras, niveles=5):
    """
    Multi-band blending (Burt & Adelson): se mezclan las pirámides laplacianas de las imágenes
    con pesos suavizados por una pirámide gaussiana. Las frecuencias bajas se mezclan en una
    zona ancha y las altas en una zona estrecha, evitando costuras y fantasmas.
    """
    W = np.stack([_pesos_distancia(m) for m in mascaras])
    mejor = W.argmax(axis=0)
    # pesos "duros" (como max_peso) que la pirámide gaussiana se encarga de suavizar
    pesos = [((mejor == i) & (W[i] > 0)).astype(np.float32) for i in range(len(imgs))]

    salida = None
    for img, w in zip(imgs, pesos):
        g = [img.astype(np.float32)]
        gw = [w]
        for _ in range(niveles):
            g.append(cv2.pyrDown(g[-1]))
            gw.append(cv2.pyrDown(gw[-1]))
        lap = [g[i] - cv2.pyrUp(g[i + 1], dstsize=g[i].shape[1::-1]) for i in range(niveles)] + [g[-1]]
        contrib = [l * gwi[..., None] for l, gwi in zip(lap, gw)]
        salida = contrib if salida is None else [s + c for s, c in zip(salida, contrib)]

    out = salida[-1]
    for i in range(niveles - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=salida[i].shape[1::-1]) + salida[i]
    union = np.any(np.stack([m > 0 for m in mascaras]), axis=0)
    out[~union] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_graphcut_multibanda(imgs, mascaras, bandas=8, escala=0.2, exposicion=False):
    """
    Compensación de exposición + costura óptima con GraphCut + multi-band.

    La ganancia de exposición de cada imagen y la costura se calculan a escala reducida
    (son operaciones costosas en memoria y tiempo) y después se aplican a resolución completa.

    `exposicion=False` por defecto: con estas fotos la ganancia global por imagen empeora el
    resultado, porque el viñeteado oscurece los bordes (justo donde están los solapes).
    """
    bgr = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in imgs]
    ms = [(m > 0).astype(np.uint8) * 255 for m in mascaras]
    esquinas = [(0, 0)] * len(bgr)
    peq = [cv2.resize(b, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) for b in bgr]
    ms_peq = [cv2.resize(m, None, fx=escala, fy=escala, interpolation=cv2.INTER_NEAREST) for m in ms]

    # 1) Compensación de exposición: una ganancia por imagen que iguala el brillo en los solapes
    ganancias = np.ones(len(bgr))
    if exposicion:
        comp = cv2.detail_GainCompensator()
        comp.feed(esquinas, peq, ms_peq)
        ganancias = np.array([float(np.asarray(g).ravel()[0]) for g in comp.getMatGains(None)])
    bgr = [np.clip(b.astype(np.float32) * g, 0, 255).astype(np.uint8) for b, g in zip(bgr, ganancias)]
    peq = [np.clip(p.astype(np.float32) * np.float32(g), 0, 255).astype(np.float32) for p, g in zip(peq, ganancias)]

    # 2) GraphCut: costura de mínimo coste (diferencia de color y gradiente) en los solapes
    ms_peq = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD").find(peq, esquinas, ms_peq)
    ms_costura = [cv2.resize(mp.get() if hasattr(mp, "get") else mp, ms[0].shape[1::-1],
                             interpolation=cv2.INTER_NEAREST) & m for mp, m in zip(ms_peq, ms)]

    # 3) Multi-band blending alrededor de la costura
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(bandas)
    blender.prepare((0, 0, bgr[0].shape[1], bgr[0].shape[0]))
    for b, m in zip(bgr, ms_costura):
        blender.feed(b.astype(np.int16), m, (0, 0))
    res, _ = blender.blend(None, None)
    return cv2.cvtColor(cv2.convertScaleAbs(res), cv2.COLOR_BGR2RGB)


BLENDINGS = {
    "superposición": blend_superposicion,
    "media": blend_media,
    "feather": blend_feather,
    "max_peso": blend_max_peso,
    "multibanda": blend_multibanda,
    "graphcut+multibanda": blend_graphcut_multibanda,
}


# =============================================================================
# 7. Parte 2: stitching planar de N imágenes ordenadas
# =============================================================================


def encadenar_a_referencia(H_adyacentes, ref):
    """
    Dadas las transformaciones H_i que llevan la imagen i a la i+1, devuelve T_i,
    que lleva la imagen i al sistema de coordenadas de la imagen de referencia.

    - Imágenes a la izquierda:  T_i = T_{i+1} · H_i
    - Imágenes a la derecha:    T_i = T_{i-1} · H_{i-1}^{-1}

    El orden del producto importa: primero se aplica la transformación de la derecha.
    """
    n = len(H_adyacentes) + 1
    T = [None] * n
    T[ref] = np.eye(3)
    for i in range(ref - 1, -1, -1):
        T[i] = T[i + 1] @ H_adyacentes[i]
    for i in range(ref + 1, n):
        T[i] = T[i - 1] @ np.linalg.inv(H_adyacentes[i - 1])
    return T


def escalar_transformacion(T, s):
    """Adapta una transformación calculada a resolución completa a imágenes reescaladas por s."""
    D = np.diag([s, s, 1.0])
    return D @ T @ np.linalg.inv(D)


def stitching_planar(imgs, metodo="sift", estrategia="flann", ratio=0.75, umbral=4.0,
                     ref=None, blending="feather", min_inliers=20, escala_salida=0.5,
                     componer=True, encadenar=None):
    """
    Stitching planar: cada imagen se proyecta con su homografía sobre el plano de la de referencia.

    Las características y homografías se calculan a resolución completa; el panorama se compone
    a `escala_salida` para limitar memoria (el lienzo planar crece mucho en los extremos).
    """
    cv2.setRNGSeed(0)
    encadenar = encadenar or encadenar_a_referencia
    m = crear_metodo(metodo)
    t0 = time.perf_counter()
    feats = [extraer(i, m) for i in imgs]
    filas, H_ady = [], []
    for i in range(len(imgs) - 1):
        a, b = feats[i], feats[i + 1]
        matches, t_m = emparejar(a.desc, b.desc, m.tipo_descriptor, estrategia, ratio)
        H, masc, n_in, t_r = homografia_ransac(a.kps, b.kps, matches, umbral)
        filas.append({"Par": f"{i}→{i + 1}", "Matches": len(matches), "Inliers": n_in,
                      "Error reproy. (px)": error_reproyeccion(H, a.kps, b.kps, matches, masc),
                      "t matching (ms)": 1000 * t_m, "t RANSAC (ms)": 1000 * t_r})
        if H is None or n_in < min_inliers:
            raise RuntimeError(f"No se pudo alinear {i}→{i + 1} ({n_in} inliers)")
        H_ady.append(H)
    ref = len(imgs) // 2 if ref is None else ref
    T = encadenar(H_ady, ref)
    pano = None
    if componer:
        s = escala_salida
        peq = [cv2.resize(i, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for i in imgs]
        mascaras = [np.full(i.shape[:2], 255, np.uint8) for i in peq]
        wi, wm = warpear(peq, mascaras, [escalar_transformacion(t, s) for t in T])
        pano = recortar_contenido(BLENDINGS[blending](wi, wm), wm)
    info = dict(
        T=T, H=H_ady, feats=feats, metodo=m,
        tiempo_total=time.perf_counter() - t0,
        kps=[len(f.kps) for f in feats],
        t_deteccion=sum(f.t_deteccion for f in feats),
        t_descripcion=sum(f.t_descripcion for f in feats),
        pares=pd.DataFrame(filas),
    )
    return pano, info


# =============================================================================
# 8. Parte 3: wide-view (cilindro / esfera), orden desconocido
# =============================================================================


def proyectar(imgs, f, superficie="cilindro"):
    fn = proyeccion_cilindrica if superficie == "cilindro" else proyeccion_esferica
    pares = [fn(i, f) for i in imgs]
    return [p[0] for p in pares], [p[1] for p in pares]


def estimar_focal(imgs, metodo="sift", estrategia="flann", factores=np.arange(0.8, 1.45, 0.1),
                  escala=0.5, superficie="cilindro"):
    """
    Estima la distancia focal (px) por búsqueda en rejilla: proyecta las imágenes con cada
    candidata f = factor · ancho y se queda con la que maximiza el total de inliers de
    traslación entre imágenes consecutivas. Con la focal correcta, tras la proyección el
    movimiento entre fotos es casi una traslación pura y hay más inliers.
    Se trabaja a resolución reducida para que sea rápido.
    """
    peq = [cv2.resize(i, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) for i in imgs]
    m = crear_metodo(metodo)
    w = peq[0].shape[1]
    filas = []
    for k in factores:
        cv2.setRNGSeed(0)
        proy, _ = proyectar(peq, k * w, superficie)
        feats = [extraer(p, m, 2000) for p in proy]
        tr = traslaciones_por_pares(feats, m.tipo_descriptor, [(i, i + 1) for i in range(len(imgs) - 1)],
                                    estrategia, umbral=1.5)
        inl = [r["inliers"] for r in tr.values()]
        filas.append({"factor": round(float(k), 2), "f (px)": k * imgs[0].shape[1],
                      "Inliers totales": sum(inl), "Inliers par más débil": min(inl)})
    tabla = pd.DataFrame(filas)
    mejor = tabla.loc[tabla["Inliers totales"].idxmax(), "f (px)"]
    return float(mejor), tabla


def traslaciones_por_pares(feats, tipo_descriptor, pares, estrategia="flann", ratio=0.75, umbral=3.0):
    """Para cada par (i, j) estima la traslación de i a j y su número de inliers."""
    res = {}
    for i, j in pares:
        matches, _ = emparejar(feats[i].desc, feats[j].desc, tipo_descriptor, estrategia, ratio)
        t, n_in = traslacion_ransac(feats[i].kps, feats[j].kps, matches, umbral)
        res[(i, j)] = dict(matches=len(matches), inliers=n_in, t=t)
    return res


def componer_wideview(imgs_proy, mascaras_proy, posiciones, ref, corregir_deriva=True):
    """
    Construye las transformaciones (traslaciones) a partir de la posición global de cada imagen.
    Opcionalmente elimina la deriva vertical ajustando una recta y = a·x + b a las posiciones.
    """
    pos = np.array(posiciones, dtype=np.float64)
    pos = pos - pos[ref]
    if corregir_deriva and len(pos) > 2:
        a, b = np.polyfit(pos[:, 0], pos[:, 1], 1)
        pos[:, 1] -= a * pos[:, 0] + b
    T = [matriz_traslacion(x, y) for x, y in pos]
    return warpear(imgs_proy, mascaras_proy, T)


def stitching_wideview(imgs, metodo="sift", estrategia="flann", f=None, superficie="cilindro",
                       ordenadas=True, blending="multibanda", min_inliers=25, ratio=0.75,
                       corregir_deriva=True):
    """
    Panorama wide-view de N imágenes:

    1. Proyección cilíndrica/esférica (con distancia focal f en píxeles).
    2. Características en las imágenes proyectadas.
    3. Si `ordenadas` es False, se recupera el orden con :func:`ordenar_imagenes`.
    4. Traslación robusta entre imágenes consecutivas y acumulación de posiciones.
    5. Corrección de deriva vertical, warping y blending.
    """
    cv2.setRNGSeed(0)
    f = f or 0.8 * imgs[0].shape[1]
    t0 = time.perf_counter()
    proy, masc = proyectar(imgs, f, superficie)
    m = crear_metodo(metodo)
    feats = [extraer(p, m) for p in proy]

    orden, info_orden = list(range(len(imgs))), None
    if not ordenadas:
        orden, info_orden = ordenar_imagenes(feats, m.tipo_descriptor, estrategia, min_inliers=min_inliers)

    pares = [(orden[k], orden[k + 1]) for k in range(len(orden) - 1)]
    tr = traslaciones_por_pares(feats, m.tipo_descriptor, pares, estrategia, ratio)
    filas, posiciones = [], [np.zeros(2)]
    for (i, j) in pares:
        r = tr[(i, j)]
        if r["t"] is None or r["inliers"] < min_inliers:
            raise RuntimeError(f"No se pudo alinear {i}→{j} ({r['inliers']} inliers)")
        dx, dy = r["t"]
        # un punto en i está en (p + d) en j  ->  la imagen j está desplazada -d respecto a i
        posiciones.append(posiciones[-1] - np.array([dx, dy]))
        filas.append({"Par": f"{i}→{j}", "Matches": r["matches"], "Inliers": r["inliers"],
                      "dx (px)": dx, "dy (px)": dy})

    proy_o = [proy[i] for i in orden]
    masc_o = [masc[i] for i in orden]
    ref = len(orden) // 2
    wi, wm = componer_wideview(proy_o, masc_o, posiciones, ref, corregir_deriva)
    pano = BLENDINGS[blending](wi, wm)
    info = dict(tiempo_total=time.perf_counter() - t0, orden=orden, pares=pd.DataFrame(filas),
                orden_info=info_orden, warped=(wi, wm), f=f)
    return pano, info


def ordenar_imagenes(feats, tipo_descriptor, estrategia="flann", min_inliers=25):
    """
    Recupera el orden izquierda→derecha de un conjunto de imágenes desordenadas.

    1. Empareja todas las parejas (i, j) y estima su traslación con RANSAC.
    2. Construye un grafo cuyas aristas son las parejas con suficientes inliers
       (peso = nº de inliers) y calcula su árbol de expansión máximo (Kruskal):
       así se conservan solo los solapes más fiables.
    3. Recorre el árbol desde un nodo cualquiera acumulando las traslaciones para obtener
       la posición horizontal global de cada imagen.
    4. El orden es simplemente el de las posiciones x de menor a mayor.
    """
    n = len(feats)
    todos = [(i, j) for i in range(n) for j in range(i + 1, n)]
    tr = traslaciones_por_pares(feats, tipo_descriptor, todos, estrategia)

    # Kruskal con union-find
    padre = list(range(n))

    def raiz(a):
        while padre[a] != a:
            padre[a] = padre[padre[a]]
            a = padre[a]
        return a

    aristas = sorted(((r["inliers"], i, j) for (i, j), r in tr.items()
                      if r["t"] is not None and r["inliers"] >= min_inliers), reverse=True)
    arbol = {i: [] for i in range(n)}
    for w, i, j in aristas:
        ri, rj = raiz(i), raiz(j)
        if ri != rj:
            padre[ri] = rj
            dx, dy = tr[(i, j)]["t"]
            arbol[i].append((j, -np.array([dx, dy])))   # posición de j respecto a i
            arbol[j].append((i, np.array([dx, dy])))
    if len({raiz(i) for i in range(n)}) > 1:
        raise RuntimeError("Hay imágenes que no solapan con ninguna otra")

    pos = {0: np.zeros(2)}
    pila = [0]
    while pila:
        a = pila.pop()
        for b, d in arbol[a]:
            if b not in pos:
                pos[b] = pos[a] + d
                pila.append(b)
    orden = sorted(range(n), key=lambda i: pos[i][0])

    matriz = np.zeros((n, n), int)
    for (i, j), r in tr.items():
        matriz[i, j] = matriz[j, i] = r["inliers"]
    return orden, dict(matriz_inliers=matriz, posiciones_x={i: float(pos[i][0]) for i in range(n)})
