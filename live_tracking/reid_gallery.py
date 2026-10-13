"""
reid_gallery.py
----------------
Reidentificacion por apariencia para trackers SIN ReID interno (ByteTrack,
cjm_bytetrack, OcSort). A diferencia de AppearanceGalleryBank (en
damages_tracking/botsort_boxmot_reid_eval.py), que recalcula embeddings de
TODAS las detecciones en TODOS los frames (para refrescar via EMA la huella
de los tracks activos), esta version solo llama al modelo de ReID en dos
momentos por segmento de trayectoria: cuando el tracker crea un id nuevo
(para buscar un match contra identidades perdidas) y cuando un id activo se
pierde (para dejar su huella lista en la galeria). Mucho mas barato -- tiene
sentido para trackers ligeros como ByteTrack, que de otra forma nunca
pagarian el costo de un forward de ReID por frame.

Por diseno, el banco SOLO contiene identidades actualmente perdidas: una
identidad activa nunca entra al banco, así que no hace falta excluir
"final_ids activos este frame" de match_newborns como en la version original
(ver AppearanceGalleryBank) -- si esta activa, no esta en el banco.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import lap


class ReidGallery:
    def __init__(self, cosine_thresh: float, max_gap: int, debug: bool = False,
                 viz_dir: Optional[str] = None, viz_top_k: int = 5):
        self.cosine_thresh = cosine_thresh
        self.max_gap = max_gap
        self.debug = debug
        self.viz_dir = viz_dir
        self.viz_top_k = viz_top_k
        # final_id -> {"embeds": (K,D) unit vectors, "frames_lost": int,
        #              "crops": list[np.ndarray]|None (hasta bank_samples recortes
        #              representativos, no solo el ultimo -- ver bury()), "buried_frame": int|None}
        self._bank: Dict[int, Dict] = {}
        self._viz_counter = 0

    @property
    def bank_size(self) -> int:
        """Cuantas identidades hay actualmente perdidas en el banco -- util
        para distinguir, desde afuera, un nacimiento sin nada contra que
        comparar (banco vacio) de un intento genuino de reid que no
        matcheo (banco no vacio pero ningun candidato supero el umbral)."""
        return len(self._bank)

    def bury(self, final_id: int, embeds: np.ndarray,
             crops: Optional[np.ndarray] = None,
             frame_id: Optional[int] = None) -> None:
        """Registra la huella de apariencia de una identidad que recien se
        perdio. `embeds` acepta un solo embedding (D,), como antes, O varias
        instancias (K,D) muestreadas a lo largo de TODA la vida del track
        (no solo el ultimo recorte) -- ver `match_newborns`, que promedia la
        similitud contra las K instancias en vez de comparar contra una
        sola. `crops` (opcional, para la visualizacion, solo hace falta si
        se quiere generar `viz_dir`) acepta un solo recorte de pixeles (como
        antes) O una lista de varios recortes representativos -- ANTES solo
        se guardaba uno (el mas reciente) aunque `--reid-stitch-bank-samples`
        fuera >1, asi que la visualizacion nunca podia mostrar mas de 1 frame
        por identidad del banco por mas muestras que tuviera el embedding;
        ahora `_save_comparison` puede dibujar un filmstrip con todas.
        `frame_id` (opcional) es el frame en el que se entierra -- se guarda
        para poder reportar el gap real en `match_newborns` y para poder
        cruzar el log de debug contra eventos de oclusion conocidos por
        frame."""
        embeds = np.atleast_2d(embeds)
        embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)
        if crops is not None and not isinstance(crops, list):
            crops = [crops]
        self._bank[final_id] = {"embeds": embeds, "frames_lost": 0,
                                 "crops": [c.copy() for c in crops] if crops else None,
                                 "buried_frame": frame_id}
        if self.debug:
            print(f"[GALLERY] bury frame={frame_id} final_id={final_id} "
                  f"n_samples={embeds.shape[0]} bank_size={len(self._bank)}")

    def age_and_prune(self, frame_id: Optional[int] = None) -> None:
        """Se llama una vez por frame. Todas las entradas del banco son, por
        diseno, identidades actualmente perdidas -- match_newborns() saca del
        banco a las que se reactivan, asi que aca simplemente se envejecen
        todas y se descartan las que superan max_gap."""
        for fid in list(self._bank.keys()):
            self._bank[fid]["frames_lost"] += 1
            if self._bank[fid]["frames_lost"] > self.max_gap:
                if self.debug:
                    print(f"[GALLERY] prune frame={frame_id} final_id={fid} "
                          f"frames_lost={self._bank[fid]['frames_lost']} max_gap={self.max_gap}")
                del self._bank[fid]

    def match_newborns(self, embeds: np.ndarray,
                        crops: Optional[List[Optional[np.ndarray]]] = None,
                        frame_id: Optional[int] = None,
                        raw_ids: Optional[List[int]] = None) -> List[Optional[int]]:
        """Empareja cada embedding de un id recien nacido (uno por fila)
        contra el banco de perdidos, via asignacion hungara (lap.lapjv)
        gateada por umbral de similitud coseno -- misma logica que
        AppearanceGalleryBank.match_newborns(). Devuelve, por fila, el
        final_id emparejado o None. Las identidades emparejadas se retiran
        del banco (vuelven a estar activas).

        En modo debug, cada intento SIN match imprime la similitud contra
        TODOS los candidatos del banco (no solo el mejor) -- para poder
        verificar, dado un final_id conocido, si efectivamente estaba en el
        banco en ese momento y con que similitud (y no solo "el mejor
        candidato no relacionado"). Los matches exitosos imprimen tambien
        `buried_frame`/`gap` reales (frame_id - frame en que se entero esa
        identidad), para poder cruzar contra `--reid-stitch-max-gap`.

        Si `viz_dir` esta seteado y se pasan `crops` (un recorte de pixeles
        por fila, alineado con `embeds`), CADA intento contra un banco no
        vacio guarda una imagen (`viz_dir/newborn_NNNN_fFRAME_id{ID}_{status}.jpg`)
        con el recorte nuevo y los `viz_top_k` candidatos del banco contra los
        que se comparo (cada uno con su similitud coseno y un filmstrip con
        TODOS los recortes guardados de esa identidad en el banco, no solo
        uno -- verde el que matcheo, naranja el mejor candidato de un intento
        sin match (no llego al umbral), rojo el resto). Guardar tambien los
        intentos SIN match (antes solo se guardaban los aciertos) sirve para
        ver a ojo los "casi" -- candidatos con similitud alta pero por debajo
        de `cosine_thresh`.

        `raw_ids` (opcional, alineado con `embeds`/`crops`): el id resultante
        (final_id emparejado si matcheo, o el propio raw_id si no) se tagea
        en el nombre del archivo -- exactamente el mismo valor que termina en
        la columna `tid_after`/MOT de esta identidad. Sin esto, el nombre del
        archivo solo tenia el numero de frame, y `build_reid_error_gallery`
        (dinov2_reid_finetune.py) no podia distinguir dos comparaciones que
        cayeran en el MISMO frame (2+ nacimientos simultaneos) -- terminaba
        quedandose con cualquiera de las dos al indexar por frame, pudiendo
        mostrar la comparacion de una identidad distinta a la que en
        realidad generaba el evento de error."""
        n = len(embeds)
        results: List[Optional[int]] = [None] * n
        lost_ids = list(self._bank.keys())
        if not lost_ids or n == 0:
            if self.debug and n > 0:
                print(f"[GALLERY] nomatch frame={frame_id} n_newborns={n} bank_size=0")
            return results

        # Cada candidato puede tener K muestras (bury con varias instancias
        # de su vida, no solo la ultima) -- la similitud contra ese
        # candidato es el PROMEDIO de la similitud contra sus K muestras,
        # no solo una comparacion 1-a-1. Con K=1 (bury de un solo embedding,
        # comportamiento de siempre) esto es identico a lo anterior.
        sims = np.empty((n, len(lost_ids)), dtype=np.float32)
        for ci, fid in enumerate(lost_ids):
            bank_embeds = self._bank[fid]["embeds"]  # (K_c, D)
            sims[:, ci] = (embeds @ bank_embeds.T).mean(axis=1)
        cost = 1.0 - sims
        cost_limit = 1.0 - self.cosine_thresh
        _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=cost_limit)
        for r, c in enumerate(x):
            matched_id = lost_ids[c] if c >= 0 else None
            if self.viz_dir is not None and crops is not None \
                    and r < len(crops) and crops[r] is not None:
                raw_id = raw_ids[r] if raw_ids is not None and r < len(raw_ids) else None
                tag_id = matched_id if matched_id is not None else raw_id
                self._save_comparison(crops[r], sims[r], lost_ids, matched_id, frame_id=frame_id,
                                       tag_id=tag_id)
            if c >= 0:
                results[r] = lost_ids[c]
                if self.debug:
                    buried_frame = self._bank[lost_ids[c]]["buried_frame"]
                    gap = (frame_id - buried_frame) if (frame_id is not None and buried_frame is not None) else None
                    cand = ",".join(f"id{fid}:{sims[r, i]:.3f}" for i, fid in enumerate(lost_ids))
                    print(f"[GALLERY] match frame={frame_id} newborn={r} final_id={lost_ids[c]} "
                          f"sim={sims[r, c]:.3f} thresh={self.cosine_thresh} "
                          f"buried_frame={buried_frame} gap={gap} candidates={cand}")
                del self._bank[lost_ids[c]]
            elif self.debug:
                cand = ",".join(f"id{fid}:{sims[r, i]:.3f}" for i, fid in enumerate(lost_ids))
                best_c = int(np.argmax(sims[r])) if sims.shape[1] else None
                best_id = lost_ids[best_c] if best_c is not None else None
                best_sim = float(sims[r, best_c]) if best_c is not None else None
                print(f"[GALLERY] nomatch frame={frame_id} newborn={r} best_final_id={best_id} "
                      f"best_sim={best_sim if best_sim is None else round(best_sim, 3)} "
                      f"thresh={self.cosine_thresh} candidates={cand}")
        return results

    def _save_comparison(self, newborn_crop: np.ndarray, sims_row: np.ndarray, lost_ids: List[int],
                          matched_id: Optional[int], tile_size: int = 128,
                          frame_id: Optional[int] = None, tag_id: Optional[int] = None,
                          thumb_size: int = 32) -> None:
        """`matched_id=None` = intento SIN match (nomatch): igual se guarda
        la imagen, resaltando en naranja el mejor candidato del banco (el que
        perdio por umbral, no por asignacion hungara) en vez de verde.

        Cada candidato se dibuja con su tile principal (el recorte MAS
        RECIENTE guardado, mismo criterio visual que antes) mas un filmstrip
        de miniaturas debajo con TODOS los recortes guardados de esa
        identidad en el banco (hasta `--reid-stitch-bank-samples`, ver
        `bury()`) -- antes solo se guardaba/mostraba un unico recorte por
        identidad sin importar cuantas muestras tuviera el embedding."""
        import cv2
        order = np.argsort(-sims_row)[:self.viz_top_k]
        cols = 1 + len(order)
        header_h = 20
        n_thumb_rows = max(1, -(-tile_size // thumb_size))  # ceil: cuantas filas de miniaturas caben en tile_size
        strip_h = n_thumb_rows * thumb_size
        canvas = np.full((header_h + tile_size + strip_h, cols * tile_size, 3), 30, dtype=np.uint8)

        q = cv2.resize(newborn_crop, (tile_size, tile_size))
        canvas[header_h:header_h + tile_size, 0:tile_size] = q
        cv2.rectangle(canvas, (0, header_h), (tile_size - 1, header_h + tile_size - 1), (0, 255, 255), 3)
        cv2.putText(canvas, "NUEVO", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        best_idx = int(np.argmax(sims_row)) if len(sims_row) else None
        thumbs_per_row = max(1, tile_size // thumb_size)
        for k, idx in enumerate(order):
            fid = lost_ids[idx]
            entry = self._bank.get(fid)
            crops = entry["crops"] if entry is not None else None
            x0 = (k + 1) * tile_size
            main_crop = crops[-1] if crops else None  # el mas reciente, mismo criterio que antes
            if main_crop is not None:
                tile = cv2.resize(main_crop, (tile_size, tile_size))
            else:
                tile = np.full((tile_size, tile_size, 3), 60, dtype=np.uint8)
            canvas[header_h:header_h + tile_size, x0:x0 + tile_size] = tile
            if fid == matched_id:
                color = (0, 200, 0)      # verde: matcheo real
            elif matched_id is None and idx == best_idx:
                color = (0, 165, 255)    # naranja: mejor candidato, no llego al umbral
            else:
                color = (0, 0, 220)      # rojo: descartado
            cv2.rectangle(canvas, (x0, header_h), (x0 + tile_size - 1, header_h + tile_size - 1), color, 3)
            cv2.putText(canvas, f"id{fid} sim={sims_row[idx]:.2f}", (x0 + 4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if crops:
                y0 = header_h + tile_size
                for j, c in enumerate(crops):
                    row, col = divmod(j, thumbs_per_row)
                    ty0 = y0 + row * thumb_size
                    tx0 = x0 + col * thumb_size
                    if ty0 + thumb_size > canvas.shape[0] or tx0 + thumb_size > x0 + tile_size:
                        break  # mas muestras de las que caen en el grid del tile -- se recorta, no se agranda el canvas
                    canvas[ty0:ty0 + thumb_size, tx0:tx0 + thumb_size] = cv2.resize(c, (thumb_size, thumb_size))
                    cv2.rectangle(canvas, (tx0, ty0), (tx0 + thumb_size - 1, ty0 + thumb_size - 1), color, 1)

        self._viz_counter += 1
        os.makedirs(self.viz_dir, exist_ok=True)
        status = f"final{matched_id}" if matched_id is not None else "nomatch"
        frame_tag = f"_f{frame_id}" if frame_id is not None else ""
        id_tag = f"_id{tag_id}" if tag_id is not None else ""
        out_path = os.path.join(self.viz_dir, f"newborn_{self._viz_counter:04d}{frame_tag}{id_tag}_{status}.jpg")
        cv2.imwrite(out_path, canvas)
