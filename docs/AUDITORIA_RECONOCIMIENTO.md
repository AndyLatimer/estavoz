# Auditoría Técnica del Sistema de Reconocimiento — EstaVoz

**Rama de trabajo:** `codex/auditoria-reconocimiento`  
**Fecha de auditoría:** 2026-11-09  
**Alcance:** Inspección integral del motor de extracción de huellas, proyección LDA, indexado, scripts de evaluación/experimentación, exportación a web, cliente ONNX en el navegador y verificación empírica sobre los artefactos reales publicados (`836` actores, `22.946` segmentos de voz y modelo ONNX `voxblink-int8.onnx`).

---

## 1. Resumen Ejecutivo

La inspección directa del código fuente, del historial de commits de Git y de los artefactos reales publicados en `SrHarrison/estavoz` demuestra que **EstaVoz evolucionó de ECAPA-TDNN (192 dims) a VoxBlink2 SimAM-ResNet34 (256 dims) y finalmente a VoxBlink2 SimAM-ResNet100 (256 dims)**, pero varios archivos del repositorio, la documentación, el servidor FastAPI y los scripts de exportación y evaluación quedaron desfasados o contienen incoherencias silenciosas:

1. **Modelo real verificado empíricamente vs. documentado:** `README.md` y el valor por defecto de `despliegue/exportar_voxblink.py` siguen apuntando a **SimAM-ResNet34** (27 MB), mientras que el artefacto real publicado en `frontend/modelo/voxblink-int8.onnx` (`52.379.886` bytes, SHA-256 `763911a57cc9...`, `105` nodos `Conv`/`ConvInteger`, `dim=256`) es inequívocamente **VoxBlink2 SimAM-ResNet100 (`VERSION_MODELO = "3"`)**, renombrado sin el sufijo `-r100` (commit `e56586e`).
2. **Riesgo de incompatibilidad silenciosa:** Tanto ResNet34 (`r34`) como ResNet100 (`r100`) producen embeddings de exactamente **256 dimensiones**, pero pertenecen a espacios vectoriales incompatibles. Ni `data/indice.npz`, ni `frontend/datos/*.bin`, ni el propio `voxblink-int8.onnx` almacenaban metadatos explícitos de versión/arquitectura antes de esta intervención.
3. **Fugas de información (data leakage) en la evaluación histórica:** Se identificaron **5 mecanismos de contaminación** entre referencia y evaluación. Al evaluar sobre las huellas reales de los 836 actores, se comprobó que conservar los clips de entrevista (`/voz_`, seleccionados supervisadamente mediante el propio índice) en la galería del actor evaluado infla el Top-1 en **+1,52 puntos porcentuales** (`83,30 %` frente a `81,79 %` en las 2.767 consultas con múltiples clips independientes) e introduce además 157 consultas de actores con un único clip en `eldoblaje.com` evaluado contra su propio clip `/voz_` derivado.
4. **Falta de calibración y alta vulnerabilidad ante voces desconocidas (verificada en los 836 actores):** La función `aPorcentaje(s)` de `frontend/index.html` aplica la transformación lineal fija `(s - 0.15) / 0.35`. En la evaluación real con 125 actores retenidos fuera de la galería (voces desconocidas), el **69,65 % de las consultas de voces desconocidas superan el 50 % en la interfaz actual (`s1 >= 0.325`)**, disparando erróneamente el mensaje *«¡Corten! La tenemos.»*
5. **Brechas entre los scripts de prueba y el navegador:** Ninguno de los scripts previos (`backend/evaluar.py`, `backend/experimentar.py`, `backend/app.py`) reproducía simultáneamente todas las etapas de `frontend/index.html` + `despliegue/exportar_web.py` (`centroides_mezcla` con peso 0,5 para `/voz_` + cuantización `int8` de `trozos.bin` + mezcla 50 % centroide / 50 % top-3 trozos).

---

## 2. Respuesta Detallada a las 6 Cuestiones de la Fase 1

### 2.1. ¿Qué modelo utiliza realmente la aplicación publicada?

#### Hechos comprobados en el código, historial de Git y artefactos publicados
| Componente | Archivo y líneas | Qué declara / contiene realmente |
|---|---|---|
| Documentación | `README.md` (L16-L21, L54) | Declara **SimAM-ResNet34** (`gaunernst/wespeaker-voxblink2-samresnet34-ft`), **27 MB**, con 69,6 % Top-1 y 78 % Top-3. |
| Exportador ONNX | `despliegue/exportar_voxblink.py` (L62-L63, L71, L77) | Antes de esta rama, si se ejecutaba sin argumentos exportaba `variante = "r34"` a `frontend/modelo/voxblink-int8.onnx`, y si se pasaba `"r100"` exportaba a `frontend/modelo/voxblink-r100-int8.onnx` (con sufijo `-r100`). |
| Arquitecturas PyTorch | `backend/voxblink.py` (L16-L18, L80-L96) | Define dos variantes bajo la misma clase `SimAMResNet34`: `"r34"` con bloques `(3, 4, 6, 3)` y `"r100"` con bloques `(6, 16, 24, 3)` (`bzikst/voxblink2_samresnet100_ft`). Ambas tienen `dim=256`. |
| Exportador Web | `despliegue/exportar_web.py` (L92-L94) | Define `VERSION_MODELO = "3"` (`1 = ECAPA, 2 = VoxBlink2 ResNet34, 3 = VoxBlink2 ResNet100`). |
| Navegador (Frontend) | `frontend/index.html` (L238, L257, L581) | L238: `// el modelo pesa ~52 MB la primera vez`. L257: descarga `modelo/voxblink-int8.onnx`. L581: envía a Supabase `version: '3' // 3 = VoxBlink2 ResNet100`. |
| Artefacto ONNX publicado | `frontend/modelo/voxblink-int8.onnx` | **Verificado con `inspeccionar_onnx()`:** `52.379.886` bytes (`52,38 MB`), SHA-256 `763911a57cc978f0d7a22799c20ff93eadaa4baac5b17ed7751a882cfc111dc8`, `num_nodos_conv = 105`, salida `embs` de dimensión `256`. Confirmado al 100 % como **VoxBlink2 SimAM-ResNet100 cuantizado a int8 (`version_modelo = "3"`)**, sin `metadata_props` inscritos previamente. |
| Artefactos de datos web publicados | `frontend/datos/*` | **Verificados con `validar_artefactos_web()`:** `836` actores en `actores.json` (`dim=256`), `lda.bin` (`263.168` bytes, `mu` + matriz `256×256`), `centroides.bin` (`856.064` bytes, `836×256`), `trozos.bin` (`5.969.308` bytes, `22.946` segmentos cuantizados en `int8`), `busqueda.json` (`16.854.316` bytes, `383.627` entradas de búsqueda contextual). |

---

### 2.2. ¿Existe alguna incompatibilidad entre la versión de las huellas y el modelo ONNX?

#### Hechos comprobados
1. **En producción actual:** Los artefactos publicados en `SrHarrison/estavoz` (`voxblink-int8.onnx`, `lda.bin`, `centroides.bin`, `trozos.bin` y `actores.json`) **son compatibles entre sí** y pertenecen todos a `version_modelo = "3"` (VoxBlink2 SimAM-ResNet100, `dim=256`).
2. **Colisión de dimensionalidad (`dim=256`) entre ResNet34 y ResNet100 en los scripts de mantenimiento:**
   - Tanto `r34` como `r100` devuelven vectores de `256` floats (`backend/voxblink.py` L80).
   - Antes de esta intervención, ni el archivo ONNX tenía `metadata_props` ni `data/indice.npz` guardaba la versión del modelo.
   - Si un desarrollador ejecutaba `python despliegue/exportar_voxblink.py` sin argumentos (como indicaba `README.md` L54), sobrescribía `frontend/modelo/voxblink-int8.onnx` con **ResNet34** (27 MB, 39 nodos Conv) manteniendo `dim=256`, lo que habría corrompido silenciosamente las inferencias frente a `lda.bin` y `centroides.bin` generados con **ResNet100**.
3. **Caída silenciosa a ECAPA-TDNN (`dim=192`) cuando falta `ESTAVOZ_ONNX`:**
   - En `backend/voz.py`, si no se definía `ESTAVOZ_ONNX` y no existía el ONNX por defecto, `huella_trozo()` caía silenciosamente a `speechbrain/spkrec-ecapa-voxceleb` (`dim=192`), y `backend/app.py` invocaba siempre `modelo()` (ECAPA) en `arrancar()`.
4. **Diferencia de ventana temporal entre el índice (6 s) y la consulta del navegador (10 s):**
   - `backend/voz.py` (L13, L65-L68) trocea todos los audios del índice en fragmentos de **`TROZO_SEG = 6` segundos**.
   - `frontend/index.html` (L214-L216, L298-L306) calcula **una única huella de `SEGUNDOS = 10` segundos** por toma (`TROZO = 10 * SR`), y cuando un usuario envía una corrección a Supabase (`L575`), esa huella de 10 s truncada a 4 decimales (`Math.round(v * 1e4) / 1e4`) se concatena en el aprendizaje nocturno (`despliegue/exportar_web.py` L168-L169) con los trozos de 6 s del índice.

---

### 2.3. ¿La evaluación utiliza datos independientes de los utilizados para entrenar la LDA y construir los centroides?

#### Hechos comprobados
1. **`backend/evaluar.py` (L78-L95):**
   - Evalúa sin LDA ni centroides (usa similitud coseno bruta y media de los 3 mejores trozos). Aparta el archivo del clip evaluado (`j != i`).
2. **`backend/experimentar.py` — `evaluar_lda()` (L100-L118) y `evaluar_proyeccion()` (L298-L311):**
   - Divide la lista de actores en `pliegues` (validación cruzada por actores). La matriz LDA `(mu, W)` se entrena con los actores de `pliegue[a] != p`.
   - **Problema detectado:** Al invocar `evaluar(clips_p)` en la línea 116, la galería contra la que compiten las consultas del pliegue `p` contiene **todos los actores**, incluidos los actores de `pliegue[a] != p` cuyos clips **sí se usaron para entrenar la LDA**.
3. **`backend/experimentar.py` — `puntuacion()` (L437-L490, de donde proceden los valores de ~77,9 % / ~85,8 % del commit `04bdec7`):**
   - En la línea 470-474:
     ```python
     qact = sorted({q[0] for q in consultas})
     ...
     sel = np.isin(y, [a for a in actores if n_clips[a] >= 2 and pl.get(a, -1) != p])
     mu, W = entrenar_lda(X[sel], y[sel])
     ```
     `qact` solo contiene el subconjunto de ~250 actores que tienen consultas en `data/cache_huellas/r100-largos2`. Para los otros ~580 actores del índice, `pl.get(a, -1)` devuelve `-1 != p`, por lo que **entran siempre en el entrenamiento de la LDA en todos los pliegues**.
   - Además, en la línea 480 (`filas = np.flatnonzero(base != clip)`), al construir el centroide del actor consultado solo se aparta `base != clip` (`act/nom.mp3` y `act/nom.mp3#movil`), pero se mantienen en el centroide del actor sus clips de `podcast_` y `voz_`, que sufren la contaminación circular descrita en el punto 2.4.

---

### 2.4. ¿Existen fugas de información (data leakage) entre entrenamiento y evaluación?

#### Hechos comprobados (5 vías de fuga identificadas y cuantificadas)

1. **Contaminación circular de galería a través de `podcasts.py`, `entrevistas.py` y `voces.py`:**
   - `scraper/podcasts.py` (L139-L165), `scraper/voces.py` (L119-L165, L227-L236) y `scraper/entrevistas.py` (L200-L219) utilizan los clips de `eldoblaje.com` (y la LDA/centroides entrenados con ellos) para aceptar solo los fragmentos de entrevistas o podcasts donde el actor ya obtiene alta similitud (`>= 0.25` / Top-5 en `voces.py`, `>= 0.35` / Top-1 en `podcasts.py`, `>= 0.40` en `entrevistas.py`).
   - **Verificación cuantitativa sobre los artefactos reales (`frontend/datos/`):**
     - De los `836` actores publicados, `158` actores tienen segmentos `/voz_` ponderados a `0.5` en `centroides_mezcla` (`336` clips `/voz_` frente a `1.706` clips `eldoblaje/podcast`).
     - En las **2.767 consultas** de actores con 2 o más clips independientes de `eldoblaje`, mantener los clips `/voz_` en la galería del actor evaluado da **83,30 % Top-1**, mientras que retirarlos (`--solo-galeria-independiente`) reduce el Top-1 a **81,79 %** (**+1,52 puntos de inflación por fuga circular**).
     - Además, **52 actores** (157 consultas) tenían un único clip en `eldoblaje.com` y pasaron a ser evaluables únicamente porque `voces.py` les añadió un clip `/voz_` seleccionado usando ese mismo clip de `eldoblaje.com` como plantilla.
2. **Ausencia de metadatos de obra/escena entre muestras de `eldoblaje.com`:**
   - `scraper/eldoblaje.py` (L28, L136-L141) descarga `Muestras.mp3`, `Saludos.mp3`, `Muestras3.mp3` y `Muestras4.mp3` sin vincular a qué película o sesión pertenece cada corte.
3. **Múltiples clips de entrevista de un mismo programa o episodio:**
   - En `scraper/voces.py`, `bajar_trozos()` nombra cada archivo según `sha1(ep["mp3"])[:12]`, pudiendo separar dos partes de una misma entrevista en archivos distintos.
4. **Fuga global de media en `preparar(..., centrar=True)` (`backend/experimentar.py` L39-L40, L49):**
   - Calcula `mu = V.mean(0)` incluyendo todos los vectores de consulta antes de apartarlos.
5. **Fuga en estadísticas de cohorte AS-norm (`backend/experimentar.py` L60-L66):**
   - `cent_todos` se calcula con todos los clips antes de computar `mu_a, sd_a`.

---

### 2.5. ¿Los porcentajes mostrados por la interfaz están correctamente calibrados?

#### Hechos comprobados y medidos sobre los 836 actores
En `frontend/index.html` (L523-L526 y L590):
```javascript
const aPorcentaje = s => Math.round(Math.max(0, Math.min(1, (s - 0.15) / 0.35)) * 100);
...
estado.textContent = pct >= 50 ? '¡Corten! La tenemos.' : 'Toma dudosa… ¿será alguno de estos?';
```
1. **Sobreconfianza sistemática medida en el conjunto real:**
   - Sobre el conjunto mixto de evaluación (voces conocidas + voces desconocidas), `aPorcentaje(s)` asigna una **confianza media del 79,60 %** frente a una **tasa de acierto real del 72,08 %** (`ECE = 0.0752`, `Brier = 0.1268`).
   - En las consultas de voces conocidas de `6 s`, `12 s` y `18 s`, la similitud media Top-1 es `0.4967`, lo que se traduce casi siempre en porcentajes cercanos al `99 % - 100 %` (`s >= 0.50` satura `aPorcentaje(s)` en `100 %`).
2. **Vulnerabilidad crítica ante voces desconocidas (demostrada empíricamente):**
   - Al retener 125 actores reales fuera de la galería (711 actores en galería) y evaluar sus consultas como **voces desconocidas**, el **69,65 % de las consultas de voces desconocidas obtienen `aPorcentaje(s) >= 50 %` (`s1 >= 0.325`)**, mostrando al usuario el mensaje **«¡Corten! La tenemos.»** con porcentajes de hasta el **85 %** para actores que ni siquiera están en la base de datos.
3. **Mejora mediante calibración isotónica (`CalibradorConfianza` en `backend/rechazo.py`):**
   - El calibrador isotónico PAVA entrenado sobre la partición de validación reduce el **Error de Calibración Esperado (ECE) de `0.0752` a `0.0137`** (reducción del **81,8 %** del error de calibración) y el **Brier Score de `0.1268` a `0.1018`**.

---

### 2.6. ¿Las pruebas actuales reproducen fielmente el funcionamiento del navegador?

#### Hechos comprobados
Ninguno de los scripts previos reproducía fielmente el navegador; la tabla siguiente resume las diferencias corregidas en `backend/sistema_evaluacion.py`:

| Etapa del pipeline | Navegador (`frontend/index.html` + `exportar_web.py`) | `backend/evaluar.py` | `backend/experimentar.py` | `backend/app.py` | `backend/sistema_evaluacion.py` (Nuevo) |
|---|---|---|---|---|---|
| Ventana de consulta | **10 s en una sola huella** | 8 s partidos en **6 s + 2 s** | 8/10 s en **6 s + 4 s** (10 s en `puntuacion`) | Hasta 20 s en **6 s** | Soporta **10 s** (`cache_huellas`) y evaluación multi-duración (`6 s`, `12 s`, `18 s`) |
| Proyección LDA | Sí (`lda.bin`, `reg=1e-2`) | **No** | Sí | **No** | **Sí** (LDA por pliegues sin fuga o `lda.bin` publicada) |
| Centroides con `/voz_` a peso 0,5 (`centroides_mezcla`) | **Sí** (`exportar_web.py` L127-L141) | **No** | **No** (`puntuacion()` hacía media simple) | **No** | **Sí** (`centroides_mezcla_web`) |
| Cuantización `int8` de `trozos.bin` | **Sí** (`escala = abs(V).max(1)/127`, `int8`) | **No** (`float32`) | **No** (`float32`) | **No** (`float32`) | **Sí** (`cuantizar_descuantizar_int8`) |
| Fórmula de puntuación | `0.5 * centroide + 0.5 * media_top3_trozos` | `1.0 * media_top3_trozos` | `1.0 * centroide` o `1.0 * top3` | `1.0 * media_top3_trozos` | **Sí** (`0.5 * centroide + 0.5 * media_top3_trozos`) |
| Búsqueda contextual y voces desconocidas | Filtro contextual `busqueda.json`; sin rechazo de desconocidas | **No** | **No** | **No** | **Sí** (abierta vs contextual real con `busqueda.json` + rechazo 3 estados) |
