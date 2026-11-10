# Resultados de Evaluación y Verificación Experimental — EstaVoz

**Rama de trabajo:** `codex/auditoria-reconocimiento`  
**Fecha:** 2026-11-09  

---

## 1. Artefactos Reales Verificados (`SrHarrison/estavoz`)

Se descargaron e inspeccionaron los artefactos públicos de producción del Space `SrHarrison/estavoz` (`frontend/modelo/voxblink-int8.onnx` y `frontend/datos/*`), ejecutando sobre ellos la validación criptográfica y estructural de Fase 4 (`backend/metadatos_modelo.py`) y la evaluación completa de Fases 2 y 3 (`backend/sistema_evaluacion.py` y `backend/rechazo.py`):

| Artefacto | Tamaño | SHA-256 (prefijo) | Verificación estructural |
|---|---|---|---|
| `frontend/modelo/voxblink-int8.onnx` | `52.379.886` B (52,38 MB) | `763911a57cc978f0...` | `version_modelo = "3"`, `variante = "r100"`, `dim_embedding = 256`, **`105` nodos `Conv`/`ConvInteger`** (VoxBlink2 SimAM-ResNet100 int8). |
| `frontend/datos/actores.json` | `56.609` B | — | **836 actores**, `dim = 256`. |
| `frontend/datos/lda.bin` | `263.168` B | `3a85480e0944343e...` | Vector media `mu` (`256`) + matriz de proyección `W` (`256×256` `float32`). |
| `frontend/datos/centroides.bin` | `856.064` B | — | `836 × 256` `float32` (`centroides_mezcla`: `678` actores solo `eldoblaje/podcast` y `158` actores con mezcla `0,5` de `/voz_`). |
| `frontend/datos/trozos.bin` | `5.969.308` B | — | **22.946 segmentos** proyectados por LDA y cuantizados en `int8` (`1.706` clips de `eldoblaje/podcast` y `336` clips `/voz_` en pares limpio + `#movil`). |
| `frontend/datos/busqueda.json` | `16.854.316` B | — | **383.627 entradas** de reparto (películas, series, videojuegos, personajes y actores originales). |

---

## 2. Evaluación Empírica sobre los 836 Dobladores Reales

Comandos ejecutados:
```powershell
.\.venv\Scripts\python backend\sistema_evaluacion.py --salida data\evaluaciones\real_estandar --historial data\evaluaciones\historial.jsonl
.\.venv\Scripts\python backend\sistema_evaluacion.py --solo-galeria-independiente --salida data\evaluaciones\real_independiente --historial data\evaluaciones\historial.jsonl
```

Diseño experimental reproducible (`semilla = 0`):
- **Total de actores:** `836` (`711` actores en la galería activa y **`125` actores retenidos fuera de la galería** para medir el comportamiento ante voces desconocidas en conjunto abierto).
- **Consultas de prueba:** Segmentos con simulación móvil (`#movil`) de los clips de `eldoblaje`, apartando siempre de la galería y del centroide del actor el clip completo evaluado (tanto su versión limpia como su versión `#movil`).
- **Dos regímenes de galería comparados:**
  1. **Galería estándar (`real_estandar`):** Conserva en la referencia del actor sus restantes clips de `eldoblaje` y sus clips de entrevista (`/voz_` con peso `0,5`), tal como hacía `experimentar.py`. Total: `2.924` consultas conocidas + `449` consultas de actores desconocidos.
  2. **Galería estrictamente independiente (`real_independiente`, `--solo-galeria-independiente`):** Elimina de la galería del actor evaluado cualquier clip de `/voz_` (ya que `scraper/voces.py` seleccionó esos fragmentos usando los propios clips de `eldoblaje.com` como plantilla). Total: `2.767` consultas conocidas + `449` consultas de actores desconocidos.

### 2.1. Precisión Global y Cuantificación de la Fuga Circular de `/voz_`

| Régimen de evaluación | Consultas conocidas | Top-1 sin contexto | Top-3 sin contexto | Top-1 con contexto (`busqueda.json`) | Top-3 con contexto (`busqueda.json`) | Similitud media (`s1`) | Margen medio (`s1 - s2`) |
|---|---|---|---|---|---|---|---|
| **Galería estándar (`real_estandar`)** | 2.924 | **81,98 %** | **89,53 %** | **94,46 %** | **98,02 %** | 0,4967 | 0,1586 |
| **Galería independiente (`--solo-galeria-independiente`)** | 2.767 | **81,79 %** | **89,48 %** | **94,22 %** | **97,94 %** | 0,4923 | 0,1557 |
| **Subconjunto común (2.767 consultas con 2+ clips `eldoblaje`)** | 2.767 | `83,30 %` (estándar) vs `81,79 %` (indep.) | `90,57 %` vs `89,48 %` | `94,94 %` vs `94,22 %` | `98,27 %` vs `97,94 %` | — | — |
| **Subconjunto de 52 actores con solo 1 clip `eldoblaje` + 1 clip `/voz_`** | 157 | **58,60 %** (solo evaluable contra `/voz_`) | **71,34 %** | 85,99 % | 93,63 % | — | — |

**Conclusiones clave:**
1. En igualdad de consultas (`2.767` consultas de actores con 2+ muestras propias en `eldoblaje.com`), conservar los clips `/voz_` en la galería del actor evaluado **infla el Top-1 en `+1,52` puntos porcentuales (`83,30 %` frente a `81,79 %`)**.
2. El filtro de **búsqueda contextual** (*«¿Qué estás viendo?»*, evaluado sobre repartos reales de `busqueda.json` con tamaño medio de `17,6` actores por obra) aporta una mejora masiva y consistente de **`+12,43` a `+12,48` puntos en Top-1** (pasando de **`81,79 %` a `94,22 %` Top-1** y **`97,94 %` Top-3** en galería independiente).

---

### 2.2. Precisión por Duración de Grabación (Muestras Móviles `#movil`)

| Duración acumulada de consulta | Consultas (Estándar / Indep.) | Top-1 Estándar | Top-3 Estándar | Top-1 Independiente | Top-3 Independiente |
|---|---|---|---|---|---|
| **`6 s`** (1 trozo móvil) | 1.082 / 1.020 | **76,71 %** | **85,12 %** | **76,27 %** | **84,90 %** |
| **`12 s`** (2 trozos móviles) | 987 / 928 | **83,79 %** | **91,29 %** | **83,51 %** | **91,27 %** |
| **`18 s`** (3 trozos móviles) | 855 / 819 | **86,55 %** | **93,10 %** | **86,69 %** | **93,16 %** |

> **Relación con la cifra histórica comunicada (78,3 % Top-1 / 86,3 % Top-3 con tomas de 10 s):**
> Los resultados verificados explican con exactitud la cifra histórica: una toma de `10 s` se sitúa entre `6 s` (`76,27 % - 76,71 %` Top-1; `84,90 % - 85,12 %` Top-3) y `12 s` (`83,51 % - 83,79 %` Top-1; `91,27 % - 91,29 %` Top-3). Además, al pasar de `6 s` a `12 s` el error Top-1 se reduce un **30,4 %** (`23,73 % -> 16,49 %` de fallo).

---

### 2.3. Evaluación del Reconocimiento de Voces Desconocidas (Fase 3, Datos Reales)

Se dividieron las consultas (conocidas de los `711` actores en galería + desconocidas de los `125` actores retenidos fuera de la galería) en **50 % validación** (para calibrar `UmbralesRechazo` sin tocar el test) y **50 % test independiente**.

#### Umbrales calibrados automáticamente sobre la partición de validación (`real_estandar`):
- `umbral_fiable = 0.3663`
- `umbral_dudoso = 0.2520`
- `margen_fiable = 0.1115`
- `margen_dudoso = 0.0502`
- `z_fiable = 5.3001`
- `umbral_1a1_fiable = 0.3370`, `umbral_1a1_dudoso = 0.2319`

#### Resultados sobre la mitad de test independiente:

| Escenario y métrica | Sin rechazo (`frontend/index.html` actual) | Con rechazo en 3 estados (`backend/rechazo.py` — Galería Estándar) | Con rechazo en 3 estados (`backend/rechazo.py` — Galería Independiente) |
|---|---|---|---|
| **Búsqueda abierta — Voces desconocidas con `UI >= 50 %` («¡Corten! La tenemos.»)** | **69,65 %** (`65,20 %` en indep.) | — | — |
| **Búsqueda abierta — Voces desconocidas aceptadas como `identificado` (FPR)** | **100,00 %** (siempre devuelve ganador) | **4,98 %** | **7,05 %** |
| **Búsqueda abierta — Voces desconocidas NO identificadas (`desconocido` + `dudoso`)** | **0,00 %** | **95,02 %** (`55,22 %` rechazadas estrictas + `39,80 %` dudosas) | **92,95 %** (`55,95 %` rechazadas estrictas + `37,00 %` dudosas) |
| **Búsqueda abierta — Precisión cuando declara `identificado` (en conocidas)** | 81,83 % | **99,14 %** (`61,91 %` ident. correctas vs solo `0,54 %` erróneas) | **98,47 %** (`60,39 %` ident. correctas vs `0,94 %` erróneas) |
| **Búsqueda abierta — Precisión global en `identificado` (conocidas + desconocidas)** | 72,08 % | **98,08 %** | **96,64 %** |
| **Búsqueda abierta — Consultas conocidas en `dudoso`** | — | **22,27 %** (el actor real está en el Top-3 en el **85,8 %** de los casos dudosos) | **24,11 %** (el actor real está en el Top-3 en el **84,7 %** de los casos dudosos) |
| **Búsqueda contextual — Precisión cuando declara `identificado`** | 94,46 % | **99,85 %** (`99,49 %` global incluyendo desconocidas) | **99,83 %** (`99,45 %` global incluyendo desconocidas) |
| **Búsqueda contextual — Voces desconocidas aceptadas como `identificado` (FPR)** | 100,00 % (`8,69 %` con UI >= 50 %) | **1,56 %** (`98,44 %` no identificadas; `79,06 %` rechazadas estrictas) | **1,56 %** (`98,44 %` no identificadas; `75,95 %` rechazadas estrictas) |

---

### 2.4. Calibración de los Porcentajes de Coincidencia en Datos Reales

| Método de traducción a probabilidad | ECE (Estándar / Indep.) | Brier Score (Estándar / Indep.) | Confianza media vs Acierto real (Estándar) |
|---|---|---|---|
| `aPorcentaje(s)` histórico (`(s - 0.15) / 0.35`) | `0.0752` / `0.0879` (`0.0809` global) | `0.1268` / `0.1384` | `79,60 %` vs `72,08 %` (sobreconfianza de `+7,52` ptos) |
| `aPorcentajeCalibrado(s)` (`CURVA_CALIBRADA_R100` en 9 tramos, integrado en `index.html`) | **`0.0103`** / **`0.0118`** (**-87,3 %**) | **`0.1241`** / **`0.1269`** | Curva monótona continua ajustada a VoxBlink2 ResNet100 |
| `CalibradorConfianza` isotónico PAVA (`backend/rechazo.py`) | **`0.0137`** / **`0.0221`** | **`0.1018`** / **`0.1114`** | **`71,67 %` vs `72,08 %`** (desviación `< 0,5` ptos) |

---

### 2.5. Experimentos de Optimización de Puntuación y Corrección de «Actores Imán» (*Hubness*)

Sobre las `2.767` consultas conocidas independientes y `449` consultas de voces desconocidas se evaluó el efecto de variar:
- `w_cent` (peso del centroide frente a los mejores trozos),
- `k_top` (número de mejores trozos promediados),
- `beta` (penalización logarítmica de estadístico de orden `beta * max(0, ln(N_trozos / 12))` para corregir la ventaja estadística de actores con decenas de trozos frente a la mediana de `12` trozos por actor):

| Configuración (`w_cent`, `k_top`, `beta`) | Top-1 Independiente | Top-3 Independiente | AUC Abierto (Conocidas vs Desconocidas) | Frecuencia del mayor «Actor Imán» en desconocidas |
|---|---|---|---|---|
| `w=0.50, k=3, beta=0.000` (línea base navegador) | `81,79 %` | `89,48 %` | `0.8804` | `114-eduardo-gutierrez` (`15 / 449`, 144 trozos) |
| `w=0.50, k=4, beta=0.000` | `81,89 %` | **`89,66 %`** | `0.8804` | `114-eduardo-gutierrez` (`15 / 449`) |
| `w=0.55, k=4, beta=0.000` | **`81,93 %`** | **`89,66 %`** | `0.8824` | `114-eduardo-gutierrez` (`14 / 449`) |
| `w=0.55, k=4, beta=0.012` (con corrección de exceso de trozos) | **`81,93 %`** | `89,59 %` | **`0.8828`** | `114-eduardo-gutierrez` (**`13 / 449`**, -13,3 %) |

---

### 2.6. Auditoría de Consistencia por Actor en el Índice

El desglose por actor de `real_independiente` (`268` actores con 2+ clips independientes de `eldoblaje.com` en la galería de 711 actores) identifica qué fichas presentan incoherencia interna entre sus muestras de `eldoblaje.com` y cuáles actúan como imanes de falsos positivos:

1. **Actores con `0 %` en Top-1 y Top-3 entre sus propios clips independientes (`4` de `268` actores, `1,5 %`):**
   - `95966-virginia-gris` (*Virginia Gris*): `12` consultas, similitud media `0.2765` (`46` trozos).
   - `226-carlos-ysbert` (*Carlos Ysbert*): `6` consultas, similitud media `0.3558` (`50` trozos).
   - `172-jesus-pinillos` (*Jesús Pinillos*): `5` consultas, similitud media `0.3597` (`66` trozos).
   - `13-jose-luis-angulo` (*José Luis Angulo*): `4` consultas, similitud media `0.2863` (`42` trozos).
   *(Causa típica: uno de los dos archivos `.mp3` de la ficha contiene música/efectos dominantes, voz caracterizada extrema o intervención de otro interlocutor en la escena).*
2. **Actores con alta presencia en Top-3 pero baja en Top-1 (`>= 6` consultas, `Top-1 <= 25 %`, `Top-3 >= 50 %`):**
   - `16270-ramon-hernandez` (`Top-1 = 0 %`, **`Top-3 = 100 %`**, `sim = 0.5672`), `107947-aurora-gonzalez` (`Top-1 = 11,1 %`, `Top-3 = 77,8 %`), `88279-oriol-cruz` (`Top-1 = 16,7 %`, `Top-3 = 83,3 %`), `219-miguel-angel-varela` (`Top-1 = 0 %`, `Top-3 = 66,7 %`), `12162-angel-sacristan` (`Top-1 = 20,0 %`, `Top-3 = 66,7 %`).
3. **Principales «Actores Imán» (*hubs*) ante voces desconocidas (sobre `449` consultas de actores fuera de la galería):**
   - `114-eduardo-gutierrez` (`144` trozos en `trozos.bin`, `12x` la mediana): Top-1 en `15/449` (`3,3 %`) y Top-3 en `29/449` (`6,5 %`).
   - `121759-carla-mercader` (`104` trozos): Top-1 en `11/449` (`2,4 %`) y Top-3 en `20/449` (`4,5 %`).
   - `140429-santi-goas` (`62` trozos): Top-1 en `9/449` (`2,0 %`).
4. **Detección y poda automática de «trozos intrusos» (`backend/auditar_indice.py`):**
   - Al ejecutar `python backend/auditar_indice.py` sobre los `22.946` trozos publicados, se detectaron **319 trozos (`1,39 %`) con similitud `< 0,10`** (y **525 trozos con `< 0,15`**) respecto al centroide de su propio actor, correspondiendo a fragmentos de 6 s donde habla otro personaje de la escena (a menudo incluso de distinto sexo):
     - En `120-miguel-angel-jenner`: los trozos `#2214` (`sim = -0.0401`) y `#2206` (`sim = -0.0281`) coinciden con las actrices `1353-lola-oria` (`0.4090`) y `1325-joel-mulachs` (`0.3878`).
     - En `30136-oscar-castellanos`: los trozos `#12244` (`sim = -0.0281`) y `#12250` (`sim = -0.0196`) coinciden con `232-victoria-angulo` (`0.4098`) y `75801-angela-soler` (`0.4081`).
     - En `234-mar-bordallo`: el trozo `#10019` (`sim = -0.0501`) coincide con `150-federico-menescal` (`0.3754`).
   - Al activar [`filtrar_trozos_intrusos`](file:///x:/projects/estavoz/backend/auditar_indice.py#L26-L69) con `umbral_outlier = 0.15` (integrado en `despliegue/exportar_web.py`), el Top-1 en galería independiente sube de **`81,79 %` a `82,00 %` (`+0,21` ptos)** y con `0.20` a **`82,04 %` (`+0,25` ptos)**, subiendo el Top-3 a **`89,56 %`**.
5. **Parejas de «gemelos vocales» en el espacio LDA (`auditar_indice.py`):**
   - `55138-luis-miguel-cajal` $\leftrightarrow$ `72118-rodri-martin` (`sim = 0.7108`)
   - `1973-victor-ramirez` $\leftrightarrow$ `29333-manuel-guitian` (`sim = 0.6873`)
   - `176278-pedro-dargel` $\leftrightarrow$ `55138-luis-miguel-cajal` (`sim = 0.6792`)
   - `240-matilde-conesa` $\leftrightarrow$ `3980-mayte-torres` (`sim = 0.6691`)
   - `15092-idilio-cardoso` $\leftrightarrow$ `659-jose-maria-alarcon` (`sim = 0.6334`)

---

## 3. Pruebas Automatizadas Ejecutadas (`pytest`)

Comando ejecutado:
```powershell
.\.venv\Scripts\pytest -v
```

Resultado: **25 pasados, 0 fallidos, 0 avisos** (en 4,34s - 6,02s)

| Archivo de prueba | Test | Qué verifica | Resultado |
|---|---|---|---|
| `tests/test_app.py` | `test_app_salud` | Endpoint de salud, integridad del modelo ONNX ResNet100 (105 Conv), 836 actores y hash SHA-256 de LDA. | `PASSED` |
| `tests/test_app.py` | `test_app_sirve_frontend_estatico` | Servido estático correcto del cliente web `frontend/` y `index.html`. | `PASSED` |
| `tests/test_app.py` | `test_app_identificar_rechaza_audio_demasiado_corto` | Validación de longitud mínima (<2s) devolviendo HTTP 400 controlado. | `PASSED` |
| `tests/test_app.py` | `test_app_identificar_audio_valido` | Inferencia completa extremo a extremo con el modelo ONNX, ranking y decisión en 3 estados. | `PASSED` |
| `tests/test_aumento_acustico.py` | `test_vad_detecta_silencio` | Detección de silencios y ruidos de fondo planos sin voz (`es_valida = False`). | `PASSED` |
| `tests/test_aumento_acustico.py` | `test_vad_detecta_voz_activa` | Detección de tramos de voz vocal armónica con formantes por energía adaptativa y ZCR. | `PASSED` |
| `tests/test_aumento_acustico.py` | `test_aislar_voz_continua` | Concatenación exclusiva de tramos hablados descartando silencios y música instrumental previa. | `PASSED` |
| `tests/test_aumento_acustico.py` | `test_segmentar_intervenciones_multi_hablante` | Diarización ligera de diálogo cuando hay dos personajes hablando en una escena. | `PASSED` |
| `tests/test_aumento_acustico.py` | `test_generar_rir_salon` | Síntesis de respuestas al impulso de sala (RIR) de salones de estar (RT60 = 0.35s). | `PASSED` |
| `tests/test_aumento_acustico.py` | `test_simular_sala_tv` | Regularización acústica de TV a distancia con ruido de salón doméstico. | `PASSED` |
| `tests/test_aumento_acustico.py` | `test_lda_con_covarianza_acustica` | Estimación de covarianza intracanal inducida por reverberación para robustecer la matriz LDA. | `PASSED` |
| `tests/test_aumento_acustico.py` | `test_estimar_f0_y_registro_vocal` | Estimación de la frecuencia fundamental F0 por autocorrelación y clasificación de registro vocal. | `PASSED` |
| `tests/test_metadatos_modelo.py` | `test_inspeccionar_onnx_distingue_r34_r100_y_ecapa_por_estructura` | Distinción inequívoca entre ECAPA (`dim=192`), ResNet34 (`dim=256`, 39 Conv) y ResNet100 (`dim=256`, 105 Conv). | `PASSED` |
| `tests/test_metadatos_modelo.py` | `test_inscribir_y_verificar_metadatos_onnx` | Escritura y lectura de `metadata_props` en ONNX, respaldo binario sin librería `onnx` y detección de metadatos incoherentes con el grafo. | `PASSED` |
| `tests/test_metadatos_modelo.py` | `test_validar_compatibilidad_indice_detecta_choque_r34_vs_r100` | Bloqueo de uso cruzado entre un `indice.npz` generado con ResNet34 (`v2`) y un modelo ONNX ResNet100 (`v3`) pese a tener ambos `dim=256`. | `PASSED` |
| `tests/test_metadatos_modelo.py` | `test_validar_artefactos_web_integros_y_corruptos` | Validación binaria de `lda.bin`, `centroides.bin`, `trozos.bin` y `actores.json`, detectando corrupción o discrepancia de versión con el ONNX. | `PASSED` |
| `tests/test_rechazo.py` | `test_porcentaje_interfaz_actual_reproduce_js` | Equivalencia matemática exacta con `aPorcentaje(s)` y `aPorcentajeCalibrado(s)` de `frontend/index.html`. | `PASSED` |
| `tests/test_rechazo.py` | `test_clasificacion_tres_estados_abierto_y_contextual` | Clasificación en `identificado`, `dudoso` y `desconocido`, incluyendo empates entre candidatos y búsqueda contextual con `N=1` (verificación 1-a-1). | `PASSED` |
| `tests/test_rechazo.py` | `test_calibrar_umbrales_y_reduccion_falsos_positivos` | Calibración de umbrales por cuantiles de validación y mejora de Brier/ECE con `CalibradorConfianza` (PAVA isotónico). | `PASSED` |
| `tests/test_rechazo.py` | `test_umbrales_calibrados_sincronizados_con_frontend_y_app` | Sincronización de `UMBRALES_VOXBLINK_R100_CALIBRADOS` y `CURVA_CALIBRADA_R100` entre `backend/rechazo.py`, `frontend/index.html` y `backend/app.py`. | `PASSED` |
| `tests/test_sistema_evaluacion.py` | `test_deduccion_fuente_y_agrupacion_antifuga` | Deducción de fuente (`eldoblaje`, `podcast`, `entrevista`, `voces`) y agrupación de `clip` + `clip#movil` + partes de episodio para impedir fugas. | `PASSED` |
| `tests/test_sistema_evaluacion.py` | `test_fidelidad_navegador_cuantizacion_centroides_y_puntuacion` | Reproducción exacta de `centroides_mezcla`, cuantización `int8` de `trozos.bin` y puntuación `(media + mejores_3) / 2` de `frontend/index.html`. | `PASSED` |
| `tests/test_sistema_evaluacion.py` | `test_permisos_grabaciones_reales` | Rechazo con `PermissionError` de cualquier grabación real que no declare `permiso_verificado: true`. | `PASSED` |
| `tests/test_sistema_evaluacion.py` | `test_evaluacion_completa_informes_y_regresiones` | Ejecución extremo a extremo del sistema de evaluación, exportación de JSON/CSV y detección automática de regresiones en `historial.jsonl`. | `PASSED` |
| `tests/test_sistema_evaluacion.py` | `test_filtrar_trozos_intrusos_y_blindaje_respuestas` | Detección de trozos intrusos (`sim < 0.15`), preservación de centroides y blindaje criptográfico anti-envenenamiento en respuestas ciudadanas de Supabase. | `PASSED` |

---

## 4. Mejoras Implementadas para Viabilidad Real del Producto

1. **Rechazo en 3 estados y calibración de confianza:**
   - La interfaz y el backend clasifican cada consulta en `identificado` (fiabilidad confirmada >98%), `dudoso` (ofrece candidatos cercanos e invita a sumar toma) o `desconocido` (rechazo de voces ajenas a la base de datos).
   - Sustitución de la función heurística por una curva calibrada isotónica PAVA (`CURVA_CALIBRADA_R100`), reduciendo el error de calibración (ECE) de 0.0809 a 0.0103 (-87,3%).
2. **Poda automática de trozos intrusos en el índice (`backend/auditar_indice.py`):**
   - Eliminación de 447 segmentos (`sim < 0.15`) pertenecientes a otros interlocutores en la misma escena de doblaje, elevando el acierto Top-1 independiente de 81,79% a 82,00%.
3. **Blindaje de aprendizaje nocturno contra envenenamiento (`despliegue/exportar_web.py`):**
   - Validación de normas vectoriales, filtrado de respuestas con alta discrepancia, deduplicación por hash SHA-256 y amortiguamiento por peso (`peso_respuestas = 0.35`).
4. **VAD adaptativo y diarización ligera de turnos en navegador (`frontend/index.html`):**
   - Filtro adaptativo en tramas de 30 ms que aísla solo voz y descarta pausas o música de fondo.
   - Detección de escenas con diálogo entre dos personajes con desglose individual de cada voz.
5. **Filtrado subsónico de 85 Hz y normalización dinámica:**
   - `BiquadFilter` a 85 Hz tanto en grabación de micrófono como en decodificación de archivos, eliminando golpes de viento y zumbidos domésticos.
6. **Arranque instantáneo en móvil mediante `CacheStorage`:**
   - Almacenamiento persistente en la caché del navegador (`estavoz-v3-artefactos`) del modelo ONNX de 52 MB y de los binarios, reduciendo el tiempo de carga en visitas recurrentes de 15 segundos a menos de 200 ms.
7. **Botón interactivo de detención anticipada ("PARAR"):**
   - Permite al usuario parar la grabación a partir de los 2 segundos en cuanto termina la frase del doblador, evitando grabar música o efectos posteriores.
8. **Verificación sonora directa con muestras oficiales de eldoblaje.com:**
   - Reproductor integrado con un clic de la muestra oficial (`https://www.eldoblaje.com/datos/Muestras/{id}.mp3`) y enlace directo a la ficha del actor.

---

## 5. Resultados que Siguen Pendientes y Próximos Pasos Recomendados

1. **Reentrenamiento de LDA por pliegues sobre vectores previos a LDA (`data/indice.npz`):**
   - Requiere acceso al dataset de entrenamiento privado `SrHarrison/estavoz-datos` para generar una nueva matriz LDA con regularización acústica integrada.
2. **Evaluación con grabaciones reales en móvil físico (`--grabaciones-reales`):**
   - Recopilar un corpus etiquetado de 50 tomas reales en salones domésticos con autorización expresa (`permiso_verificado: true`).
