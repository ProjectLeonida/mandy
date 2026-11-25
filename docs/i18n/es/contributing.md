# Leonida — Cómo contribuir (`es`)

> [!NOTE]
> La versión en inglés de [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) es **la referencia oficial**. Esta traducción
> está sincronizada con el commit `2b567edc`. Si hay alguna diferencia, siga la versión en inglés.

Gracias por ayudar. Unas pocas reglas permiten que el proyecto se pueda seguir alojando:

- **Nunca datos del juego.** Eso incluye los archivos empaquetados, el ejecutable, los archivos extraídos, los shaders
  (DXBC, FSSM, desensamblado, SPIR-V), las texturas, los modelos, el audio, los vídeos y los informes o registros
  que los citen. El `.gitignore` bloquea los tipos más comunes; aun así, revise `git status` antes de hacer commit.
- **Nada de claves, descifrado ni formas de eludir protecciones**, ni enlaces a dónde conseguirlos. Las herramientas
  solo leen archivos que ya están en claro en el disco.
- Se admiten en la documentación y en los comentarios extractos cortos que hagan falta para explicar un formato
  (unos pocos bytes de cabecera, una secuencia de instrucciones).

## Antes de empezar

Primero, reserve la tarea en [la issue de coordinación][claims]: la issue abierta titulada «Coordinating who works on
what». Cuando el hilo se alarga, pasa a una issue nueva; por eso el enlace lanza una búsqueda. Revise sus tablas y
las pull requests abiertas, publique allí `claiming: <ID>` (o una descripción de una línea si la tarea aún no está en
la lista) y ponga el identificador en su pull request. Cuando la pull request esté abierta, publique allí
`done: <ID> -> #<PR>`.

¿Ha encontrado algo que hay que hacer pero no va a hacerlo usted? Publique allí `finding: <una línea>`, indicando
dónde está y por qué importa, y pasará a la lista para que otra persona lo haga. Si aun así dos pull requests se
solapan, se fusiona la que llega primero y a la otra se le pide que haga rebase o que se cierre. Las pull requests
van contra `main`.

El repositorio original (`ItsNotRexed/Leonida`) pasó a ser privado el 23 de noviembre de 2026, y sus issues y pull
requests se fueron con él. Un `#N` escrito antes de esa fecha en la documentación, en un comentario del código o en
un mensaje de commit es un número **antiguo** y ya no lleva a ninguna parte, incluidos los dos a los que esta página
remitía antes: #166 (resultados de los tests de GPU) y #313 (la página de la capa D3D12 → Vulkan).

- Los títulos de las issues que estaban abiertas allí están recogidos en
  [#18](https://github.com/ubiquitarianism/leonida/issues/18). Para retomar una, vuelva a abrirla aquí con un enlace
  a ese índice.
- Las pull requests que estaban abiertas allí se conservan como ramas `old-pr/<número antiguo>`, listadas en
  [#17](https://github.com/ubiquitarianism/leonida/issues/17). Para retomar una, haga rebase sobre `main`, conserve
  los commits de su autor y abra aquí una pull request.
- Un resultado de test de GPU es una pull request que añade un archivo a
  [`gpu-test-results/`](../../../gpu-test-results) con el formato de los archivos que ya hay;
  [docs/testing.md](../../testing.md) explica cómo ejecutar los tests.
- El plan y el estado de la capa D3D12 → Vulkan están en [docs/d3d12_vulkan.md](../../d3d12_vulkan.md).

[claims]: https://github.com/ubiquitarianism/leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22

## Trabajar en el código

- Las herramientas de Python se ejecutan con `python -I` y solo usan la biblioteca estándar, salvo que el archivo
  diga lo contrario.
- Un archivo de texto que escribe una herramienta (JSON, CSV, un manifiesto, un listado de ensamblador) indica su
  fin de línea: `open(path, "w", encoding="utf-8", newline="\n")` (lo mismo para `Path.open`, `Path.write_text` y
  un `tempfile.NamedTemporaryFile` en modo texto). Así la salida es LF también en Windows, y la misma entrada da los
  mismos bytes en todas las plataformas. `tests/test_text_write_line_endings.py` falla si una herramienta lo olvida.
- Un generador de informes rechaza una salida que coincida con una de sus entradas o con otra de sus salidas, y para
  saberlo llama a `tools/report_paths.same_report_path(a, b)` en lugar de volver a calcularlo: la función resuelve
  las dos rutas, aplica las reglas de mayúsculas y minúsculas del volumen y, si hace falta, recurre a
  `os.path.samefile`. Así una herramienta nueva reutiliza esa respuesta en vez de tener su propia copia.
  `tests/test_report_paths.py` lo cubre.
- `tools/rpf9dump` necesita el submódulo: `git submodule update --init`.
- Una regla nueva del traductor de shaders va en la tabla de contrato de `docs/design.md`, junto con cómo se
  confirmó: un test de GPU, una referencia de Mesa/LLVM o un recuento sobre el corpus.
- `python -I tests/test_<nombre>.py` ejecuta uno de los tests unitarios (solo biblioteca estándar; la CI los ejecuta
  todos). Los archivos de test sin bloque `__main__` o con un `import pytest` (necesitan los paquetes de
  `tests/requirements.txt`) se ejecutan con `python -m pytest tests/test_<nombre>.py`, como explica
  [AGENTS.md](../../../AGENTS.md); con `python -I` fallan al importar o no ejecutan nada.
- Tests del runtime: `cmake -S runtime -B runtime/build`, `cmake --build runtime/build --config RelWithDebInfo` y
  luego `ctest --test-dir runtime/build -C RelWithDebInfo --output-on-failure`. En Windows se configuran los tests de
  CPU que no necesitan datos del juego; Linux y macOS configuran el subconjunto portable (`gta6rt` y los shims solo
  existen para Windows). Los tests que necesitan el paquete comercial lo leen en su sitio a través de
  `LEONIDA_GAME_DIR` (y `LEONIDA_MOVIE` para el test de decodificación de Bink) y muestran `SKIP` si la variable no
  está definida; no copie ni haga commit nunca de esos archivos.
- Tests del streaming del visor: `node --test tests/test_viewer_streaming.cjs` (Node.js 22+, la CI prueba la 24
  LTS; sin paquetes, navegador, red ni datos del juego). Ejecutan el código de streaming del visor con un cargador y
  un reloj sintéticos.
- `loader_test` escribe pequeñas imágenes PE en una carpeta `leonida_loader_test` dentro de `%TEMP%`. Si un antivirus
  bloquea los ejecutables recién escritos, el test se detiene indicando ese motivo; excluya la carpeta o haga que
  `LEONIDA_TEST_DIR` apunte a una que el antivirus no revise.
- Comprobación de las herramientas: `python -m compileall -q make_game.py tools tests`. La herramienta opcional
  `vk_pipeline_check <spirv dir> [cache.bin] [--jobs N]` crea un módulo (y, en los shaders compute, un pipeline)
  para cada shader traducido, e informa de los fallos y de la ruta de la caché guardada.
- Los tests de GPU son pequeños programas RDNA2 escritos a mano, en `runtime/test/shaders`. `build.py` los convierte
  en SPIR-V con `llvm-mc` y el target AMDGPU (LLVM 21+). `check.py` los ejecuta en su GPU y compara los resultados.
  Un test nuevo lleva sus valores esperados en el comentario del principio y una comprobación en `check.py`.
  [docs/testing.md](../../testing.md) explica la configuración para cada sistema operativo, incluidas las opciones
  nativas de Windows, y los problemas conocidos.
- Indique en la pull request qué ha probado y en qué GPU.
- Ejecute los tests del traductor con `python -I scripts/run_semantic_tests.py tests/semantic` (consulte
  `tests/semantic/README.md`). Una regla nueva del traductor debe venir con una pequeña fixture `.asm` escrita a mano
  y las palabras de salida esperadas.

## Pull requests

- Un solo tema por pull request, basada en el `main` actual. El trabajo posterior va en una pull request nueva, no
  dentro de una que ya está abierta.
- Rellene la [plantilla](../../../.github/pull_request_template.md): la reserva, `Depends on:`, `Fixes: #N` si cierra
  una issue, qué ha probado, `GPUs tested:` (fabricante, modelo, driver y sistema operativo) y si se ha hecho con
  ayuda de IA.
- Una pull request que entra en conflicto con `main` recibe la etiqueta `needs rebase` (`tools/pr_conflict_label.py`,
  que se ejecuta en cada push a `main` y cada hora). Haga usted el rebase sobre `main`; la etiqueta desaparece en
  menos de una hora cuando ya se puede fusionar sin conflictos. Las revisiones empiezan por las pull requests que no
  tienen conflictos.
