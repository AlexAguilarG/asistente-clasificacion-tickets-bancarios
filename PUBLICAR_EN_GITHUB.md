# Publicar el repositorio en GitHub

## Antes de publicar

1. Revisa `README.md`, `LICENSE` y `data/README.md`. La licencia del código no determina la licencia de los datos ni la de los modelos.
2. El dataset completo no se incluye. Los datos se descargan desde Hugging Face cuando corren los scripts.
3. No agregues la carpeta `results/run_latest/` sin inspeccionar sus archivos: las predicciones locales pueden incluir textos del conjunto de prueba.
4. No subas tokens, secretos, ni tickets reales.

## Opción A: desde la interfaz web

1. En GitHub, crea un repositorio nuevo con el nombre `asistente-clasificacion-tickets-bancarios`.
2. Selecciona **Private** hasta confirmar los términos de la variante traducida y revisar qué resultados deseas redistribuir. Puede cambiarse a público posteriormente.
3. No inicialices el repositorio con README, licencia ni `.gitignore`, porque ya están incluidos.
4. Descomprime `Repositorio_Asistente_Tickets_Bancarios.zip`, abre la carpeta `repositorio_asistente_tickets_bancarios` y carga sus archivos a la raíz del repositorio.

## Opción B: Git desde terminal

Desde la raíz `repositorio_asistente_tickets_bancarios`, tras crear el repositorio vacío en GitHub, ejecuta:

```bash
git init
git add .
git status
# Revisa que no aparezcan datos locales o archivos sensibles.
git commit -m "Estructura inicial del proyecto de clasificación de tickets"
git branch -M main
git remote add origin https://github.com/<USUARIO>/<REPOSITORIO>.git
git push -u origin main
```

Sustituye `<USUARIO>` y `<REPOSITORIO>` por los valores que correspondan. Si GitHub requiere autenticación, sigue su método oficial; no guardes contraseñas o tokens en los scripts.
