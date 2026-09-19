@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

title "Crossing Hurdles: Talent Retention Survival Engine"

echo ======================================================================
echo  Crossing Hurdles: Talent Retention Survival Engine (GP-023)
echo  Automated Pipeline, In-Memory OLAP and Test Suite (1-Click Run)
echo ======================================================================
echo.

rem Resolucion de interprete Python
set "PY_CMD=python"
%PY_CMD% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    set "PY_CMD=py -3"
    %PY_CMD% --version >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] No se encontro Python en el sistema.
        exit /b 1
    )
)

echo [1/8] Generando dataset sintetico calibrado (50,000 registros)...
%PY_CMD% src\data_generator.py --records 50000 --output data\raw_dataset.parquet
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el generador de datos.
    exit /b %ERRORLEVEL%
)

echo.
echo [2/8] Ejecutando Motor Analitico Core (Kaplan-Meier, Tablas Actuariales y Hazard Ratios)...
%PY_CMD% src\core_engine.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el motor analitico.
    exit /b %ERRORLEVEL%
)

echo.
echo [3/8] Entrenando Motor de Machine Learning de Supervivencia y Calibracion IPCW...
%PY_CMD% src\survival_ml.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el motor de Machine Learning de supervivencia.
    exit /b %ERRORLEVEL%
)

echo.
echo [4/8] Ejecutando Telemetria Longitudinal y Landmark Analysis en Semanas Criticas (3, 5, 7)...
%PY_CMD% src\longitudinal_engine.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el motor de telemetria longitudinal y landmark analysis.
    exit /b %ERRORLEVEL%
)

echo.
echo [5/8] Ejecutando Motor Causal de Uplift y Optimizador Knapsack de Presupuesto de Mentoria...
%PY_CMD% src\causal_uplift_engine.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el motor causal de uplift y optimizador knapsack.
    exit /b %ERRORLEVEL%
)

echo.
echo [6/8] Ejecutando Motor de Incertidumbre Conformal y Cotas de Supervivencia al 90%%...
%PY_CMD% src\conformal_engine.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el motor de incertidumbre conformal.
    exit /b %ERRORLEVEL%
)

echo.
echo [7/8] Ejecutando Suite Automatizada de Pruebas Unitarias (Pytest: 27 tests)...
%PY_CMD% -m pytest tests\ -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pruebas unitarias fallidas.
    exit /b %ERRORLEVEL%
)

echo.
echo [8/8] Ejecutando Benchmarks Cuantitativos de Latencia y Memoria (50 iteraciones)...
%PY_CMD% tests\benchmark.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el benchmark de latencia.
    exit /b %ERRORLEVEL%
)

echo ======================================================================
echo  [OK] Ejecucion Exitosa: Pipeline, Survival ML, Landmarks, Uplift, Conformal y 27 Tests al 100%%
echo  [WEB] Para lanzar el Simulador Web Interactivo:
echo        %PY_CMD% src\web_dashboard.py --timeout=0
echo        o abrir directamente web\index.html en el navegador.
echo ======================================================================
endlocal
