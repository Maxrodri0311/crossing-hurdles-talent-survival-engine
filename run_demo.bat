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

echo [1/4] Generando dataset sintetico calibrado (50,000 registros)...
%PY_CMD% src\data_generator.py --records 50000 --output data\raw_dataset.parquet
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el generador de datos.
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Ejecutando Motor Analitico Core (Kaplan-Meier, Tablas Actuariales y Hazard Ratios)...
%PY_CMD% src\core_engine.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el motor analitico.
    exit /b %ERRORLEVEL%
)

echo.
echo [3/5] Entrenando Motor de Machine Learning de Supervivencia y Calibracion IPCW...
%PY_CMD% src\survival_ml.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el motor de Machine Learning de supervivencia.
    exit /b %ERRORLEVEL%
)

echo.
echo [4/5] Ejecutando Suite Automatizada de Pruebas Unitarias (Pytest: 11 tests)...
%PY_CMD% -m pytest tests\ -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pruebas unitarias fallidas.
    exit /b %ERRORLEVEL%
)

echo.
echo [5/5] Ejecutando Benchmarks Cuantitativos de Latencia y Memoria (50 iteraciones)...
%PY_CMD% tests\benchmark.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo en el benchmark de latencia.
    exit /b %ERRORLEVEL%
)

echo ======================================================================
echo  [OK] Ejecucion Exitosa: Pipeline, Survival ML y Pruebas al 100%%
echo ======================================================================
endlocal
