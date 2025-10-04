@echo off
REM Bridge VIV Risk Assessment - Experiment Runner Script (Windows)

setlocal enabledelayedexpansion

REM Configuration
if "%CONFIG_FILE%"=="" set CONFIG_FILE=config\config.yaml
if "%EXPERIMENT_NAME%"=="" (
    for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set mydate=%%c%%a%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
    set EXPERIMENT_NAME=bridge_viv_!mydate!_!mytime!
)
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=experiments
if "%PYTHON_CMD%"=="" set PYTHON_CMD=python

echo [INFO] Starting Bridge VIV Risk Assessment Experiments
echo [INFO] Experiment Name: %EXPERIMENT_NAME%
echo [INFO] Configuration: %CONFIG_FILE%
echo [INFO] Output Directory: %OUTPUT_DIR%

REM Check dependencies
echo [INFO] Checking dependencies...
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ or set PYTHON_CMD environment variable.
    exit /b 1
)

%PYTHON_CMD% -c "import numpy, pandas, sklearn" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required Python packages not found. Please install requirements: pip install -r requirements.txt
    exit /b 1
)

echo [SUCCESS] Dependencies check passed

REM Setup directories
echo [INFO] Setting up directories...
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "results" mkdir "results"
if not exist "logs" mkdir "logs"
echo [SUCCESS] Directories created

REM Validate data
echo [INFO] Validating data...
if not exist "%CONFIG_FILE%" (
    echo [ERROR] Configuration file not found: %CONFIG_FILE%
    exit /b 1
)

REM Run experiments based on argument
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=full

if "%COMMAND%"=="baseline" goto run_baseline
if "%COMMAND%"=="hyperopt" goto run_hyperopt
if "%COMMAND%"=="predict" goto run_predict
if "%COMMAND%"=="test" goto run_test
if "%COMMAND%"=="full" goto run_full
goto unknown_command

:run_baseline
echo [INFO] Running baseline experiments...
%PYTHON_CMD% src\train.py --config "%CONFIG_FILE%" --experiment-name "%EXPERIMENT_NAME%_baseline" --models linear random_forest --output-dir "%OUTPUT_DIR%" > "logs\%EXPERIMENT_NAME%_baseline.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Baseline experiments failed
    exit /b 1
)
echo [SUCCESS] Baseline experiments completed
goto end

:run_hyperopt
echo [INFO] Running hyperparameter optimization...
%PYTHON_CMD% src\hyperparam_search.py --config "%CONFIG_FILE%" --experiment-name "%EXPERIMENT_NAME%_hyperopt" --n-trials 50 --output-dir "%OUTPUT_DIR%" > "logs\%EXPERIMENT_NAME%_hyperopt.log" 2>&1
if errorlevel 1 (
    echo [WARNING] Hyperparameter optimization failed, continuing with baseline results
) else (
    echo [SUCCESS] Hyperparameter optimization completed
)
goto end

:run_predict
echo [INFO] Generating predictions...
REM Find best model (simplified for Windows)
if exist "%OUTPUT_DIR%\complete_experiment_results.json" (
    %PYTHON_CMD% src\predict.py --comprehensive --input bridge_dataset_fixed.csv --output "results\predictions_%EXPERIMENT_NAME%.csv" > "logs\%EXPERIMENT_NAME%_predict.log" 2>&1
    if errorlevel 1 (
        echo [WARNING] Prediction generation failed
    ) else (
        echo [SUCCESS] Predictions generated: results\predictions_%EXPERIMENT_NAME%.csv
    )
) else (
    echo [WARNING] Experiment results not found, skipping prediction generation
)
goto end

:run_test
echo [INFO] Running tests...
pytest tests\ -v --tb=short > "logs\%EXPERIMENT_NAME%_tests.log" 2>&1
if errorlevel 1 (
    echo [WARNING] Some tests failed, check logs for details
) else (
    echo [SUCCESS] All tests passed
)
goto end

:run_full
echo [INFO] Running complete experiment pipeline...

REM Baseline
call :run_baseline
if errorlevel 1 exit /b 1

REM Hyperopt
call :run_hyperopt

REM Complete experiments
echo [INFO] Running complete experiment pipeline...
%PYTHON_CMD% src\experiments.py --config "%CONFIG_FILE%" --experiment-name "%EXPERIMENT_NAME%" --output-dir "%OUTPUT_DIR%" --full-pipeline > "logs\%EXPERIMENT_NAME%_complete.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Complete experiments failed
    exit /b 1
)
echo [SUCCESS] Complete experiments finished

REM Predictions
call :run_predict

REM Tests
call :run_test

REM Generate report
echo [INFO] Generating experiment report...
set REPORT_FILE=results\experiment_report_%EXPERIMENT_NAME%.md

echo # Bridge VIV Risk Assessment - Experiment Report > "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"
echo **Experiment Name:** %EXPERIMENT_NAME% >> "%REPORT_FILE%"
echo **Date:** %DATE% %TIME% >> "%REPORT_FILE%"
echo **Configuration:** %CONFIG_FILE% >> "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"
echo ## Experiment Summary >> "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"
echo This report summarizes the results of the Bridge VIV risk assessment experiment. >> "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"
echo ### Files Generated >> "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"
echo - **Models:** %OUTPUT_DIR%\ >> "%REPORT_FILE%"
echo - **Predictions:** results\predictions_%EXPERIMENT_NAME%.csv >> "%REPORT_FILE%"
echo - **Logs:** logs\%EXPERIMENT_NAME%_*.log >> "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"
echo ### Usage >> "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"
echo To use the trained models for prediction: >> "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"
echo ```bash >> "%REPORT_FILE%"
echo python src\predict.py --model %OUTPUT_DIR%\best_model.pkl --input your_data.csv --output predictions.csv --comprehensive >> "%REPORT_FILE%"
echo ``` >> "%REPORT_FILE%"
echo. >> "%REPORT_FILE%"
echo --- >> "%REPORT_FILE%"
echo *Generated by Bridge VIV Risk Assessment System* >> "%REPORT_FILE%"

echo [SUCCESS] Experiment report generated: %REPORT_FILE%
goto end

:unknown_command
echo [ERROR] Unknown command: %COMMAND%
echo Usage: %0 [baseline^|hyperopt^|predict^|test^|full]
exit /b 1

:end
echo [SUCCESS] Experiment pipeline completed successfully!
endlocal