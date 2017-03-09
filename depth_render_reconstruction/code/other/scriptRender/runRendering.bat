mkdir ..\renderDepth
mkdir .\finishedModel
for %%i in (*.obj) do ( generateDepth %%i camPosList.txt %%~ni.run %%~ni
                        runRendering %%~ni.run 
                        move %%i .\finishedModel\%%i
                        move %%~ni.run .\finishedModel\%%~ni.run )