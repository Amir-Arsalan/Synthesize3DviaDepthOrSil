# Synthesizing 3D Shapes via Modeling Multi-View Depth Maps and Silhouettes with Deep Generative Networks


This Readme file is to be completely modified.

How to process the data and start training:
* Your raw data should be in .zip files
* Create one .zip file for each object category (e.g. chair.zip, sofa.zip etc)
* Copy your .zip files to /Data/[benchmark/nonbenchmark]
* Each zip file MUST contain either (or both) of the following directories:
* 	- catName_depth_float
*	- catName_depth_rgb
* "catName" is the class label. E.g. "sofa", "chair, night_stand" etc
* Each directory contains files with patterns as follow: (%a+_%d+_%a+_%d*.[png/txt]). E.g. model_000043_Cam_0.[png/txt]. In our experiments we have 20 view points so the file names for each 3D model have Cam_0 to Cam_19 in their names.
* The option opt.rawDataType determines whether to process the .txt (float) or png (int) files. Set opt.rawDataType to 'float' and 'int' to read .txt or .png files respectively


Before proceeding, create a directory named "Data" in the cloned repository's directory
* How to process nonbenchmark data:
* If you are working with your own data set then create the directory /Data/nonbenchmark
* Copy your .zip files in "Data/nonbenchmark"
* Set opt.benchmark = 0

How to process benchmark data (ModelNet40):
* If you are working with some benchmark data set then create the directory /Data/benchmark
* Create two directories "/Data/benchmark/train" and "/Data/benchmark/test"
* Copy your train and test .zip files in the corresponding directories
* Set opt.benchmark = 1

Note: All zip files must have the format described above


How to run the code:
If the purpose is to process the ModelNet40 data from scratch run:

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 1 -fromScratch 1 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 0 -onlySilhouettes 0  -singleVPNet 1 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 1 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD70-350Z-BS4-Conditional"

After running the above command the program uncompresses the .zip files and stores the images into Torch tensors for training and validation phase. Then it starts the training and stores the intermediate results, if any, in /expDirName.
For the next runs use -zip 0 -fromScratch 0 arguments.


# Note: Only conditional training produces classification accuracy results

## Unconditional training:
- AllVPNet:

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 0 -VpToKeep 30 -onlySilhouettes 0  -singleVPNet 0 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 0 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-AllVPNet-Depth"

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 0 -onlySilhouettes 1  -singleVPNet 0 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 0 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-AllVPNet-Silhouette"

- DropoutNet:

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 1 -onlySilhouettes 0  -singleVPNet 0 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 0 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-DropoutNet-Depth"

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 1 -onlySilhouettes 1  -singleVPNet 0 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 0 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-DropoutNet-Silhouette"

- SingleVPNet:

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 1 -onlySilhouettes 0  -singleVPNet 1 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 0 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-SingleVPNet-Depth"

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 1 -onlySilhouettes 1  -singleVPNet 1 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 0 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-SingleVPNet-Silhouette"


## Conditional training:
- AllVPNet:

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 0 -VpToKeep 30 -onlySilhouettes 0  -singleVPNet 0 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 1 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-AllVPNet-Depth-Conditional"

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 0 -onlySilhouettes 1  -singleVPNet 0 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 1 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-AllVPNet-Silhouette-Conditional"

- DropoutNet:

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 1 -onlySilhouettes 0  -singleVPNet 0 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 1 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-DropoutNet-Depth-Conditional"

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 1 -onlySilhouettes 1  -singleVPNet 0 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 1 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-DropoutNet-Silhouette-Conditional"

- SingleVPNet:

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 1 -onlySilhouettes 0  -singleVPNet 1 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 1 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-SingleVPNet-Depth-Conditional"

* th main.lua -testPhase 0 -batchSize 4 -nLatents 400 -rawDataType 'int' -zip 0 -fromScratch 0 -nCh 64 -maxEpochs 50 -sampleType 'interpolate' -nSamples 6 -lr 0.000085 -initialLR 0.000002 -lrDecay 0.98 -nReconstructions 30 -sampleOnly 0 -batchSizeChangeEpoch 20 -batchSizeChange 2 -targetBatchSize 8 -modelPath "" -var 1 -mean 0 -canvasHW 4 -tanh 0 -dropoutNet 1 -onlySilhouettes 1  -singleVPNet 1 -maxMemory 4000 -KLD 80 -benchmark 1 -conditional 1 -expDirName "ResNet-64Chs-lr0.000085-L1L1-KLD80-350Z-BS4-SingleVPNet-Silhouette-Conditional"
