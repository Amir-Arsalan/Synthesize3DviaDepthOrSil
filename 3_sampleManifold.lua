#!~/torch/install/bin/th

require 'torch'
require 'image'
require 'paths'
local commonFuncs = require '0_commonFuncs'



local sampleManifold = {}

function sampleManifold.sample(sampleType, sampleCategory, canvasHW, nSamples, data, model, modelPath, samplesPath, mean, var, nLatents, imgSize, numVPs, epoch, batchSize, targetBatchSize, sampleOnly, testPhase, tanh, dropoutNet, VpToKeep, onlySilhouettes, sampleZembeddings, singleVPNet, conditional, expType, benchmark)
    local conditionalAvailable = conditional and 1 or 0
    if sampleOnly and modelPath ~= '' then
        require 'nngraph'
        require '2_2_Sampler'

        -- Load a model
        gMod = torch.load(modelPath)
        modelTrain = gMod:get(1)
        model = gMod:get(2)
        modelTrain:evaluate()

        -- Load the data
        if sampleType == 'data' or sampleType == 'interpolation' and not expType then
            _, dataFiles = commonFuncs.obtainDataPath(benchmark, testPhase, testPhase)
            dataFiles = commonFuncs.randPermTableContents(dataFiles)
            data = torch.load(dataFiles[1])
        elseif not expType then
            _, dataFiles = commonFuncs.obtainDataPath(benchmark, false, true)
            data = torch.load(dataFiles[1])
        end
        numVPs = data.dataset:size(2)
        imgSize = data.dataset:size(3)

        if sampleOnly then
            gMod = gMod:cuda()
        end
    end

    local offlineExpDirName
    if sampleType == 'random' or sampleType == 'data' or sampleType == 'interpolate' or sampleType == 'datainterpolate' then
        local savePathRandom, savePathDataInterpolate
        -- if sampleType == 'random' then
            -- savePathRandom = string.format('%s/%s/Mean_%.3f-Var_%.2f/', samplesPath, 'random', mean, var)
        -- elseif sampleType == 'data' or sampleType == 'interpolate' or sampleType == 'datainterpolate' then
            offlineExpDirName = opt.expType and 'interpolation' .. (commonFuncs.numOfDirs(samplesPath)+1 >= 1 and  commonFuncs.numOfDirs(samplesPath)+1 or 1) or 'interpolation'
            savePathDataInterpolate = string.format('%s/%s', samplesPath, offlineExpDirName)
        -- end
        -- if sampleType == 'random' then
        if not expType or expType == 'random' then
            offlineExpDirName = opt.expType and 'RandomOrConditional' .. (commonFuncs.numOfDirs(samplesPath)+1 >= 1 and commonFuncs.numOfDirs(samplesPath)+1 or 1) or 'random'
            if not sampleZembeddings then
                paths.mkdir(string.format('%s/%s/Mean_%.3f-Var_%.2f/', samplesPath, offlineExpDirName, mean, var))
            else
                paths.mkdir(string.format('%s/%s/empirical/', samplesPath, offlineExpDirName))
            end
            local canvasSize = (not expType and conditional and canvasHW - 1) or canvasHW
            local meanVec = torch.Tensor(1, nLatents):fill(mean)
            local diagLogVarianceVec = torch.Tensor(1, nLatents):fill(var):log() -- Take the log of the diagonal elements of a covariance matrix
            local canvas = torch.Tensor(numVPs, canvasSize * imgSize, canvasSize * imgSize)
            local silhouetteCanvas = torch.Tensor(numVPs, canvasSize * imgSize, canvasSize * imgSize)
            for c=1, conditional and (data and #data.category) or 1 do
                local allowSampleForCategory = false
                if conditional and type(sampleCategory) == 'table' then
                    for l=1, #sampleCategory do
                        if sampleCategory[l] == data.category[c] then
                            allowSampleForCategory = true
                        end
                    end
                else
                    allowSampleForCategory = true
                end
                if allowSampleForCategory or sampleCategory == '' or not expType then
                    if conditional then
                        if not sampleZembeddings then
                            savePathRandom = string.format('%s/%s/Mean_%.3f-Var_%.2f/%s/', samplesPath, offlineExpDirName, mean, var, data.category[c])
                        else
                            savePathRandom = string.format('%s/%s/empirical/%s/', samplesPath, offlineExpDirName, data.category[c])
                        end
                    else
                        if not sampleZembeddings then
                            savePathRandom = string.format('%s/%s/Mean_%.3f-Var_%.2f/', samplesPath, offlineExpDirName, mean, var)
                        else
                            savePathRandom = string.format('%s/%s/empirical/', samplesPath, offlineExpDirName)
                        end
                    end
                    for j=1, conditional and nSamples-3 or nSamples do
                        local counter = 1
                        local zVectors
                        if not sampleZembeddings then
                            zVectors = commonFuncs.sampleDiagonalMVN(meanVec, diagLogVarianceVec, canvasSize ^ 2)
                        else
                            zVectors = commonFuncs.sampleDiagonalMVN({sampleZembeddings[1]:mean(1):float(), sampleZembeddings[1]:var(1):log():float()}, {sampleZembeddings[2]:mean(1):float(), sampleZembeddings[2]:var(1):log():float()}, canvasSize ^ 2)
                        end
                        for i=1, canvasSize ^ 2 do
                            local z
                            z = zVectors[{{i}}]
                            z = torch.cat(z, z, 1)
                            z = z:cuda()
                            local reconstruction, targetClassLabels
                            if conditional then
                                targetClassLabels = torch.zeros(2, #data.category)
                                for l=1, 2 do
                                    targetClassLabels[l][c] = 1
                                end
                                targetClassLabels = targetClassLabels:type(model:type())
                                reconstruction = model:get(conditionalAvailable+4):forward({z, targetClassLabels})
                            else
                                reconstruction = model:get(4):forward(z)
                            end
                           
                            local silhouettes = reconstruction[2]:clone()
                            reconstruction[2] = nil
                            reconstruction = reconstruction[1]
                            collectgarbage()
                            if tanh then reconstruction = commonFuncs.normalizeBackToZeroToOne(reconstruction) end

                            for k=1, numVPs do
                                canvas[{{k}, {(counter-1) * imgSize + 1, counter * imgSize}, {(i - 1) % canvasSize * imgSize + 1, ((i - 1) % canvasSize + 1) * imgSize}}]:copy(reconstruction[{1, k}]:type(torch.getdefaulttensortype()))
                                silhouetteCanvas[{{k}, {(counter-1) * imgSize + 1, counter * imgSize}, {(i - 1) % canvasSize * imgSize + 1, ((i - 1) % canvasSize + 1) * imgSize}}]:copy(silhouettes[{1, k}]:type(torch.getdefaulttensortype()))
                            end
                            if i % canvasSize == 0 then counter = counter + 1 end
                            z = nil
                            reconstruction = nil
                            silhouettes = nil
                            collectgarbage()
                        end
                        paths.mkdir(string.format('%s/sample%d/', savePathRandom, j))
                        paths.mkdir(string.format('%s/sample%d/mask', savePathRandom, j))
                        for k=1, numVPs do
                            image.save(string.format(savePathRandom .. 'sample%d/VP-%d.png', j, k-1), canvas[{k}])
                            image.save(string.format(savePathRandom .. 'sample%d/mask/VP-%d.png', j, k-1), silhouetteCanvas[{k}])
                        end
                    end
                end
            end -- END for k=1, conditional and #data.category or 1
            canvas = nil
            silhouetteCanvas = nil
            model:clearState()
            collectgarbage()
        end
        -- elseif sampleType == 'data' or sampleType == 'interpolate' or sampleType == 'datainterpolate' then
            if not expType and epoch >= 1 and epoch % 1 == 0 or expType and expType == 'interpolate' then
                offlineExpDirName = opt.expType and 'interpolation' .. (commonFuncs.numOfDirs(samplesPath)+1 >= 1 and  commonFuncs.numOfDirs(samplesPath)+1 or 1) or 'interpolation'
                paths.mkdir(string.format('%s/%s/', samplesPath, offlineExpDirName))
                print ("==> Doing interpolation and sampling around randomly-selected data samples' Z-vectors. Configs - Number of Samples: " .. nSamples - 2 .. ", Canvas Size: " .. canvasHW - 1 .. " X " .. canvasHW - 1)
                nSamples = nSamples - 1 --Just to save computation time
                canvasHW = canvasHW - 1 --Just to save computation time
                for class=1, #data.category do
                    local continueFlag = false
                    if #sampleCategory > 0 then
                        for sampleNo=1, #sampleCategory do
                            if data.category[class] == sampleCategory[sampleNo] then
                                continueFlag = true
                            end
                        end
                    else
                        continueFlag = true
                    end

                    if continueFlag then
                        local numOfVPsToDrop = torch.zeros(1) -- A placeholder (to be filled later)
                        local dropIndices = torch.zeros(numVPs) -- A placeholder (to be filled later)
                        local pickedVPs = torch.Tensor(2) -- A placeholder (to be filled later)
                        if not expType or VpToKeep >= numVPs then
                            pickedVPs[1] = torch.random(1, numVPs)
                            pickedVPs[2] = pickedVPs[1]
                        else
                            pickedVPs[1] = VpToKeep
                            pickedVPs[2] = VpToKeep
                        end

                        local matchingElements = data.labels:eq(torch.Tensor(data.dataset:size(1)):fill(class)) -- Find the samples within one of the classes
                        if matchingElements:sum() > 1 then
                            local tempData = data.dataset:index(1, torch.range(1, data.dataset:size(1))[matchingElements]:long()):clone() -- Extract the samples belonging to the class of interest
                            local batchIndices = torch.randperm(tempData:size(1)):long():split(math.max(math.ceil(batchSize/2), targetBatchSize))

                            -- Correct the last index set size
                            if #batchIndices > 1 then
                                local tempbatchIndices = {}
                                for ll=1, tempData:size(1) - math.max(math.ceil(batchSize/2), targetBatchSize) * (#batchIndices - 1) do
                                    tempbatchIndices[ll] = batchIndices[#batchIndices][ll]
                                end
                                batchIndices[#batchIndices] = torch.LongTensor(tempbatchIndices)
                            end

                            local nTotalSamples = 0
                            local batchesVisited = 0
                            local i = 1
                            while nTotalSamples < nSamples and batchesVisited < #batchIndices do -- Do this for all samples
                                batchesVisited = batchesVisited + 1
                                local passFlag = true
                                if batchIndices[i]:size(1) + nTotalSamples > nSamples then
                                    batchIndices[i] = batchIndices[i][{{1, nSamples - nTotalSamples}}]
                                end

                                if batchIndices[i]:size(1) == 1 then
                                    batchIndices[i] = batchIndices[i]:repeatTensor(2)
                                end

                                if passFlag then
                                    

                                    local depthMaps, droppedInputs 
                                    depthMaps = tempData:index(1, batchIndices[i]):clone():type(model:type())

                                    -- Generate the mask for the current samples
                                    local silhouettes = depthMaps:clone()
                                    if tanh then
                                        silhouettes[silhouettes:gt(-1)] = 1
                                        silhouettes[silhouettes:eq(-1)] = 0
                                    else
                                        silhouettes[silhouettes:gt(0)] = 1
                                    end

                                    -- local permutedInput
                                    local predClassVec
                                    if dropoutNet or singleVPNet then
                                        droppedInputs = commonFuncs.dropInputVPs({depthMaps, silhouettes}, not singleVPNet and VpToKeep or nil, true, numOfVPsToDrop, dropIndices, singleVPNet, pickedVPs)
                                        if opt.conditional then
                                            mean, log_var, predictedClassScores = unpack(model:get(2):forward(not opt.onlySilhouettes and droppedInputs[1] or droppedInputs[2]))
                                            predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, #data.category)
                                            model:get(conditionalAvailable+4):forward({model:get(3):forward({mean, log_var}), predClassVec})
                                        else
                                            model:forward(not opt.onlySilhouettes and droppedInputs[1] or droppedInputs[2])
                                        end
                                    else
                                        droppedInputs = commonFuncs.dropInputVPs(not onlySilhouettes and depthMaps or silhouettes, nil, true, numOfVPsToDrop, dropIndices, singleVPNet, pickedVPs)
                                        if opt.conditional then
                                            mean, log_var, predictedClassScores = unpack(model:get(2):forward(droppedInputs))
                                            predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, #data.category)
                                            model:get(conditionalAvailable+4):forward({model:get(3):forward({mean, log_var}), predClassVec})
                                        else
                                            model:forward(droppedInputs)
                                        end
                                    end
                                    
                                    local dataBeingUsed = depthMaps:clone()
                                    local reconstructions = model:get(conditionalAvailable+4).output
                                    local originalSilhouettes = depthMaps:clone()

                                    local originalSilhouettesReconstructions = reconstructions[2]:clone():type(torch.getdefaulttensortype())
                                    reconstructions[2] = nil
                                    local originalReconstructions = reconstructions[1]:clone():type(torch.getdefaulttensortype())
                                    collectgarbage()
                                    if tanh then originalReconstructions = commonFuncs.normalizeBackToZeroToOne(originalReconstructions) dataBeingUsed = commonFuncs.normalizeBackToZeroToOne(dataBeingUsed) end


                                    local zVecPrevExample
                                    local canvas = torch.Tensor(numVPs, canvasHW * imgSize, canvasHW * imgSize)
                                    local silhouetteCanvas = torch.Tensor(numVPs, canvasHW * imgSize, canvasHW * imgSize)
                                    for l=1, nSamples > 2 and batchIndices[i]:size(1) or 2 do
                                        nTotalSamples = nTotalSamples + 1

                                        meanVec = model:get(2).output[1][{{l}}]:clone():type(torch.getdefaulttensortype())
                                        diagLogVarianceVec = model:get(2).output[2][{{l}}]:clone():type(torch.getdefaulttensortype())
                                        if var > 0 then
                                            diagLogVarianceVec:exp():mul(var):log()
                                        end

                                        -- Sample z vectors
                                        local zVectors
                                        zVectors = torch.zeros(canvasHW ^ 2, nLatents)
                                        zVectors[2]:copy(model:get(3).output[l]:type(torch.getdefaulttensortype()))
                                        zVectors[{{3, canvasHW ^ 2}}]:copy(commonFuncs.sampleDiagonalMVN(meanVec, diagLogVarianceVec, canvasHW ^ 2 - 2)) -- The minus 2 is there since for each depth map we have 1 original depth map and 1 reconstructed version of the same depth map. Therefore, we require 2 less sampled Z vectors

                                        -- Prepare the vectors for doing interpolation
                                        local interpolationCanvas = torch.Tensor(numVPs, canvasHW * imgSize, canvasHW * imgSize)
                                        local interpolationsilhouetteCanvas
                                        interpolationsilhouetteCanvas = torch.Tensor(numVPs, canvasHW * imgSize, canvasHW * imgSize)
                                        local interpolationZVectors = torch.zeros(canvasHW ^ 2, nLatents)
                                        if l >= 2 then
                                            if sampleType ~= 'data' then
                                                interpolationZVectors[{2}]:copy(zVecPrevExample)
                                                interpolationZVectors[{{3, canvasHW ^ 2 - 2}}]:copy(commonFuncs.interpolateZVectors(zVecPrevExample, zVectors[{{2}}], canvasHW ^ 2 - 4)) -- The minus 4 is there since for each depth map we have one original depth map, one reconstructed version of the same depth map before interpolation (both located on top left), one interpolation target reconstructed depth map along with its original depth map (located at the bottom right). Therefore, we require 4 less interpolated versions of Z vectors
                                                interpolationZVectors[{canvasHW ^ 2 - 1}]:copy(zVectors[{2}])
                                            end
                                        
                                        
                                            -- Fill up the canvas(es) by passing the z vectors through the decoder
                                            -- and drawing the result on the canvas for each view point
                                            local counter = 1
                                            for j=2, canvasHW ^ 2 do
                                                local samplesReconstructions, interpolationReconstructions, samplesSilhouettesReconstructions, interpolationSilhouettesReconstructions
                                                local zSamples = zVectors[{{j}}]:repeatTensor(2, 1)
                                                local zInterpolations = interpolationZVectors[{{j}}]:repeatTensor(2, 1)
                                                zSamples = zSamples:cuda() zInterpolations = zInterpolations:cuda()
                                                if sampleType ~= 'interpolate' then
                                                    samplesReconstructions = model:get(conditionalAvailable+4):forward(zSamples)
                                                    samplesSilhouettesReconstructions = samplesReconstructions[2]:clone():type(torch.getdefaulttensortype())
                                                    samplesReconstructions[2] = nil
                                                    samplesReconstructions = samplesReconstructions[1]:clone():type(torch.getdefaulttensortype())
                                                    collectgarbage()
                                                end
                                                if tanh then samplesReconstructions = commonFuncs.normalizeBackToZeroToOne(samplesReconstructions) end


                                                -- Fill the canvas(es)
                                                for k=1, numVPs do


                                                    if sampleType ~= 'interpolate' then
                                                        canvas[{{k}, {1, dataBeingUsed:size(3)}, {1, dataBeingUsed:size(3)}}] = dataBeingUsed[{{l}, {k}}]:type(torch.getdefaulttensortype())
                                                        canvas[{{k}, {(counter-1) * imgSize + 1, counter * imgSize}, {(j - 1) % canvasHW * imgSize + 1, ((j - 1) % canvasHW + 1) * imgSize}}]:copy(samplesReconstructions[{1, k}]:type(torch.getdefaulttensortype()))
                                                        silhouetteCanvas[{{k}, {1, originalSilhouettes:size(3)}, {1, originalSilhouettes:size(3)}}]:copy(originalSilhouettes[{{l}, {k}}])
                                                        silhouetteCanvas[{{k}, {(counter-1) * imgSize + 1, counter * imgSize}, {(j - 1) % canvasHW * imgSize + 1, ((j - 1) % canvasHW + 1) * imgSize}}]:copy(samplesSilhouettesReconstructions[{1, k}]:type(torch.getdefaulttensortype()))
                                                    end

                                                    if sampleType ~= 'data' then
                                                        interpolationReconstructions = model:get(conditionalAvailable+4):forward(conditionalAvailable == 0 and zInterpolations or {zInterpolations, predClassVec})
                                                        interpolationSilhouettesReconstructions = interpolationReconstructions[2]:clone():type(torch.getdefaulttensortype())
                                                        interpolationReconstructions[2] = nil
                                                        interpolationReconstructions = interpolationReconstructions[1]:clone():type(torch.getdefaulttensortype())

                                                        -- Fill the interpolation canvas
                                                        interpolationsilhouetteCanvas[{{k}, {1, originalSilhouettes:size(3)}, {1, originalSilhouettes:size(3)}}]:copy(originalSilhouettes[{{l - 1}, {k}}])
                                                        interpolationsilhouetteCanvas[{{k}, {(counter-1) * imgSize + 1, counter * imgSize}, {(j - 1) % canvasHW * imgSize + 1, ((j - 1) % canvasHW + 1) * imgSize}}]:copy(interpolationSilhouettesReconstructions[{1, k}])
                                                        if tanh then interpolationReconstructions = commonFuncs.normalizeBackToZeroToOne(interpolationReconstructions) end

                                                        interpolationCanvas[{{k}, {1, dataBeingUsed:size(3)}, {1, dataBeingUsed:size(3)}}] = dataBeingUsed[{{l - 1}, {k}}]:type(torch.getdefaulttensortype())
                                                        interpolationCanvas[{{k}, {(counter-1) * imgSize + 1, counter * imgSize}, {(j - 1) % canvasHW * imgSize + 1, ((j - 1) % canvasHW + 1) * imgSize}}]:copy(interpolationReconstructions[{1, k}]:type(torch.getdefaulttensortype()))
                                                    end
                                                end
                                                if j % canvasHW == 0 then counter = counter + 1 end
                                                zSamples = nil
                                                zInterpolations = nil
                                                samplesReconstructions = nil
                                                samplesSilhouettesReconstructions = nil
                                                interpolationReconstructions = nil
                                                interpolationSilhouettesReconstructions = nil
                                                collectgarbage()
                                            end

                                            if sampleType ~= 'data' then
                                                for k=1, numVPs do
                                                    interpolationCanvas[{{k}, {(counter-2) * imgSize + 1, (counter - 1) * imgSize}, {(canvasHW ^ 2 - 1) % canvasHW * imgSize + 1, ((canvasHW ^ 2 - 1) % canvasHW + 1) * imgSize}}] = dataBeingUsed[{{l}, {k}}]:type(torch.getdefaulttensortype())
                                                    interpolationsilhouetteCanvas[{{k}, {(counter-2) * imgSize + 1, (counter - 1) * imgSize}, {(canvasHW ^ 2 - 1) % canvasHW * imgSize + 1, ((canvasHW ^ 2 - 1) % canvasHW + 1) * imgSize}}]:copy(originalSilhouettes[{{l}, {k}}])
                                                end
                                            end

                                            local handle
                                            if sampleType ~= 'interpolate' then
                                                paths.mkdir(string.format('%s/%s/example-%d/samples', savePathDataInterpolate, data.category[class], nTotalSamples - 1))
                                                paths.mkdir(string.format('%s/%s/example-%d/samples/mask', savePathDataInterpolate, data.category[class], nTotalSamples - 1))
                                            end
                                            if sampleType ~= 'data' then
                                                paths.mkdir(string.format('%s/%s/example-%d/interpolations', savePathDataInterpolate, data.category[class], nTotalSamples - 1))
                                                paths.mkdir(string.format('%s/%s/example-%d/interpolations/mask', savePathDataInterpolate, data.category[class], nTotalSamples - 1))
                                            end
                                            for k=1, numVPs do
                                                
                                                if sampleType ~= 'interpolate' then
                                                    image.save(string.format(savePathDataInterpolate .. '/%s/example-%d/samples/VP-%d.png', data.category[class], nTotalSamples - 1, k-1), canvas[{k}])
                                                    image.save(string.format(savePathDataInterpolate .. '/%s/example-%d/samples/mask/VP-%d.png', data.category[class], nTotalSamples - 1, k-1), silhouetteCanvas[{k}])
                                                end
                                                if sampleType ~= 'data' then
                                                    image.save(string.format(savePathDataInterpolate .. '/%s/example-%d/interpolations/VP-%d.png', data.category[class], nTotalSamples - 1, k-1), interpolationCanvas[{k}])
                                                    image.save(string.format(savePathDataInterpolate .. '/%s/example-%d/interpolations/mask/VP-%d.png', data.category[class], nTotalSamples - 1, k-1), interpolationsilhouetteCanvas[{k}])
                                                end
                                            end
                                        end
                                        -- Clone the mean and [log] variance vectors for interpolation use
                                        zVecPrevExample = zVectors[{{2}}]:clone()
                                    end
                                    canvas = nil
                                    silhouetteCanvas = nil
                                    originalSilhouettes = nil
                                    originalReconstructions = nil
                                    originalSilhouettesReconstructions = nil
                                    model:clearState()
                                    collectgarbage()
                                end -- END if passFlag
                                i = i + 1
                            end -- END while loop
                            tempData = nil
                            model:clearState()
                            collectgarbage()
                        end -- END if matchingElements:sum() > 1
                    end -- END if continueFlag
                end -- END for class
            end -- END for if epoch % 3 == 0
        -- end -- END elseif sampleType == 'data' or sampleType == 'interpolate' or sampleType == 'datainterpolate'
    end -- END if 
end

return sampleManifold