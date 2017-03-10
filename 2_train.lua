#!~/torch/install/bin/th
-- The original VAE implementation obtained from 'https://github.com/y0ast/VAE-Torch'

--[[
TODO: Describe the data structure saved on disk
--]]


require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'optim'
local VAE = require '2_0_VAE'
require '2_1_KLDCriterion'
require '2_2_Sampler'
require 'image'
require 'paths'
local commonFuncs = require '0_commonFuncs'
local sampleManifold = require '3_sampleManifold'

if not opt then

    cmd = torch.CmdLine()
    cmd:text()
    cmd:text()
    cmd:text('Options:')
    -- Global
    cmd:option('-globalDataType', 'float', 'Sets the default data type for Torch tensors: float, double')
    cmd:option('-seed', 1, "The default seed to be used for the random number generator: Any positive integer number")
    cmd:option('-testPhase', 0, 'Whether we want to run some small tests just to make sure everything works using the test set data: 0 | 1')
    cmd:option('-expDirName', '', 'An string to be used for the name of the directory in which the reconstructions and models will be stored')
    cmd:option('-modelPath', '', 'The path for a saved model')
    cmd:option('-benchmark', 0, "Determines how to process the raw data. '0' is used for your own data set: 0 | 1")
    -- Data reading/storing
    cmd:option('-imgSize', 60, 'The size of each image/depth map')
    cmd:option('-numVPs', 20, 'Number of view points for the 3D models')
    cmd:option('-fromScratch', 0, "It indicates whether to use the pre-stored, resized data or do the process of resizing again: 0 | 1")
    cmd:option('-imgSize', 224, '3D grid size. E.g. 224')
    -- Model:
    cmd:option('-nCh', 64, "A multiplier to control the number of feature maps for each convolutional layer")
    cmd:option('-nLatents', 225, 'Determines the number of latent variables: Any positive real number')
    cmd:option('-tanh', 0, "Set to 1 if you want to normalize the input/output values to be between -1 and 1 instead of 0 to 1")
    -- Training:
    cmd:option('-batchSize', 8, 'Batch size for training (SGD): any integer number (1 or higher)')
    cmd:option('-batchSizeChangeEpoch', 15, 'Change the batch size every opt.batchSizeChangeEpoch epochs: any integer number (1 or higher)')
    cmd:option('-batchSizeChange', 5, 'The number to be subtracted/added every opt.batchSizeChangeEpoch from opt.batchSize: any integer number (1 or higher)')
    cmd:option('-targetBatchSize', 25, 'The minimum batch size for training (SGD): any integer number (2 or higher)')
    cmd:option('-nReconstructions', 50, 'An integer indicating how many reconstuctions to be generated from the test data set')
    cmd:option('-initialLR', 0.000003, 'The learning rate to be used for the very first few epochs of training: Any positive small decimal value')
    cmd:option('-lr', 0.00022, 'The learning rate: Any positive decimal value')
    cmd:option('-lrDecay', 0.97, 'A number by which the original learning rate will be multiplied on each epoch: Any real number')
    cmd:option('-maxEpochs', 250, 'The maximum number of epochs: Any positive real number')
    cmd:option('-dropoutNet', 0, 'Indicates whether 15 to 18 views should be dropped during training')
    cmd:option('-VpToKeep', 100, 'Drops all VPs except this one [0 ... 19]. Leave at 100 if do not want to drop a specific VP')
    cmd:option('-onlySilhouettes', 0, 'Indicates whether only the silhouettes must be used for training')
    cmd:option('-singleVPNet', 0, 'If set to 1, will perform random permutation on the input vector view point channels')
    cmd:option('-conditional', 0, 'Indicates whether the model is trained conditionally')
    cmd:option('-KLD', 80, 'The coefficient for the gradients of the KLD loss')
    -- Testing
    cmd:option('-canvasHW', 15, 'Determines the height and width of the canvas on which the samples from the manifold will be drawn: Any positive integer number')
    cmd:option('-nSamples', 15, 'The number of samples to be drawn from the prior (z): Any positive integer number')
    cmd:option('-sampleType', 'random', 'Determines the number of latent variables: data | interpolate | random')
    cmd:option('-sampleCategory', '', "The category name from which one would like to start generating samples. Will be used if opt.sampleType == 'data': A valid category name for which there are examples in the train data set")
    cmd:option('-mean', 0, 'The mean on the z vector elements: Any real number')
    cmd:option('-var', 0, 'The variance of the z vector elements. In case sampleType = data then it indicates the ratio by which the predicted model variance will be multiplied by: Any positive real number')
    cmd:option('-range', 3, 'The range on the real line from which to generate samples: Any positive real number')
    cmd:text()
    opt = cmd:parse(arg or {})

    if opt.fromScratch == 1 then opt.fromScratch = true elseif opt.fromScratch == 0 then opt.fromScratch = false else print "==> Incorrect value for fromScratch argument" os.exit() end
    if opt.testPhase == 1 then opt.testPhase = true elseif opt.testPhase == 0 then opt.testPhase = false else print "==> Incorrect value for testPhase argument" os.exit() end
    if opt.benchmark == 1 then opt.benchmark = true elseif opt.benchmark == 0 then opt.benchmark = false else print "==> Incorrect value for 'benchmark' argument" os.exit() end
    if opt.tanh == 1 then opt.tanh = true elseif opt.tanh == 0 then opt.tanh = false else print "==> Incorrect value for 'tanh' argument" os.exit() end
    if opt.dropoutNet == 1 then opt.dropoutNet = true opt.VpToKeep = opt.VpToKeep + 1 elseif opt.dropoutNet == 0 then opt.dropoutNet = false opt.VpToKeep = 30 else print "==> Incorrect value for dropoutNet argument" os.exit() end
    if opt.onlySilhouettes == 1 then opt.onlySilhouettes = true elseif opt.onlySilhouettes == 0 then opt.onlySilhouettes = false else print "==> Incorrect value for 'onlySilhouettes' argument" os.exit() end
    if opt.singleVPNet == 1 then opt.singleVPNet = true elseif opt.singleVPNet == 0 then opt.singleVPNet = false else print "==> Incorrect value for 'singleVPNet' argument" os.exit() end
    if opt.conditional == 1 then opt.conditional = true elseif opt.conditional == 0 then opt.conditional = false else print "==> Incorrect value for 'conditional' argument" os.exit() end
    if opt.batchSize < 2 then print '==> The batch size cannot be less than 3 for technical reasons' opt.batchSize = 2 end

    -- Set the default data type for Torch
    if opt.globalDataType == 'float' then torch.setdefaulttensortype('torch.FloatTensor') dataTypeNumBytes = 4
    elseif opt.globalDataType == 'double' then torch.setdefaulttensortype('torch.DoubleTensor') dataTypeNumBytes = 8
    else print ("You are not allowed to use Torch data type other than 'float' or 'double'. Please set the input 'globalDataType' to either 'float' or 'double'") end

    if opt.seed > 0 then torch.manualSeed(opt.seed) end
    if not opt.lr or opt.lr <= 0 then opt.lr = 0.0002 end
    if opt.nCh < 1 then opt.nCh = 1 end
    if opt.expDirName == '' then opt.expDirName = string.format('exp%.4f', tostring(torch.rand(1):totable()[1])) end
end

print ("============= Train, Validation and Test Phase =============")

if opt.imgSize == 0 then opt.imgSize = nil end

nngraph.setDebug(false)
-- Get the train, validation and test data set paths
local trainDataFiles, validationDataFiles, testDataFiles = commonFuncs.obtainDataPath(opt.benchmark, opt.testPhase, true)

if opt.testPhase == true then
    -- Just do everything with one file
    for i=2, #trainDataFiles do
        trainDataFiles[i] = nil
    end

    for i=2, #validationDataFiles do
        validationDataFiles[i] = nil
    end

    for i=2, #testDataFiles do
        testDataFiles[i] = nil
    end
    collectgarbage()
end

if trainDataFiles then -- If there are training files on disk
    print ("==> Loading the first training file. Please wait ...")
    local data = torch.load(trainDataFiles[1])
    if opt.tanh then data.dataset = commonFuncs.normalizeMinusOneToOne(data.dataset) end

    local inputTensorSize = {data.dataset:size(2), opt.imgSize, opt.imgSize}
    local encoder, parallelDecoder, discriminator, perceptLossModel
    local batch_size = opt.batchSize


    -- Build the model
    local modelParams = {opt.numVPs, opt.nCh, opt.nLatents, opt.tanh, opt.singleVPNet, opt.conditional, not opt.conditional and 0 or #data.category}

    local input = nn.Identity()()
    local conditionVector = nn.Identity()()
    local silhouettes = nn.Identity()()

    local encoder = VAE.get_encoder(modelParams)
    local mean, log_var, predClassScores
    if opt.conditional then
        mean, log_var, predClassScores = encoder(input):split(3)
    else
        mean, log_var = encoder(input):split(2)
    end

    local z = nn.Sampler()({mean, log_var})

    local decoder = VAE.get_decoder(modelParams)
    local reconstruction
    if opt.conditional then
        reconstruction = decoder({z, conditionVector})
    else
        reconstruction = decoder(z)
    end

    -- Build the gModule using nngraph
    local model
    if opt.conditional then
        model = nn.gModule({input, conditionVector},{reconstruction, mean, log_var, predClassScores})
    else
        model = nn.gModule({input},{reconstruction, mean, log_var})
    end
    local modelTest = model:clone('weight', 'bias', 'gradWeight', 'gradBias', 'running_mean', 'runnig_var', 'save_mean', 'save_var')

    local gMod = nn.Container()
    gMod:add(model):add(modelTest):add(encoder):add(decoder)
    gMod = gMod:cuda()
    if pcall(require, 'cudnn') then
        require 'cudnn'
        cudnn.benchmark = true
        cudnn.fastest = true
        cudnn.convert(gMod, cudnn)
        print '\n'
    end

    -- The loss functions
    local cr1, cr2, criterion, oldCriterion, perceptualCriterion, depthCriterion, classLabelCriterion
    cr1 = nn.AbsCriterion()
    cr1.sizeAverage = false
    cr2 = nn.AbsCriterion()
    cr2.sizeAverage = false
    criterion = nn.ParallelCriterion():add(cr1):add(cr2)
    criterion = criterion:cuda()
    KLD = nn.KLDCriterion(opt.KLD):cuda()
    if opt.conditional then
        classLabelCriterion = nn.CrossEntropyCriterion()
        classLabelCriterion.sizeAverage = false
        classLabelCriterion = classLabelCriterion:cuda()
    end


    -- Some code to draw computational graph
    -- dummy_x = torch.rand(dim_input)
    -- model:forward({dummy_x})

    -- Uncomment to get structure of the Variational Autoencoder
    -- model = model:cuda()
    -- model:forward(torch.rand(2, 20, 224, 224):cuda())
    -- graph.dot(model.fg, 'Variational Autoencoder', 'net')
    -- os.exit()

    -- Transfer the model and the data to the GPU, if there is enough memory on the GPU for both the model and data batch
    model:evaluate()
    local parameters, gradients = gMod:getParameters()
    print ("==> Configurations, expDirName: " .. opt.expDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. batch_size .. ", Batch Size Change Epoch: " .. opt.batchSizeChangeEpoch .. ", Batch Size Change: " .. opt.batchSizeChange .. ", Target Batch Size: " .. opt.targetBatchSize .. ", No. Output Channels: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', DropoutNet: ' .. (opt.dropoutNet and "True" or "False") .. ', KeepVP: ' .. opt.VpToKeep .. ', onlySilhouettes: ' .. (opt.onlySilhouettes and "True" or "False") .. ', singleVPNet: ' .. (opt.singleVPNet and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False"))

    -- Start training, validation and testing
    model:training()
    local epoch = 1
    local lrDecayDrastic = 0
    local validationTotalErrorList = {}
    local trainTotalErrorList = {}
    local trainKLDErrList = {}
    local trainSilhouetteErrList = {}
    local trainDepthMapErrList = {}
    local trainClassErrList = {}
    local trainClassAccuracyList = {}
    local validKLDErrList = {}
    local validSilErrList = {}
    local validDepthMapErrList = {}
    local validClassErrList = {}
    local validClassAccuracyList = {}
    local continueTraining = true

    local config = {
        learningRate = opt.initialLR
    }
    local state = {}

    print ("==> Number of Model Parameters: " .. parameters:nElement())
    while continueTraining and epoch <= opt.maxEpochs do

        local totalError = 0
        local tic = torch.tic()
        local numTrainSamples = 0
        local empiricalMeansLabels = {}
        local empiricalMeans = {}
        local empiricalLog_Vars = {}
        local sampleZembeddings = {} -- A table with two entries: a tensor containings the sampled means, a tensor
        trainTotalErrorList[epoch] = 0
        trainKLDErrList[epoch] = 0
        trainSilhouetteErrList[epoch] = 0
        trainDepthMapErrList[epoch] = 0
        trainClassErrList[epoch] = 0
        trainClassAccuracyList[epoch] = 0

        print ("==> Epoch: " .. epoch .. " Training for '" .. #trainDataFiles .. "' file(s) containing the train data set samples on the disk")
        for i=2, #trainDataFiles + 1 do -- For all training data set files

            if data.dataset:size(1) < batch_size then batch_size = data.dataset:size(1) end
            indices = commonFuncs.generateBatchIndices(data.dataset:size(1), batch_size)
            if #indices[#indices] ~= batch_size then
                indices[#indices] = nil
            end
            numTrainSamples = numTrainSamples + #indices * batch_size
            empiricalMeans[i-1] = torch.zeros(1, opt.nLatents):type(gMod:type())
            empiricalLog_Vars[i-1] = torch.zeros(1, opt.nLatents):type(gMod:type())
            empiricalMeansLabels[i-1] = torch.zeros(1):type(gMod:type())


            ticc = torch.tic()
            for t,v in ipairs(indices) do
                
                local targetClassIndices, targetClassHotVec, droppedInputs
                local depthMaps = torch.CudaTensor(torch.LongStorage(commonFuncs.tableConcat({v:size(1)}, inputTensorSize))):copy(data.dataset:index(1,v))
                -- Create vectors to be used for training and computing classification accuracy
                local labels = data.labels:index(1, v)
                if opt.conditional then
                    targetClassIndices = data.labels:index(1, v):type(depthMaps:type())
                    targetClassHotVec = torch.zeros(batch_size, #data.category)
                    for l=1, labels:nElement() do
                        targetClassHotVec[l][labels[l]] = 1
                    end
                    targetClassHotVec = targetClassHotVec:type(depthMaps:type())
                end

                -- Get the silhouette for the current samples
                local silhouettes = depthMaps:clone()
                if opt.tanh then
                    silhouettes[silhouettes:gt(-1)] = 1
                    silhouettes[silhouettes:eq(-1)] = 0
                else
                    silhouettes[silhouettes:gt(0)] = 1
                end

                local opfunc = function(x)
                    if x ~= parameters then
                        parameters:copy(x)
                    end

                    gMod:zeroGradParameters()
                    local reconstruction, mean, log_var, predictedClassScores, disReal, disFake, disGen


                    if opt.dropoutNet then
                        -- Randomly drop 15-18 views (used for DropOutNet)
                        droppedInputs = commonFuncs.dropInputVPs(not opt.onlySilhouettes and depthMaps or silhouettes, opt.VpToKeep, false, nil, nil, opt.singleVPNet, nil, targetClassHotVec)
                    else
                        droppedInputs = commonFuncs.dropInputVPs(not opt.onlySilhouettes and depthMaps or silhouettes, nil, false, nil, nil, opt.singleVPNet, nil, targetClassHotVec)
                    end
                    reconstruction, mean, log_var, predictedClassScores= unpack(model:forward(droppedInputs))

                    -- Fill the empirical distribution mean and log_var matrices
                    empiricalMeansLabels[i-1] = torch.cat(empiricalMeansLabels[i-1], data.labels:index(1, v):type(model:type()), 1)
                    empiricalMeans[i-1] = torch.cat(empiricalMeans[i-1], mean, 1)
                    empiricalLog_Vars[i-1] = torch.cat(empiricalLog_Vars[i-1], log_var, 1)
                    if t == 1 then
                        empiricalMeansLabels[i-1] = empiricalMeansLabels[i-1][{{2, 1+batch_size}}]
                        empiricalMeans[i-1] = empiricalMeans[i-1][{{2, 1+batch_size}}]
                        empiricalLog_Vars[i-1] = empiricalLog_Vars[i-1][{{2, 1+batch_size}}]
                    end
                    
                    -- The error & gradient
                    local dEn_dwClass
                    if opt.conditional then
                        local classErr = classLabelCriterion:forward(predictedClassScores, targetClassIndices)
                        -- The number of examples from the current batch that are correctly classified
                        trainClassAccuracyList[epoch] = trainClassAccuracyList[epoch] + commonFuncs.computeClassificationAccuracy(predictedClassScores, targetClassIndices)
                        dEn_dwClass = classLabelCriterion:backward(predictedClassScores, targetClassIndices)
                        trainClassErrList[epoch] = trainClassErrList[epoch] + classErr
                    end
                    criterion:forward(reconstruction, {depthMaps, silhouettes})
                    local df_dw = criterion:backward(reconstruction, {depthMaps, silhouettes})

                    if opt.tanh then
                        -- Compute the error after converting the outputs back to [0-1] so that it's easier to interpret
                        local tempRecon = {}
                        tempRecon[1] = reconstruction[1]:clone()
                        tempRecon[2] = reconstruction[2]:clone()
                        tempInputs = depthMaps:clone()
                        tempRecon[1] = commonFuncs.normalizeBackToZeroToOne(tempRecon[1])
                        tempInputs = commonFuncs.normalizeBackToZeroToOne(tempInputs)
                        criterion:forward(tempRecon, {tempInputs, silhouettes})
                    end
                    
                    trainSilhouetteErrList[epoch] = trainSilhouetteErrList[epoch] + criterion.criterions[2].output
                    trainDepthMapErrList[epoch] = trainDepthMapErrList[epoch] + criterion.criterions[1].output
                    local err = criterion.output

                    local dKLD_dmu, dKLD_dlog_var, batchTotalError
                    local KLDerr = KLD:forward(mean, log_var)
                    dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))
                    
                    local error_grads
                    if opt.conditional then
                        error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var, dEn_dwClass}
                    else
                        error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
                    end

                    trainKLDErrList[epoch] = trainKLDErrList[epoch] + KLDerr
                    batchTotalError = err + KLDerr

                    -- Compute the backward pass for the model
                    model:backward(droppedInputs, error_grads)

                    if t % 2 == 0 then collectgarbage() end
                    return batchTotalError, gradients
                end

                x, batchTotalError = optim.adam(opfunc, parameters, config, state)

                totalError = totalError + batchTotalError[1]
                
                inputs = nil
                silhouettes = nil
                batchTotalError = nil
                collectgarbage()
            end

            data = nil
            collectgarbage()
            if i <= #trainDataFiles then
                data = torch.load(trainDataFiles[i])
                if opt.tanh then data.dataset = commonFuncs.normalizeMinusOneToOne(data.dataset) end
            end
            collectgarbage()
        end -- for i=2, #trainDataFiles + 1
        

        trainTotalErrorList[epoch] = totalError/numTrainSamples
        trainKLDErrList[epoch] = trainKLDErrList[epoch]/numTrainSamples
        trainSilhouetteErrList[epoch] = trainSilhouetteErrList[epoch]/numTrainSamples
        trainDepthMapErrList[epoch] = trainDepthMapErrList[epoch]/numTrainSamples
        trainClassErrList[epoch] = trainClassErrList[epoch]/numTrainSamples
        trainClassAccuracyList[epoch] = trainClassAccuracyList[epoch]/numTrainSamples

        if opt.conditional then
            print(string.format("==> Epoch: %d, Total Err: %d, KLD: %.1f, Sil. Err: %d, Depth Map Err: %d, Class. Err: %.2f, Acc: %.3f", epoch, totalError/numTrainSamples, trainKLDErrList[epoch], trainSilhouetteErrList[epoch], trainDepthMapErrList[epoch], trainClassErrList[epoch], trainClassAccuracyList[epoch]) .. ". No. Train 3D Models: " .. numTrainSamples)
        else
            print(string.format("==> Epoch: %d, Total Err: %d, KLD: %.1f, Sil. Err: %d, Depth Map Err: %d", epoch, totalError/numTrainSamples, trainKLDErrList[epoch], trainSilhouetteErrList[epoch], trainDepthMapErrList[epoch]) .. ". No. Train 3D Models: " .. numTrainSamples)
        end

        data = nil
        gMod:clearState()
        collectgarbage()

        -- Validation
        model:evaluate()
        local validationTotalError = 0
        local batchTotalError = 0
        local numValidSamples = 0
        data = torch.load(validationDataFiles[1])
        if opt.tanh then data.dataset = commonFuncs.normalizeMinusOneToOne(data.dataset) end
        validationTotalErrorList[epoch] = 0
        validKLDErrList[epoch] = 0
        validSilErrList[epoch] = 0
        validDepthMapErrList[epoch] = 0
        validClassErrList[epoch] = 0
        validClassAccuracyList[epoch] = 0
        print ("==> Epoch: " .. epoch .. " Validation for '" .. #validationDataFiles .. "' file(s) containing the validation data set samples on the disk")
        for i=2, #validationDataFiles + 1 do

            indices = commonFuncs.generateBatchIndices(data.dataset:size(1), batch_size)
            if #indices[#indices] ~= batch_size then
                indices[#indices] = nil
            end
            numValidSamples = numValidSamples + #indices * batch_size
            
            for t,v in ipairs(indices) do
                local droppedInputs, labels, targetClassIndices, targetClassHotVec
                local depthMaps = torch.CudaTensor(torch.LongStorage(commonFuncs.tableConcat({v:size(1)}, inputTensorSize))):copy(data.dataset:index(1,v))
                local labels = data.labels:index(1, v)
                -- Create vectors to be used for training and computing classification accuracy
                if opt.conditional then
                    targetClassIndices = data.labels:index(1, v):type(depthMaps:type())
                    targetClassHotVec = torch.zeros(batch_size, #data.category)
                    for l=1, labels:nElement() do
                        targetClassHotVec[l][labels[l]] = 1
                    end
                    targetClassHotVec = targetClassHotVec:type(depthMaps:type())
                end

                -- Get the silhouettes for the current samples
                local silhouettes = depthMaps:clone()
                if opt.tanh then
                    silhouettes[silhouettes:gt(-1)] = 1
                    silhouettes[silhouettes:eq(-1)] = 0
                else
                    silhouettes[silhouettes:gt(0)] = 1
                end

                local reconstruction, mean, log_var, predictedClassScores
                if opt.dropoutNet then
                    droppedInputs = commonFuncs.dropInputVPs(not opt.onlySilhouettes and depthMaps or silhouettes, opt.VpToKeep, false, nil, nil, opt.singleVPNet, nil, targetClassHotVec)
                    reconstruction, mean, log_var, predictedClassScores = unpack(model:forward(droppedInputs))
                else
                    droppedInputs = commonFuncs.dropInputVPs(not opt.onlySilhouettes and depthMaps or silhouettes, nil, false, nil, nil, opt.singleVPNet, nil, targetClassHotVec)
                    reconstruction, mean, log_var, predictedClassScores = unpack(model:forward(droppedInputs))
                end

                if opt.conditional then
                    classErr = classLabelCriterion:forward(predictedClassScores, targetClassIndices)
                    validClassAccuracyList[epoch] = validClassAccuracyList[epoch] + commonFuncs.computeClassificationAccuracy(predictedClassScores, targetClassIndices)
                    validClassErrList[epoch] = validClassErrList[epoch] + classErr
                end
                if opt.tanh then
                    -- Compute the error after converting the outputs back to [0-1] scale so that it's easy to
                    -- compare the error with the sigmoid-generated ones
                    local tempRecon = {}
                    tempRecon[1] = reconstruction[1]:clone()
                    tempRecon[2] = reconstruction[2]:clone()
                    tempInputs = depthMaps:clone()
                    tempRecon[1] = commonFuncs.normalizeBackToZeroToOne(tempRecon[1])
                    tempInputs = commonFuncs.normalizeBackToZeroToOne(tempInputs)
                    criterion:forward(tempRecon, {tempInputs, silhouettes})
                else
                    criterion:forward(reconstruction, {depthMaps, silhouettes})
                end

                validSilErrList[epoch] = validSilErrList[epoch] + criterion.criterions[2].output
                validDepthMapErrList[epoch] = validDepthMapErrList[epoch] + criterion.criterions[1].output
                local err = criterion.output
                
                local KLDerr = KLD:forward(mean, log_var)
                validKLDErrList[epoch] = validKLDErrList[epoch] + KLDerr
                batchTotalError = batchTotalError + err + KLDerr

                if t % 2 == 0 then collectgarbage() end
                -- Some clean up
                depthMaps = nil
                silhouettes = nil
            end
            
            validationTotalError = validationTotalError + batchTotalError
            batchTotalError = 0
            

            -- Some clean up and load the next file, if any
            data = nil
            criterion.output = nil
            KLD.output = nil
            collectgarbage()
            if i <= #validationDataFiles then
                data = torch.load(validationDataFiles[i])
                if opt.tanh then data.dataset = commonFuncs.normalizeMinusOneToOne(data.dataset) end
            end

        end -- for i=2, #validationDataFiles + 1

        validationTotalErrorList[epoch] = validationTotalError/numValidSamples
        validKLDErrList[epoch] = validKLDErrList[epoch]/numValidSamples
        validSilErrList[epoch] = validSilErrList[epoch]/numValidSamples
        validDepthMapErrList[epoch] = validDepthMapErrList[epoch]/numValidSamples
        validClassErrList[epoch] = validClassErrList[epoch]/numValidSamples
        validClassAccuracyList[epoch] = validClassAccuracyList[epoch]/numValidSamples
        if opt.conditional then
            print(string.format("==> Epoch: %d, Total Err: %d, KLD: %.1f, Sil. Err: %d, Depth Map Err: %d, Class. Err. %.2f, Acc: %.3f", epoch, validationTotalError/numValidSamples, validKLDErrList[epoch], validSilErrList[epoch], validDepthMapErrList[epoch], validClassErrList[epoch], validClassAccuracyList[epoch]) .. ". No. Valid. 3D Models: " .. numValidSamples)
        else
            print(string.format("==> Epoch: %d, Total Err: %d, KLD: %.1f, Sil. Err: %d, Depth Map Err: %d", epoch, validationTotalError/numValidSamples, validKLDErrList[epoch], validSilErrList[epoch], validDepthMapErrList[epoch]) .. ". No. Valid. 3D Models: " .. numValidSamples)
        end

        data = nil
        gMod:clearState()
        collectgarbage()

        sampleZembeddings = commonFuncs.combineMeanLogVarTensors(empiricalMeans, empiricalLog_Vars, empiricalMeansLabels)
        -- Print some statistics for the Zs
        -- sampleZembeddings[2]:exp() -- For easier interpretability
        -- print (string.format('==> Mean:mean(1):mean(): %.3f, Mean:var(1):mean(): %.3f, Mean:var(1):std(): %.3f, Mean:var(1):min(): %.3f, mean:var(1):max(): %.3f', sampleZembeddings[1]:mean(1):mean(), sampleZembeddings[1]:var(1):mean(), sampleZembeddings[1]:var(1):std(), sampleZembeddings[1]:var(1):min(), sampleZembeddings[1]:var(1):max()))
        -- print ( string.format('==> Var:mean(1):mean(): %.3f, Var:var(1):mean(): %.3f, Var:var(1):std(): %.3f, Var:var(1):min(): %.3f, Var:var(1):max(): %.3f', sampleZembeddings[2]:mean(1):mean(), sampleZembeddings[2]:var(1):mean(), sampleZembeddings[2]:var(1):std(), sampleZembeddings[2]:var(1):min(), sampleZembeddings[2]:var(1):max()))
        -- print ( string.format('==> Means:max(): %.3f, Means:min(): %.2f, percent > 1.5: %.3f, percent < -1.5: %.3f', sampleZembeddings[1]:max(), sampleZembeddings[1]:min(), sampleZembeddings[1][sampleZembeddings[1]:gt(1.5)]:nElement()/sampleZembeddings[1]:nElement(), sampleZembeddings[1][sampleZembeddings[1]:lt(-1.5)]:nElement()/sampleZembeddings[1]:nElement()))
        -- print ( string.format('==> Vars:max(): %.3f, Vars:min(): %.3f, percent > 0.8: %.3f, percent < 0.1: %.3f', sampleZembeddings[2]:max(), sampleZembeddings[2]:min(), sampleZembeddings[2][sampleZembeddings[2]:gt(0.8)]:nElement()/sampleZembeddings[2]:nElement(), sampleZembeddings[2][sampleZembeddings[2]:lt(0.1)]:nElement()/sampleZembeddings[2]:nElement()))
        -- sampleZembeddings[2]:log()


        if epoch == 1 then
            trainTotalErrTensor = torch.Tensor(1,1):fill(totalError/numTrainSamples)
            validTotalErrTensor = torch.Tensor(1,1):fill(validationTotalError/numValidSamples)
        else
            trainTotalErrTensor = torch.cat(trainTotalErrTensor,torch.Tensor(1,1):fill(totalError/numTrainSamples),1)
            validTotalErrTensor = torch.cat(validTotalErrTensor, torch.Tensor(1,1):fill(validationTotalError/numValidSamples),1)
        end
        if lrDecayDrastic <= 3 and epoch >= 18 and epoch % 6 == 0 then
            lrDecayDrastic = lrDecayDrastic + 1
            print ("==> Learning rate has been DRASTICALLY decreased to " .. config.learningRate * 0.35 .. " from its previous value of " .. config.learningRate)
            config.learningRate = config.learningRate * 0.35
        end

        
        -- Reconstruct some of the test samples
        N = 1
        if continueTraining and epoch >= 14 and epoch % 2 == 0 then
            data = torch.load(testDataFiles[1])
            reconBatchSizePerTestFile = math.floor(opt.nReconstructions / #testDataFiles)
            local reconItersPerTestFile = 1
            if reconBatchSizePerTestFile > 50 then -- At most, only transfer 50 samples to GPU
                while reconBatchSizePerTestFile > 50 do
                    reconBatchSizePerTestFile = math.floor(reconBatchSizePerTestFile * 0.9)
                end
                reconItersPerTestFile = math.ceil(opt.nReconstructions / reconBatchSizePerTestFile)
            end


            print ("==> Reconstructing " ..  math.floor(opt.nReconstructions / #testDataFiles) * #testDataFiles  .. " randomly-selected 3D models from the test set")

            for j=2, #testDataFiles + 1 do

                local numRecon = 0
                local indices
                if reconBatchSizePerTestFile <= data.dataset:size(1) and reconBatchSizePerTestFile <= opt.batchSize then
                    indices = torch.randperm(data.dataset:size(1)):long():split(reconBatchSizePerTestFile)
                elseif reconBatchSizePerTestFile > opt.batchSize then
                    indices = torch.randperm(data.dataset:size(1)):long():split(opt.batchSize)
                else
                    indices = {torch.linspace(1, data.dataset:size(1), data.dataset:size(1)):long()}
                end

                
                if #indices > 1 then
                    local tempIndices = {}
                    for ll=1, data.dataset:size(1) - reconBatchSizePerTestFile * (#indices - 1) do
                        tempIndices[ll] = indices[#indices][ll]
                    end
                    
                    -- The Batch Normalization layers require 4D tensors
                    if #tempIndices > 1 then
                        indices[#indices] = torch.LongTensor(tempIndices)
                    else
                        indices[#indices] = nil
                    end
                end
                
                local flag = true
                local recon, silhouettes
                local labels
                for t, v in ipairs(indices) do
                    -- xlua.progress(t, #indices)
                    local depthMaps, silhouettes, droppedInputs
                    if flag then
                        depthMaps = torch.CudaTensor(torch.LongStorage(commonFuncs.tableConcat({v:size(1)}, inputTensorSize))):copy(data.dataset:index(1,v))

                        -- Get the silhouettes for the current samples
                        silhouettes = depthMaps:clone()
                        if opt.tanh then
                        silhouettes[silhouettes:gt(-1)] = 1
                        silhouettes[silhouettes:eq(-1)] = 0
                        else
                            silhouettes[silhouettes:gt(0)] = 1
                        end

                        local mean, log_var, predictedClassScores
                        if opt.dropoutNet or opt.singleVPNet then
                            droppedInputs = commonFuncs.dropInputVPs({depthMaps, silhouettes}, not opt.singleVPNet and opt.VpToKeep or nil, true, nil, nil, opt.singleVPNet)
                            if opt.conditional then
                                mean, log_var, predictedClassScores = unpack(encoder:forward(not opt.onlySilhouettes and droppedInputs[1] or droppedInputs[2]))
                                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, #data.category)
                                recon = decoder:forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
                            else
                                recon = unpack(model:forward(not opt.onlySilhouettes and droppedInputs[1] or droppedInputs[2]))
                            end
                        else
                            droppedInputs = commonFuncs.dropInputVPs(not opt.onlySilhouettes and depthMaps or silhouettes, nil, true, nil, nil, opt.singleVPNet)
                            if opt.conditional then
                                mean, log_var, predictedClassScores = unpack(encoder:forward(droppedInputs))
                                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, #data.category)
                                recon = decoder:forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
                            else
                                recon = unpack(model:forward(droppedInputs))
                            end
                        end                        

                        reconSil = recon[2]:clone()
                        recon[2] = nil
                        recon = recon[1]
                        collectgarbage()

                        if opt.tanh then
                            recon = commonFuncs.normalizeBackToZeroToOne(recon)
                        end
                        
                        k = 1
                        while k <= depthMaps:size(1) and numRecon < math.floor(opt.nReconstructions / #testDataFiles) do
                            local tempRecon = recon[k]:view(opt.numVPs, opt.imgSize, opt.imgSize)
                            local tempOr = depthMaps[k]
                            local tempLabel = data.labels[v[k]]

                            local tempSilRecon, tempSilOrig
                            tempSilRecon = reconSil[k]
                            tempSilOrig = silhouettes[k]
                            tempSilOrig[tempSilOrig:gt(0)] = 1
                            local handle = assert(io.popen(string.format('mkdir -p %s/images-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/reconstruction/model%d-%s', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch, N, data.category[tempLabel])))
                            local handle = assert(io.popen(string.format('mkdir -p %s/images-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/reconstruction/model%d-%s/mask', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch, N, data.category[tempLabel])))
                            handle:close()

                            for ll=1, opt.numVPs do
                                image.save(string.format('%s/images-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/reconstruction/model%d-%s/file%d-img%d-%d-rec.png',opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch, N, data.category[tempLabel], j - 2, v[k], ll-1), tempRecon[ll])
                                image.save(string.format('%s/images-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/reconstruction/model%d-%s/file%d-img%d-%d-or.png', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch, N, data.category[tempLabel], j - 2, v[k], ll-1), tempOr[ll])
                                image.save(string.format('%s/images-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/reconstruction/model%d-%s/mask/file%d-img%d-%d-rec.png',opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch, N, data.category[tempLabel], j - 2, v[k], ll-1), tempSilRecon[ll])
                                image.save(string.format('%s/images-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/reconstruction/model%d-%s/mask/file%d-img%d-%d-or.png', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch, N, data.category[tempLabel], j - 2, v[k], ll-1), tempSilOrig[ll])
                            end
                            numRecon = numRecon + 1
                            N = N + 1
                            k = k + 1
                        end
                        gMod:clearState()
                        if numRecon >= math.floor(opt.nReconstructions / #testDataFiles) then flag = false end
                        recon = nil
                        depthMaps = nil
                        silhouettes = nil
                        collectgarbage()
                    end
                end
                data = nil
                collectgarbage()
                if j <= #testDataFiles then
                    data = torch.load(testDataFiles[j])
                else
                    trainDataFiles = commonFuncs.randPermTableContents(trainDataFiles)
                    testDataFiles = commonFuncs.randPermTableContents(testDataFiles)
                    data = torch.load(trainDataFiles[1]) -- Load the first training file for the next epoch
                    if opt.tanh then data.dataset = commonFuncs.normalizeMinusOneToOne(data.dataset) end
                end
            end -- END for j=2, #testDataFiles + 1
        else
            trainDataFiles = commonFuncs.randPermTableContents(trainDataFiles)
            data = torch.load(trainDataFiles[1]) -- Load the first training file for the next epoch
            -- data.dataset = data.dataset[{{1, data.dataset:size(1)}, {1}}]
            if opt.tanh then data.dataset = commonFuncs.normalizeMinusOneToOne(data.dataset) end
        end -- END if continueTraining then and epoch >= 10


        -- Save the model and parameters
        local handle = assert(io.popen(string.format('mkdir -p %s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch)))
        handle:close()
        if continueTraining and epoch >= 14 and epoch % 2 == 0 then
            state.v = state.v:float()
            state.m = state.m:float()
            state.denom = state.denom:float()
            collectgarbage()
            print (string.format('==> Saving the model on iteration %d.', epoch))
            gMod:clearState() -- Clear the gradInput and output fields of the modules for the model before saving
            torch.save(string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/model.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch), gMod:clone():float())
            torch.save(string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/parameters.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch), parameters:clone():float())
            torch.save(string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/state.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch), state)
            if sampleZembeddings then
                for l=1, #sampleZembeddings do
                    sampleZembeddings[l] = sampleZembeddings[l]:float()
                end
                torch.save(string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/mean_logvar.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch), sampleZembeddings)
                for l=1, #sampleZembeddings do
                    sampleZembeddings[l] = sampleZembeddings[l]:cuda()
                end
            end

            state.v = state.v:cuda()
            state.m = state.m:cuda()
            state.denom = state.denom:cuda()
            collectgarbage()
        end

        -- Sample the manifold

        if continueTraining and epoch >= 14 and epoch %2 == 0 then
            print ('==> Doing sampling/interpolation with the model. Configs: Type -- ' .. opt.sampleType .. ', Number of Samples: ' .. opt.nSamples .. ', Canvas Size: ' .. opt.canvasHW .. ' x ' .. opt.canvasHW .. ', Mean: ' .. opt.mean .. ', Variance: ' .. opt.var)
            local samplesPath = string.format(paths.cwd() .. '/%s/images-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/manifold',opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, epoch)
            sampleManifold.sample(opt.sampleType, opt.sampleCategory, opt.canvasHW, opt.nSamples, data, modelTest, '', samplesPath, opt.mean, opt.var, opt.nLatents, opt.imgSize, opt.numVPs, epoch, false, opt.testPhase, opt.tanh, opt.dropoutNet, opt.VpToKeep, opt.onlySilhouettes, sampleZembeddings, opt.singleVPNet, opt.conditional, nil, opt.benchmark)
        end


        --Save train and validation lowerbound Torch tensors on disk
        local trainErPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/ErrTotalTrainSet.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr)
        local trainErKLDPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/ErrKLDTrainSet.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr)
        local trainErDepthMapPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/ErrDepthMapTrainSet.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr)
        torch.save(trainErPath, torch.FloatTensor(trainTotalErrTensor))
        torch.save(trainErKLDPath, torch.FloatTensor(trainKLDErrList))
        torch.save(trainErDepthMapPath, torch.FloatTensor(trainDepthMapErrList))

        local validErPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/ErrTotalValidationSet.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr)
        local validErKLDPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/ErrKLDValidationSet.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr)
        local validErDepthMapPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/ErrDepthMapValidationSet.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr)
        torch.save(validErPath, torch.FloatTensor(validTotalErrTensor))
        torch.save(validErKLDPath, torch.FloatTensor(validKLDErrList))
        torch.save(validErDepthMapPath, torch.FloatTensor(validDepthMapErrList))

        local trainErSilPath = '' local validErSilPath = ''
        trainErSilPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/ErrMaskTrainSet.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr)
        validErSilPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/ErrMaskValidationSet.t7', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr)
        torch.save(trainErSilPath, torch.FloatTensor(trainSilhouetteErrList))
        torch.save(validErSilPath, torch.FloatTensor(validSilErrList))
        
        local ErrorPlotNames = {'ErrorTotal - L1Depth-L1Sil', 'ErrorKLD - L1Depth-L1Sil', 'ErrorDepthMap - L1Depth-L1Sil', 'ErrorMask - L1Depth-L1Sil'}
        local trainErrPaths = {trainErPath, trainErKLDPath, trainErDepthMapPath, trainErSilPath}
        local validErrPaths = {validErPath, validErKLDPath, validErDepthMapPath, validErSilPath}

        -- Save a plot of train and validation lowerbound error
        local plotTitle = "exp: " .. opt.expDirName .. ", Latent: " .. opt.nLatents .. ", Batch: " .. opt.batchSize .. ", CNN Ch.: " .. opt.nCh .. ", lr " .. opt.lr
        local plotYAxis = string.format("KLD+%s", 'L1Depth-L1Sil')
        local plotSavePath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/', opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr)
        commonFuncs.plotError(trainErrPaths, validErrPaths, ErrorPlotNames, plotYAxis, plotTitle, plotSavePath)

        -- Learning rate for the first 12 epochs
        if epoch <= 10 then
            config.learningRate = torch.linspace(opt.initialLR, opt.lr, 26)[(epoch + 1) * 2]
        elseif epoch == 11 then
            config.learningRate = opt.lr
            commonFuncs.clearOptimState(state, true)
        end

        -- Learning rate decay
        if epoch >= 12 and epoch <= 30 and opt.lrDecay ~= 1 then 
            print ('==> LR decay: The learning rate has been changed to ' .. config.learningRate * opt.lrDecay .. ' from its previous value of ' .. config.learningRate)
            config.learningRate = config.learningRate * opt.lrDecay
        elseif epoch > 30 and epoch < 60 or epoch > 60 and config.learningRate >= 0.000002 then
            print ('==> LR decay: The learning rate has been changed to ' .. config.learningRate * math.max(opt.lrDecay - 0.01, 0.94) .. ' from its previous value of ' .. config.learningRate)
            config.learningRate = config.learningRate * math.max(opt.lrDecay - 0.01, 0.94)
        end
        
        -- Change the batch size
        if epoch % opt.batchSizeChangeEpoch == 0 then
            local prevBatchSize = batch_size
            if batch_size < opt.targetBatchSize then
                print ('==> The new batch size is ' .. math.min(math.ceil(batch_size + opt.batchSizeChange), opt.targetBatchSize) .. '. The previous batch size was ' .. batch_size)
                batch_size = math.min(math.ceil(batch_size + opt.batchSizeChange), opt.targetBatchSize)
            elseif batch_size > opt.targetBatchSize then
                print ('==> The new batch size is ' .. math.max(math.ceil(batch_size - opt.batchSizeChange), opt.targetBatchSize) .. '. The previous batch size was ' .. batch_size)
                batch_size = math.max(math.ceil(batch_size - opt.batchSizeChange), opt.targetBatchSize)
            end
        end

        if lrDecayDrastic <= 3 and epoch >= 18 and (epoch % 6) - 1 == 0 then
            -- Increase the learning rate for %20 on the next epoch after drastically decreaseing it
            print ("==> Learning rate has been increased to " .. config.learningRate * 1.2 .. " from its previous value of " .. config.learningRate)
            config.learningRate = config.learningRate * 1.2
        end

        print ("==> Total time for epoch " .. epoch .. ": " .. torch.toc(tic)/60 .. " minutes")
        print ('==> Free GPU Mem (MBs) is: ' .. ({commonFuncs.getGPUMem()})[1] .. '. Total GPU Mem (MBs) is ' .. ({commonFuncs.getGPUMem()})[2])
        print '\n'
        epoch = epoch + 1
        model:training()
    end -- END the main while loop

    if continueTraining == false then
        print ("==> Training stopped on epoch " .. epoch .. " since validation set's lower bound was not going down")
    end

else
    print "==> Could not find any training data. Make sure the training data are located in "
end -- if trainDataFiles
