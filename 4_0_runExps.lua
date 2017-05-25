#!~/torch/install/bin/th

require 'image'
require 'nn'
require '2_2_Sampler'
require 'cutorch'
require 'cunn'
require 'nngraph'
require '2_2_Sampler'
require 'gnuplot'
local commonFuncs = require '0_commonFuncs'
local sampleManifold = require '3_sampleManifold'
if pcall(require, 'cudnn') then
    require 'cudnn'
end


trainDataFiles, validationDataFiles, testDataFiles = commonFuncs.obtainDataPath(opt.benchmark, opt.testPhase, true)
local allData = {}
allData[1] = trainDataFiles
allData[2] = validationDataFiles
allData[3] = testDataFiles

local currentModelDirName = (opt.expType == 'random' or opt.expType == 'conditionalSample') and 'samples' or opt.expType == 'interpolate' and 'interpolation' or opt.expType == 'forwardPass' and 'forwardPass'
local experimentResultOutputPath = string.format('%s/experiments/%s/', paths.cwd() ..'/' .. opt.modelDirName, (opt.fromEpoch > 0 and 'epoch' .. opt.fromEpoch or ''))
local modelPath = string.format('%s/model/%s/model.t7', paths.cwd() ..'/' .. opt.modelDirName, (opt.fromEpoch > 0 and 'epoch' .. opt.fromEpoch or ''))

--Path to load the empirical distribution
local meanLogVarPath = string.format('%s/model/%s/mean_logvar.t7', paths.cwd() ..'/' .. opt.modelDirName, (opt.fromEpoch > 0 and 'epoch' .. opt.fromEpoch or ''))
if not paths.filep(meanLogVarPath) then
    meanLogVarPath = nil
end

local gMod = torch.load(modelPath) -- Load the model
gMod = gMod:cuda()
if cudnn then
    cudnn.convert(gMod, cudnn)
end
gMod:evaluate()
local model = gMod:get(1) -- Use the training model
print ''

if opt.expType == 'randomSampling' then
	print ("==> Configurations, modelDirName: " .. opt.modelDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size (BS) Change Epoch: " .. opt.batchSizeChangeEpoch .. ", BS Change: " .. opt.batchSizeChange .. ", Target BS: " .. opt.targetBatchSize .. ", Output Fea. Maps: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", KLD Grad. Coeff:" .. opt.KLD .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', DropoutNet: ' .. (opt.dropoutNet and "True" or "False") .. ', KeepVP: ' .. opt.VpToKeep .. ', silhouetteInput: ' .. (opt.silhouetteInput and "True" or "False") .. ', singleVPNet: ' .. (opt.singleVPNet and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False") .. ', From Epoch: ' .. opt.fromEpoch)
	print ('==> Generating ' .. (opt.conditional and 'conditional' or '') .. ' random samples')
	print ("==> The results will be stored at '" .. experimentResultOutputPath .. (opt.conditional and 'conditionalSamples' or 'randomSamples') .. "'")
	if opt.conditional then
		if opt.benchmark then
			data = torch.load(validationDataFiles[torch.random(1, #validationDataFiles)])
		else
			data = torch.load(allData[2][torch.random(1, #allData[2])]) -- Choose one file randomly chosen from the validation/test set
			local tempData = torch.load(allData[3][torch.random(1, #allData[3])]) -- Choose one randomly from the test set
			-- Concatenate the two data sets tensors
			data.labels = torch.cat(data.labels, tempData.labels, 1)
			data.dataset = torch.cat(data.dataset, tempData.dataset, 1)
			tempData.labels = nil
			tempData.dataset = nil
			tempData = nil
		end
	end
	local sampleZembeddings = meanLogVarPath and torch.load(meanLogVarPath) or nil -- If the mean_logvar.t7 file does not exist on 
	sampleManifold.sample(opt.manifoldExp, opt.sampleCategory, opt.canvasHW, opt.nSamples, data, model, experimentResultOutputPath, opt.mean, opt.var, opt.nLatents, opt.imgSize, opt.numVPs, opt.fromEpoch, opt.batchSize, opt.targetBatchSize, opt.testPhase, opt.tanh, opt.dropoutNet, opt.VpToKeep, opt.silhouetteInput, sampleZembeddings, opt.singleVPNet, opt.conditional, opt.expType, opt.benchmark)
	print ('==> Finshed drawing ' .. (opt.conditional and 'conditional' or '') .. ' random samples')

elseif opt.expType == 'interpolation' then
	print ("==> Configurations, modelDirName: " .. opt.modelDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size (BS) Change Epoch: " .. opt.batchSizeChangeEpoch .. ", BS Change: " .. opt.batchSizeChange .. ", Target BS: " .. opt.targetBatchSize .. ", Output Fea. Maps: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", KLD Grad. Coeff:" .. opt.KLD .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', DropoutNet: ' .. (opt.dropoutNet and "True" or "False") .. ', KeepVP: ' .. opt.VpToKeep .. ', silhouetteInput: ' .. (opt.silhouetteInput and "True" or "False") .. ', singleVPNet: ' .. (opt.singleVPNet and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False") .. ', From Epoch: ' .. opt.fromEpoch)
	print ("==> Running the interpolation experiment")
	print ("==> The results will be stored at '" .. experimentResultOutputPath .. "'")
	if opt.benchmark then
		data = torch.load(validationDataFiles[1])
	else
		data = torch.load(allData[2][torch.random(1, #allData[2])]) -- Choose one randomly from the validation set
		local tempData = torch.load(allData[3][torch.random(1, #allData[3])]) -- Choose one randomly from the test set
		-- Concatenate the two data sets tensors
		data.labels = torch.cat(data.labels, tempData.labels, 1)
		data.dataset = torch.cat(data.dataset, tempData.dataset, 1)
		tempData.labels = nil
		tempData.dataset = nil
		tempData = nil
	end
	sampleManifold.sample(opt.manifoldExp, opt.sampleCategory, opt.canvasHW, opt.nSamples, data, model, experimentResultOutputPath, opt.mean, opt.var, opt.nLatents, opt.imgSize, opt.numVPs, opt.fromEpoch, opt.batchSize, opt.targetBatchSize, opt.testPhase, opt.tanh, opt.dropoutNet, opt.VpToKeep, opt.silhouetteInput, sampleZembeddings, opt.singleVPNet, opt.conditional, opt.expType, opt.benchmark)
	print ('==> Finshed running doing interpolation ')

elseif opt.expType == 'forwardPass' then
	print ("==> Configurations, modelDirName: " .. opt.modelDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size (BS) Change Epoch: " .. opt.batchSizeChangeEpoch .. ", BS Change: " .. opt.batchSizeChange .. ", Target BS: " .. opt.targetBatchSize .. ", Output Fea. Maps: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", KLD Grad. Coeff:" .. opt.KLD .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', DropoutNet: ' .. (opt.dropoutNet and "True" or "False") .. ', KeepVP: ' .. opt.VpToKeep .. ', silhouetteInput: ' .. (opt.silhouetteInput and "True" or "False") .. ', singleVPNet: ' .. (opt.singleVPNet and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False") .. ', From Epoch: ' .. opt.fromEpoch)
	print ("==> Doing forward pass for the '" .. opt.forwardPassType .. "' experiment.")

	if opt.forwardPassType == 'userData' then
		if not paths.dirp('ExtraData/userData') then
			print ('==> Please first copy your images (single view depth maps or silhouettes placed, roughly, in the middle an image of size 224 x 224) to ExtraData/userData')
			os.exit()
		end
		print ("==> Doing reconstruction for the silhouettes/depth maps  of user's choice")
		experimentResultOutputPath = experimentResultOutputPath .. 'userData'
		print ("==> The results will be stored at '" .. experimentResultOutputPath .. "'")
		local dataTensor = commonFuncs.loadExtraData('ExtraData/userData', opt.forwardPassType, opt.numVPs, opt.silhouetteInput)
		for i=1, dataTensor:size(1) do
			image.save(string.format('test%d.png', i), dataTensor[i])
		end
		for i=1, dataTensor:size(1) do
			dataTensor = dataTensor:cuda()
			local tempTensor = torch.cat(dataTensor[{{i}}], dataTensor[{{i}}], 1)
			if opt.conditional then
				-- Use the predicted classes to do the reconstruction
				local mean, log_var, predictedClassScores = unpack(model:get(2):forward(tempTensor))
                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, predictedClassScores:size(2))
                recon = model:get(4+(opt.conditional and 1 or 0)):forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
			else
				recon = unpack(model:forward(tempTensor))
			end
			paths.mkdir(experimentResultOutputPath .. '/model' .. i .. '-userData/mask')
			image.save(experimentResultOutputPath .. '/model' .. i .. '-userData/x-originalInputImage.png', tempTensor[1][1])
			for k=1, recon[1]:size(2) do
				image.save(experimentResultOutputPath .. '/model' .. i .. '-userData/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
				image.save(experimentResultOutputPath .. '/model' .. i .. '-userData/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])
			end
		end
		print ("==> Finished doing forwardPass for user's images")
	elseif opt.forwardPassType == 'nyud' then
		if not opt.singleVPNet then
			print '==> You cannot do the NYUD experiment with DropoutNet or AllVPNet.'
			os.exit()
		end
		print('==> Doing forward pass on the NYUD data set')
		print("==> The results will be stored at '" .. experimentResultOutputPath .. 'nyud' .. "'")
		experimentResultOutputPath = {experimentResultOutputPath, experimentResultOutputPath}
		experimentResultOutputPath[1] = experimentResultOutputPath[1] .. '/nyud/chair'
		experimentResultOutputPath[2] = experimentResultOutputPath[2] .. '/nyud/bottle'
		local dataPaths = {'ExtraData/nyud/chair', 'ExtraData/nyud/bottle'}
		local dirText = {'chair', 'bottle'}
		for j=1, #experimentResultOutputPath do
			local originalDataTensorTable = commonFuncs.loadExtraData(dataPaths[j], opt.forwardPassType, opt.numVPs)
			if opt.silhouetteInput then
				local dataTensor = originalDataTensorTable[2]
				for i=1, originalDataTensorTable[1]:size(1) do
					dataTensor = dataTensor:cuda()
					local inputTensor = torch.cat(dataTensor[{{i}}], dataTensor[{{i}}], 1)
					if opt.conditional then
						-- Use the predicted classes to do the reconstruction
						local mean, log_var, predictedClassScores = unpack(model:get(2):forward(inputTensor))
		                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true,  predictedClassScores:size(2))
		                recon = model:get(4+(opt.conditional and 1 or 0)):forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
					else
						recon = unpack(model:forward(inputTensor))
					end
					paths.mkdir(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask')
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/x-originalSilhouette-Input.png', inputTensor[1][1])
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/x-originalDepth.png', originalDataTensorTable[1][i][1])
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/x-originalRGB.png', originalDataTensorTable[3][i])
					for k=1, recon[1]:size(2) do
						image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
						image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])
					end
				end
			else
				local depthTensor = originalDataTensorTable[1]
				for i=1, originalDataTensorTable[1]:size(1) do
					depthTensor = depthTensor:cuda()
					local inputTensor = torch.cat(depthTensor[{{i}}], depthTensor[{{i}}], 1)
					if opt.conditional then
						-- Use the predicted classes to do the reconstruction
						local mean, log_var, predictedClassScores = unpack(model:get(2):forward(inputTensor))
		                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true,  opt.benchmark and 40 or 54)
		                recon = model:get(4+(opt.conditional and 1 or 0)):forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
					else
						recon = unpack(model:forward(inputTensor))
					end
					paths.mkdir(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask')
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/x-originalDepth-Input.png', inputTensor[1][1])
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/x-originalSilhouette.png', originalDataTensorTable[2][i][1])
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/x-originalRGB.png', originalDataTensorTable[3][i])
					for k=1, recon[1]:size(2) do
						image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
						image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])
					end
				end
			end
		end
		print ("==> Finished doing forward pass for the NYUD data set")

	elseif opt.forwardPassType == 'randomReconstruction' then
		print ('==> Reconstructing randomly-chosen samples from the test/validation set')
		experimentResultOutputPath = experimentResultOutputPath .. 'reconstruction'
		print("==> The results will be stored at '" .. experimentResultOutputPath)
		if not opt.benchmark then
			data = torch.load(allData[2][torch.random(1, #allData[2])]) -- Choose one randomly from the validation/test set
			local tempData = torch.load(allData[3][torch.random(1, #allData[3])]) -- Choose one randomly from the test set
			-- Concatenate the two data sets tensors
			data.labels = torch.cat(data.labels, tempData.labels, 1)
			data.dataset = torch.cat(data.dataset, tempData.dataset, 1)
			tempData.labels = nil
			tempData.dataset = nil
			tempData = nil
		else
			data = torch.load(allData[2])
		end

		local indicesToChoose = torch.randperm(data.dataset:size(1))
		indicesToChoose = indicesToChoose[{{1, opt.nReconstructions}}]:long() -- Do not use very large opt.nReconstructions
		local depthMaps = data.dataset:index(1, indicesToChoose)
		local labels = data.labels:index(1, indicesToChoose)
		depthMaps = depthMaps:cuda()
		local silhouettes = depthMaps:clone()
		if opt.tanh then
            silhouettes[silhouettes:gt(-1)] = 1
            silhouettes[silhouettes:eq(-1)] = 0
        else
            silhouettes[silhouettes:gt(0)] = 1
        end

		for i=1, opt.nReconstructions do

			local numOfVPsToDrop = torch.zeros(1) -- A placeholder to hold the number of view points to be dropped for the current category
	        local dropIndices = torch.zeros(opt.numVPs) -- A placeholder to hold the indices of the tensor to be zeroed-out  -- Used for dropoutNet
	        local pickedVPs = torch.Tensor(2) -- A placeholder to hold the view point to be kept -- Used for singleVPNet
	        if opt.VpToKeep >= opt.numVPs then
	            pickedVPs[1] = torch.random(1, opt.numVPs)
	            pickedVPs[2] = pickedVPs[1]
	        else
	        	pickedVPs[1] = opt.VpToKeep
	            pickedVPs[2] = opt.VpToKeep
	        end
	        local tempDepthImg = torch.cat(depthMaps[{{i}}], depthMaps[{{i}}], 1) -- This resolve the batch normalization issue
	        local tempSilImg = torch.cat(silhouettes[{{i}}], silhouettes[{{i}}], 1) -- This resolve the batch normalization issue
	        droppedInputs = commonFuncs.dropInputVPs({tempDepthImg, tempSilImg}, true, opt.dropoutNet, numOfVPsToDrop, dropIndices, opt.singleVPNet, pickedVPs)

			local cat = data.category[labels[i]]
			local networkInput = {}
			for j=1, 2 do -- j == 1 for depth maps and j == 2 for silhouettes
				local temp = droppedInputs[j][{{1}}]:clone()
				temp = torch.cat(temp, temp, 1)
				networkInput[j] = temp
			end

			if opt.conditional then
				-- Use the predicted classes to do the reconstruction
				local mean, log_var, predictedClassScores = unpack(model:get(2):forward(opt.silhouetteInput and networkInput[2] or networkInput[1]))
                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, predictedClassScores:size(2))
                recon = model:get(4+(opt.conditional and 1 or 0)):forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
			else
				recon = unpack(model:forward(opt.silhouetteInput and networkInput[2] or networkInput[1]))
			end

			local reconPath = experimentResultOutputPath .. '/test/' .. '/' .. cat .. '/model' .. i .. (opt.dropoutNet and 'VPs' .. (opt.numVPs - numOfVPsToDrop[1]) or opt.singleVPNet and 'VP' .. pickedVPs[1] or '')  .. '-' .. cat
			paths.mkdir(reconPath .. '/mask')
			for k=1, recon[1]:size(2) do
				image.save(reconPath ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
				image.save(reconPath ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])

				-- Save the original depth maps and silhouettes (marked on their top-right corner)
				image.save(reconPath ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', tempDepthImg[1][k])
				image.save(reconPath ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', tempSilImg[1][k])
			end

		end

	elseif opt.forwardPassType == 'reconstructAllSamples' then
		cr1 = nn.AbsCriterion()
		cr1.sizeAverage = false
		cr2 = nn.AbsCriterion()
		cr2.sizeAverage = false
		criterion = nn.ParallelCriterion():add(cr1):add(cr2)
		criterion = criterion:cuda()


		local tic = torch.tic()
		local flag = false
		print('==> Reconstructing all samples in training and validation/test sets' .. (opt.singleVPNet and opt.allViewsExp and ' for all ' .. opt.numVPs .. ' views' or ''))
		experimentResultOutputPath = experimentResultOutputPath .. (opt.singleVPNet and opt.allViewsExp and 'AllSamplesReconstruction-Views' or 'AllSamplesReconstruction')
		print("==> The results will be stored at '" .. experimentResultOutputPath)
		for t=2, 2 do -- t == 1 does the reconstruction for the training set and t == 2 for the test set
			for l=1, opt.singleVPNet and opt.allViewsExp and opt.numVPs or 1 do
				local silErr = 0
		        local depthMapErr = 0
		        local totalError = 0
				local classAccuracy = 0
				local numSamples = 0

				local counter = 1
				print ('==> Doing reconstruction for the ' .. (t==1 and 'training' or 'test') .. ' set' .. (opt.singleVPNet and opt.allViewsExp and (' for view ' .. l) or '')) 
				if t == 1 then
					dataFilePaths = allData[1]
				elseif flag == false then
					flag = true
					if opt.benchmark then
						dataFilePaths = allData[2]
					else
						dataFilePaths = commonFuncs.tableConcat(allData[2], allData[3])
					end
				end

				for i=1, #dataFilePaths do
					print ('  ==> Storing the reconstruction results for file ' .. i .. '/' .. #dataFilePaths)
					data = torch.load(dataFilePaths[i])
					numSamples = numSamples + data.dataset:size(1)

					for j=1, data.dataset:size(1) do
						-- Create hot vectors for training conditional models
			            if opt.conditional then
			                targetClassIndices = data.labels[{{j}}]
			                targetClassIndices = torch.cat(targetClassIndices, targetClassIndices, 1):cuda()
			            end

			            local numOfVPsToDrop = torch.zeros(1) -- A placeholder to hold the number of view points to be dropped for the current category
		                local dropIndices = torch.zeros(opt.numVPs) -- A placeholder to hold the indices of the tensor to be zeroed-out  -- Used for dropoutNet
		                local pickedVPs = torch.Tensor(2) -- A placeholder to hold the view point to be kept -- Used for singleVPNet
		                if not opt.allViewsExp and opt.VpToKeep >= numVPs then
		                    pickedVPs[1] = torch.random(1, opt.numVPs)
		                    pickedVPs[2] = pickedVPs[1]
		                else
		                    pickedVPs[1] = opt.singleVPNet and opt.allViewsExp and l or opt.VpToKeep
		                    pickedVPs[2] = opt.singleVPNet and opt.allViewsExp and l or opt.VpToKeep
		                end

						local depthMaps = data.dataset[{{j}}]:cuda()
						local catLabel = data.category[data.labels[j]]
						depthMaps = torch.cat(depthMaps, depthMaps, 1)
						local silhouettes = depthMaps:clone()
						if opt.tanh then
				            silhouettes[silhouettes:gt(-1)] = 1
				            silhouettes[silhouettes:eq(-1)] = 0
				        else
				            silhouettes[silhouettes:gt(0)] = 1
				        end
						local droppedInputs = commonFuncs.dropInputVPs({depthMaps, silhouettes}, true, opt.dropoutNet, numOfVPsToDrop, dropIndices, opt.singleVPNet, pickedVPs)
						local tempTensor = opt.silhouetteInput and droppedInputs[2] or droppedInputs[1]
						local tempNoisyInput = tempTensor[tempTensor:gt(0)]
                		tempTensor[tempTensor:gt(0)] = tempNoisyInput:add(torch.rand(tempNoisyInput:size()):div(100):cuda())

						if opt.conditional then
							-- Use the predicted classes to do the reconstruction
							local mean, log_var, predictedClassScores = unpack(model:get(2):forward(opt.silhouetteInput and droppedInputs[2] or droppedInputs[1]))
			                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, predictedClassScores:size(2))
			                recon = model:get(4+(opt.conditional and 1 or 0)):forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
		                    classAccuracy = classAccuracy + commonFuncs.computeClassificationAccuracy(predictedClassScores, targetClassIndices)
						else
							recon = unpack(model:forward(opt.silhouetteInput and droppedInputs[2] or droppedInputs[1]))
						end
						local originalDepth = data.dataset[{{j}}]:cuda()
	                	originalDepth = torch.cat(originalDepth, originalDepth, 1)
	                	local originalSil = originalDepth:clone()
	                	if opt.tanh then
				            originalSil[originalSil:gt(-1)] = 1
				            originalSil[originalSil:eq(-1)] = 0
				        else
				            originalSil[originalSil:gt(0)] = 1
				        end
						criterion:forward(recon, {originalDepth, originalSil})
	                	silErr = silErr + criterion.criterions[2].output
			            depthMapErr = depthMapErr + criterion.criterions[1].output
			            totalError = totalError + criterion.output
	                    originalDepth = nil
	                    originalSil = nil
						if opt.maxNumOfRecons > 0 and counter <= opt.maxNumOfRecons or opt.maxNumOfRecons == 0 then
							-- Will reconstruct all of the samples if opt.maxNumOfRecons is set to 0
							local reconPath = experimentResultOutputPath .. (opt.singleVPNet and opt.allViewsExp and '/view' .. l or '') .. (t == 1 and '/train' or '/test') .. catLabel .. '/model' .. counter .. (opt.dropoutNet and 'VPs' .. (opt.numVPs - numOfVPsToDrop[1]) or opt.singleVPNet and not opt.allViewsExp and 'VP' .. pickedVPs[1] or '') .. '-' .. catLabel
							paths.mkdir(reconPath .. '/mask')
							for k=1, recon[1]:size(2) do
								image.save(reconPath .. '/file1-img-' .. k-1 .. '-rec.png', recon[1][1][k])
								image.save(reconPath .. '/mask/file1-img-' .. k-1 .. '-rec.png', recon[2][1][k])

								-- Save the original depth maps and silhouettes (with a mark on their top-right corner for the ones not fed to the model)
								image.save(reconPath .. '/file1-img-' .. k-1 .. '-or.png', depthMaps[1][k])
								image.save(reconPath .. '/mask/file1-img-' .. k-1 .. '-or.png', silhouettes[1][k])
							end
						end
						counter = counter + 1
						if i == 4 then collectgarbage() end
					end
					data = nil
					collectgarbage()
				end
				totalError = totalError/numSamples/(opt.imgSize^2*opt.numVPs)/2
			    silErr = silErr/numSamples/(opt.imgSize^2*opt.numVPs)/2
			    depthMapErr = depthMapErr/numSamples/(opt.imgSize^2*opt.numVPs)/2
				if opt.conditional then
					-- The division by 2 is due to feeding the same input twice in the same tensor to the model
				    classAccuracy = classAccuracy/numSamples/2
				    if opt.singleVPNet and opt.allViewsExp then
				    	print (string.format("  ==> Statistics for view %d: Classification Accuracy: %.3f, Depth Err: %.4f, Sil. Err: %.4f, Total Err: %.4f", l, classAccuracy, depthMapErr, silErr, totalError))
				    else
				    	print (string.format("  ==> Statistics: Classification Accuracy: %.3f, Depth Err: %.4f, Sil. Err: %.4f, Total Err: %.4f", classAccuracy, depthMapErr, silErr, totalError))
				    end
				else
					if opt.singleVPNet and opt.allViewsExp then
				    	print (string.format("  ==> Statistics for view %d: Depth Err: %.4f, Sil. Err: %.4f, Total Err: %.4f", l, depthMapErr, silErr, totalError))
				    else
				    	print (string.format("  ==> Statistics: Depth Err: %.4f, Sil. Err: %.4f, Total Err: %.4f", depthMapErr, silErr, totalError))
				    end
				end
			end
		end
		print ("==> Total time for doing the reconstruction" .. (opt.singleVPNet and opt.allViewsExp and (' for all ' .. opt.numVPs .. ' views') or '') ..  ": " .. torch.toc(tic)/60 .. " minutes")
	end
elseif opt.expType == 'NNs' then

	--[[
	Before running the nearest neighbor experiment make sure you have have copied your sample sets to 'experiments/epochX/conditionalSamples' chosen your desired samples from each sample set
	by creating a viz.txt file and writing the row and column numbers of your desired samples in it.

	For instance, if you set opt.canvasHW to 5 then each viewpoint canvas will show 25 images. In the vix.txt file you can 
	type the followings:
	1, 2
	1, 4
	3, 2
	1, 5
	5, 5
    4, 5
    The first and second numbers represent the row and column numbers, respectively, on the canvases for the chosen samples
	--]]

	print ("==> Doing the nearest neighbors experiment")
	print ("==> Configurations, modelDirName: " .. opt.modelDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size (BS) Change Epoch: " .. opt.batchSizeChangeEpoch .. ", BS Change: " .. opt.batchSizeChange .. ", Target BS: " .. opt.targetBatchSize .. ", Output Fea. Maps: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", KLD Grad. Coeff:" .. opt.KLD .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', DropoutNet: ' .. (opt.dropoutNet and "True" or "False") .. ', KeepVP: ' .. opt.VpToKeep .. ', silhouetteInput: ' .. (opt.silhouetteInput and "True" or "False") .. ', singleVPNet: ' .. (opt.singleVPNet and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False") .. ', From Epoch: ' .. opt.fromEpoch)
	local samplesPath = experimentResultOutputPath .. (opt.conditional and 'conditionalSamples' or 'randomSamples') -- The path from which the samples will be read
	experimentResultOutputPath = experimentResultOutputPath .. 'nearestNeighbors' .. (opt.conditional and '/conditionalSamples' or '/randomSamples')
	print ('==> Running nearest neighbors experiment. Please be patient. This experiment might take a long time especially if your training set is large.')
	print('==> The results will be stored at ' .. "'" .. experimentResultOutputPath .. "'")


	-- Get the Zs for the training data set
	local counter = 1
	local Zs = {}
	local labels = {}
	local sampler = nn.Sampler()
	local dropIndices = torch.zeros(opt.numVPs) -- A placeholder to hold the indices of the tensor to be zeroed-out  -- Used for dropoutNet
    local pickedVPs = torch.Tensor(2) -- A placeholder to hold the view point to be kept -- Used for singleVPNet
    if opt.singleVPNet then
    	-- Fix on view 12
        pickedVPs[1] = 12
        pickedVPs[2] = 12
    end
	sampler = sampler:cuda()
	for i=1, #trainDataFiles do
		data = torch.load(trainDataFiles[i])

		local localZs = torch.zeros(data.dataset:size(1), opt.nLatents*2):cuda()
		local localLabels = torch.zeros(data.dataset:size(1))
		for j=1, data.dataset:size(1) do

			local depthMaps = data.dataset[{{j}}]:cuda()
			local catLabel = data.category[data.labels[j]]
			depthMaps = torch.cat(depthMaps, depthMaps, 1)
			local silhouettes = depthMaps:clone()
			if opt.tanh then
	            silhouettes[silhouettes:gt(-1)] = 1
	            silhouettes[silhouettes:eq(-1)] = 0
	        else
	            silhouettes[silhouettes:gt(0)] = 1
	        end
	        local droppedInputs = commonFuncs.dropInputVPs(not opt.silhouetteInput and depthMaps or silhouettes, false, opt.dropoutNet, nil, dropIndices, opt.singleVPNet, pickedVPs)
			local tempTensor = droppedInputs
			local tempNoisyInput = tempTensor[tempTensor:gt(0)]
			tempTensor[tempTensor:gt(0)] = tempNoisyInput:add(torch.rand(tempNoisyInput:size()):div(100):cuda())

			local encodings = commonFuncs.getEncodings(droppedInputs, model:get(2), sampler, opt.conditional) -- model:get(2) points to the encoder
			localZs[j]:copy(encodings[1])
			localLabels[j] = data.labels[j]
		end
		Zs[i] = localZs:float()
		labels[i] = localLabels

	end

	-- Get the Zs for the samples
	local samplesMainDirs = commonFuncs.getFileNames(samplesPath, nil, false)
	for c =1, opt.conditional and #samplesMainDirs or 1 do -- Go over each category, if conditional
		local catName = opt.conditional and commonFuncs.splitTxt(samplesMainDirs[c], '/') or ''
		catName = opt.conditional and catName[#catName]
		local catDirs = commonFuncs.getFileNames(samplesMainDirs[c], nil, false)
		samplesDirs = opt.conditional and catDirs or samplesMainDirs
		local numOfSamplestoVisualize = commonFuncs.getNumOfSamplesToViz(samplesDirs)
		local depthMapsTensor = torch.zeros(numOfSamplestoVisualize, opt.numVPs, opt.imgSize, opt.imgSize)
		local silhouetteSTensor = torch.zeros(numOfSamplestoVisualize, opt.numVPs, opt.imgSize, opt.imgSize)
		local sampleCounter = 1
		for i=1, #samplesDirs do
			local samplesToVisualize = commonFuncs.getFileNames(samplesDirs[i], 'viz.txt')
			if #samplesToVisualize == 1 then -- if there exists a viz.txt file
				local depthMapFilesToLoad = commonFuncs.getFileNames(samplesDirs[i], '.png')
				local silhouetteFilesToLoad = commonFuncs.getFileNames(samplesDirs[i] .. '/mask', '.png')
				f = assert(io.open(samplesToVisualize[1], 'r'))
				for line in f:lines() do
					local rowNum, colNum = commonFuncs.commaSeparatedStrToTable(line, true)
					for j=1, opt.numVPs do
						local tempDepthImg = image.load(depthMapFilesToLoad[j])[1]
						depthMapsTensor[sampleCounter][j] = tempDepthImg[{{(rowNum-1)*opt.imgSize+1, rowNum*opt.imgSize}, {(colNum-1)*opt.imgSize+1, colNum*opt.imgSize}}]
					 	local tempSilImg = image.load(silhouetteFilesToLoad[j])[1]
					 	silhouetteSTensor[sampleCounter][j] = tempSilImg[{{(rowNum-1)*opt.imgSize+1, rowNum*opt.imgSize}, {(colNum-1)*opt.imgSize+1, colNum*opt.imgSize}}]
					end
					sampleCounter = sampleCounter + 1
				end
				f:close()
			end
		end
		local samplesZs = torch.zeros(depthMapsTensor:size(1), opt.nLatents*2):cuda()
		local sampleLabels = torch.zeros(depthMapsTensor:size(1))
		for i=1, depthMapsTensor:size(1) do
			local droppedInputs = commonFuncs.dropInputVPs({depthMapsTensor[{{i}}], silhouetteSTensor[{{i}}]}, true, opt.dropoutNet, nil, dropIndices, opt.singleVPNet, pickedVPs)
			droppedInputs = {droppedInputs[1]:cuda(), droppedInputs[2]:cuda()}
			local tempTensor = opt.silhouetteInput and droppedInputs[2] or droppedInputs[1]
			local tempNoisyInput = tempTensor[tempTensor:gt(0)]
			tempTensor[tempTensor:gt(0)] = tempNoisyInput:add(torch.rand(tempNoisyInput:size()):div(100):cuda())
			local encoding = commonFuncs.getEncodings(opt.silhouetteInput and droppedInputs[2]:cuda() or droppedInputs[1]:cuda(), model:get(2), sampler, opt.conditional)
			samplesZs[i]:copy(encoding[1])
			if opt.conditional then
				local predClassVec = commonFuncs.computeClassificationAccuracy(encoding[2], nil, true, #data.category)
				_, temp = predClassVec:max(2)
				sampleLabels[i] = temp[1][1]
			end
		end
		

		-- Compute the similarity
		samplesZs = samplesZs:float()
		local minZFileNo = {}
		local minZIndex = {}
		local groundTruthCatName = {}
		local reconDepth = torch.zeros(samplesZs:size(1), opt.numVPs, opt.imgSize, opt.imgSize)
		local reconSil = torch.zeros(samplesZs:size(1), opt.numVPs, opt.imgSize, opt.imgSize)
		for i=1, samplesZs:size(1) do

			-- Find the closest representation
			local possibleLabel
			local minDist = 1000000 -- Set the initial distance
			local minLabel
			local L1Norm
			for j=1, #Zs do
				for k=1, Zs[j]:size(1) do
					L1Norm = torch.add(Zs[j][k], torch.mul(samplesZs[i], -1)):abs():sum()
					if L1Norm < minDist then
						minZFileNo[i] = j
						minZIndex[i] = k
						minDist = L1Norm
						minLabel = labels[j][k]
						groundTruthCatName[i] = data.category[labels[j][k]]
					end
				end
			end

			-- Store the sample's depth maps and silhouettes on disk
			local localCounter = 0
			local randomSamplePath = experimentResultOutputPath .. '/' .. (not opt.conditional and catName or groundTruthCatName[i]) .. '/' .. i .. '/sample' .. (opt.conditional and '-PredictedClass-' .. data.category[sampleLabels[i]] or '') .. '/'
			paths.mkdir(randomSamplePath .. 'mask')
			for k=1, depthMapsTensor:size(2) do
				image.save(randomSamplePath .. 'file-' .. counter .. localCounter .. '-img-' .. k-1 .. '-rec.png', depthMapsTensor[i][k])
				image.save(randomSamplePath .. '/mask/file-' .. counter .. localCounter.. '-img-' .. k-1 .. '-rec.png', silhouetteSTensor[i][k])
			end


			-- Obtain the reconstruction of the nearest-neighbor training sample
			local mean_log_var = Zs[minZFileNo[i]][minZIndex[i]]:view(2, opt.nLatents):clone()
			local mean = mean_log_var[{{1}}]:cuda()
			local log_var = mean_log_var[{{2}}]:cuda()
			local sampleVec = nn.Sampler():cuda():forward({mean, log_var})
			sampleVec = torch.cat(sampleVec, sampleVec, 1)
			if opt.conditional then
				-- Create the ground-truth target vectors
				targetClassHotVec = torch.CudaTensor(2, #data.category):fill(0)
                targetClassHotVec[1][labels[minZFileNo[i]][minZIndex[i]]] = 1
                targetClassHotVec[2][labels[minZFileNo[i]][minZIndex[i]]] = 1
                recon = model:get(4+(opt.conditional and 1 or 0)):forward({sampleVec, targetClassHotVec})
			else
				recon = model:get(4):forward(sampleVec)
			end
			reconDepth[{{i}}]:copy(recon[1][{{1}}])
			reconSil[{{i}}]:copy(recon[2][{{1}}])
			localCounter = localCounter + 1
			minLabel = nil
		end
		data = nil
		collectgarbage()

		local lastFileID = minZFileNo[1]
		local localCounter = 1
		data = torch.load(trainDataFiles[minZFileNo[1]])
		for i=1, #minZFileNo do
			local nearestPath = experimentResultOutputPath .. '/' .. (not opt.conditional and catName or groundTruthCatName[i]) .. '/' .. i .. '/nearest' .. (opt.conditional and '-' .. groundTruthCatName[i] or '') 
			paths.mkdir(nearestPath .. '/mask')
			paths.mkdir(nearestPath .. '/nearestRecon/mask') -- To store the reconstruction of the nearest neighbor
			if lastFileID ~= minZFileNo[i] then
				-- This process is very time-consuming and inefficiently implemented especially if there are many training files on disk
				-- Make it optimized!
				lastFileID = minZFileNo[i]
				data = nil
				collectgarbage()
				data = torch.load(trainDataFiles[minZFileNo[i]])
			end
			local depths = data.dataset[{{minZIndex[i]}}]:clone()
			local silhouettes = depths:clone()
			if opt.tanh then
	            silhouettes[silhouettes:gt(-1)] = 1
	            silhouettes[silhouettes:eq(-1)] = 0
	        else
	        	silhouettes[silhouettes:gt(0)] = 1
	        end
	        commonFuncs.dropInputVPs({depths, silhouettes}, true, opt.dropoutNet, nil, dropIndices, opt.singleVPNet, pickedVPs)
			for k=1, depthMapsTensor:size(2) do
				image.save(nearestPath .. '/file' .. counter .. localCounter .. '-img-' .. k-1 .. '-or.png', depths[1][k])
				image.save(nearestPath .. '/mask/file' .. counter ..localCounter .. '-img-' .. k-1 .. '-or.png', silhouettes[1][k])

				-- Store the reconstruction
				image.save(nearestPath .. '/nearestRecon/file' .. counter .. localCounter .. '-img-' .. k-1 .. '-rec.png', reconDepth[i][k])
				image.save(nearestPath .. '/nearestRecon/mask/file' .. counter ..localCounter .. '-img-' .. k-1 .. '-rec.png', reconSil[i][k])
			end
			localCounter = localCounter + 1
		end
		counter = counter + 1
		print ("==> Done storing results for " .. (opt.conditional and 'chosen samples from' .. catName .. ' category' or 'random samples'))
	end
	print '==> Nearest neighbor experiment is done'
	data = nil
	collectgarbage()
elseif opt.expType == 'tSNE' then
	print ("==> Configurations, modelDirName: " .. opt.modelDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size (BS) Change Epoch: " .. opt.batchSizeChangeEpoch .. ", BS Change: " .. opt.batchSizeChange .. ", Target BS: " .. opt.targetBatchSize .. ", Output Fea. Maps: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", KLD Grad. Coeff:" .. opt.KLD .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', DropoutNet: ' .. (opt.dropoutNet and "True" or "False") .. ', KeepVP: ' .. opt.VpToKeep .. ', silhouetteInput: ' .. (opt.silhouetteInput and "True" or "False") .. ', singleVPNet: ' .. (opt.singleVPNet and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False") .. ', From Epoch: ' .. opt.fromEpoch)
	experimentResultOutputPath = experimentResultOutputPath .. 'tSNE'
	print ('==> Running tSNE experiment for validation/test samples. The results will be stored in ' .. experimentResultOutputPath)

	local allZs, allLabels, data
	if not paths.filep(experimentResultOutputPath .. '/allZs.t7') then
		paths.mkdir(experimentResultOutputPath)
		local Zs = {}
		local labels = {}
		local sampler = nn.Sampler()
		-- local numOfVPsToDrop = torch.zeros(1) -- A placeholder to hold the number of view points to be dropped for the current category
  --       local dropIndices = torch.zeros(opt.numVPs) -- A placeholder to hold the indices of the tensor to be zeroed-out  -- Used for dropoutNet
  --       local pickedVPs = torch.Tensor(2) -- A placeholder to hold the view point to be kept -- Used for singleVPNet
		-- -- Fix on view 12
  --       pickedVPs[1] = 12
  --       pickedVPs[2] = 12
		sampler = sampler:cuda()
		for i=1, #trainDataFiles do
			data = torch.load(trainDataFiles[i])

			local localZs = torch.zeros(data.dataset:size(1), opt.nLatents * 2):cuda()
			local localLabels = torch.zeros(data.dataset:size(1))
			for j=1, data.dataset:size(1) do

				local depthMaps = data.dataset[{{j}}]:cuda()
				depthMaps = torch.cat(depthMaps, depthMaps, 1)
				local silhouettes = depthMaps:clone()
				if opt.tanh then
		            silhouettes[silhouettes:gt(-1)] = 1
		            silhouettes[silhouettes:eq(-1)] = 0
		        else
		            silhouettes[silhouettes:gt(0)] = 1
		        end
		        local droppedInputs = commonFuncs.dropInputVPs(not opt.silhouetteInput and depthMaps or silhouettes, false, opt.dropoutNet, numOfVPsToDrop, dropIndices, opt.singleVPNet, pickedVPs)
		        local encoding = commonFuncs.getEncodings(droppedInputs, model:get(2), sampler, opt.silhouettes, opt.silhouetteInput)
 				localZs[j]:copy(encoding[1])
				localLabels[j] = data.labels[j]
			end
			Zs[i] = localZs:float()
			labels[i] = localLabels
		end

		allZs = torch.Tensor(1, opt.nLatents * 2)
		allLabels = torch.Tensor(1)
		for i=1, #Zs do
			Zs[i] = Zs[i]:float()
			labels[i] = labels[i]:float()
			allZs = torch.cat(allZs, Zs[i], 1)
			allLabels = torch.cat(allLabels, labels[i], 1)
		end
		allZs = allZs[{{2, allZs:size(1)}}]
		allLabels = allLabels[{{2, allLabels:size(1)}}]
		torch.save(experimentResultOutputPath .. '/allZs.t7', allZs)
		torch.save(experimentResultOutputPath .. '/allLabels.t7', allLabels)
		print ('==> Embedding have been saved on disk. Running tSNE now')
	else
		print ("==> The embeddings are already stored on disk in '" .. experimentResultOutputPath .. "'. Running tSNE now")
		allZs = torch.load(experimentResultOutputPath .. '/allZs.t7')
		allLabels = torch.load(experimentResultOutputPath .. '/allLabels.t7')
		data = torch.load(trainDataFiles[1])
	end
	tsne = require 'tsne'
	allZs = allZs:double()
	local y = tsne(allZs, 2, 1900, 6500, 0.15, 20)
	commonFuncs.show_scatter_plot('tSNE-Plot', y, allLabels, #data.category, data.category, experimentResultOutputPath)

	print '==> Finished running tSNE experiment'
end