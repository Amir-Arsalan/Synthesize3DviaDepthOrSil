#!~/torch/install/bin/th

require 'image'
require 'nn'
require '2_2_Sampler'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'cudnn'
require '2_2_Sampler'
require 'gnuplot'
local commonFuncs = require '0_commonFuncs'
local sampleManifold = require '3_sampleManifold'




trainDataFiles, validationDataFiles, testDataFiles = commonFuncs.obtainDataPath(opt.benchmark, opt.testPhase, true)
local allData = {}
allData[1] = trainDataFiles
allData[2] = validationDataFiles
allData[3] = testDataFiles

local currentModelDirName = (opt.expType == 'random' or opt.expType == 'conditionalSample') and 'samples' or opt.expType == 'interpolate' and 'interpolation' or opt.expType == 'forwardPass' and 'forwardPass'
local experimentResultOutputPath = string.format('%s/experiments/epoch%d/', paths.cwd() ..'/' .. opt.modelDirName, opt.fromEpoch)
modelPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/model.t7', paths.cwd() ..'/' ..opt.modelDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, opt.fromEpoch)

--Path to load the empirical distribution
meanLogVarPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/mean_logvar.t7', paths.cwd() ..'/' ..opt.modelDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, opt.fromEpoch)
if not paths.filep(meanLogVarPath) then
    meanLogVarPath = nil
end

local gMod = torch.load(modelPath) -- Load the model
cudnn.convert(gMod, nn)
gMod = gMod:cuda()
cudnn.convert(gMod, cudnn)
gMod:evaluate()
local model = gMod:get(1)
print ''

if opt.expType == 'randomSampling' then
	print ("==> Configurations, modelDirName: " .. opt.modelDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size (BS) Change Epoch: " .. opt.batchSizeChangeEpoch .. ", BS Change: " .. opt.batchSizeChange .. ", Target BS: " .. opt.targetBatchSize .. ", Output Fea. Maps: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", KLD Grad. Coeff:" .. opt.KLD .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', DropoutNet: ' .. (opt.dropoutNet and "True" or "False") .. ', KeepVP: ' .. opt.VpToKeep .. ', silhouetteInput: ' .. (opt.silhouetteInput and "True" or "False") .. ', singleVPNet: ' .. (opt.singleVPNet and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False") .. ', From Epoch: ' .. opt.fromEpoch)
	print ('==> Generating ' .. (opt.conditional and 'conditional' or '') .. ' random samples')
	print ("==> The results will be stored at '" .. experimentResultOutputPath .. (conditional and '/conditionalSamples' or '/randomSamples') .. "'")
	if opt.conditional then
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
	end
	local sampleZembeddings = meanLogVarPath and torch.load(meanLogVarPath) or nil
	sampleManifold.sample(opt.sampleType, opt.sampleCategory, opt.canvasHW, opt.nSamples, data, model, experimentResultOutputPath, opt.mean, opt.var, opt.nLatents, opt.imgSize, opt.numVPs, opt.fromEpoch, opt.batchSize, opt.targetBatchSize, opt.testPhase, opt.tanh, opt.dropoutNet, opt.VpToKeep, opt.silhouetteInput, sampleZembeddings, opt.singleVPNet, opt.conditional, opt.expType, opt.benchmark)
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
	sampleManifold.sample(opt.sampleType, opt.sampleCategory, opt.canvasHW, opt.nSamples, data, model, experimentResultOutputPath, opt.mean, opt.var, opt.nLatents, opt.imgSize, opt.numVPs, opt.fromEpoch, opt.batchSize, opt.targetBatchSize, opt.testPhase, opt.tanh, opt.dropoutNet, opt.VpToKeep, opt.silhouetteInput, sampleZembeddings, opt.singleVPNet, opt.conditional, opt.expType, opt.benchmark)
	print ('==> Finshed running doing interpolation ')

elseif opt.expType == 'forwardPass' then
	print ("==> Doing forward pass for the '" .. opt.forwardPassType .. "' experiment")
	print ("==> Configurations, modelDirName: " .. opt.modelDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size (BS) Change Epoch: " .. opt.batchSizeChangeEpoch .. ", BS Change: " .. opt.batchSizeChange .. ", Target BS: " .. opt.targetBatchSize .. ", Output Fea. Maps: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", KLD Grad. Coeff:" .. opt.KLD .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', DropoutNet: ' .. (opt.dropoutNet and "True" or "False") .. ', KeepVP: ' .. opt.VpToKeep .. ', silhouetteInput: ' .. (opt.silhouetteInput and "True" or "False") .. ', singleVPNet: ' .. (opt.singleVPNet and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False") .. ', From Epoch: ' .. opt.fromEpoch)

	if opt.forwardPassType == 'userData' then
		if not paths.dirp('ExtraData/userData') then
			print ('==> Please first copy your data (single view depth maps or silhouettes) to ExtraData/userData')
			os.exit()
		end
		print ("==> Doing reconstruction for the silhouettes/depth maps  of user's choice")
		experimentResultOutputPath = experimentResultOutputPath .. 'userData'
		print ("==> The results will be stored at '" .. experimentResultOutputPath .. "'")
		local dataTensor = commonFuncs.loadExtraData('ExtraData/userData', opt.forwardPassType, opt.numVPs)
		for i=1, dataTensor:size(1) do
			dataTensor = dataTensor:cuda()
			local tempTensor = torch.cat(dataTensor[{{i}}], dataTensor[{{i}}], 1)
			if opt.conditional then
				-- Use the predicted classes to do the reconstruction
				local mean, log_var, predictedClassScores = unpack(model:get(2):forward(tempTensor))
                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, opt.benchmark and 40 or 54)
                recon = model:get(4+(opt.conditional and 1 or 0)):forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
			else
				recon = unpack(model:forward(tempTensor))
			end
			paths.mkdir(experimentResultOutputPath .. '/model' .. i .. '-userData/mask')
			image.save(experimentResultOutputPath .. '/model' .. i .. '-userData/100-originalInputImage.png', tempTensor[1][1])
			for k=1, recon[1]:size(2) do
				image.save(experimentResultOutputPath .. '/model' .. i .. '-userData/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
				image.save(experimentResultOutputPath .. '/model' .. i .. '-userData/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])
			end
		end
		print ("==> Finished doing forwardPass for user's data")
	elseif opt.forwardPassType == 'nyud' then
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
		                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true,  opt.benchmark and 40 or 54)
		                recon = model:get(4+(opt.conditional and 1 or 0)):forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
					else
						recon = unpack(model:forward(inputTensor))
					end
					paths.mkdir(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask')
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/100-originalMask.png', inputTensor[1][1])
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/100-originalDepth.png', originalDataTensorTable[1][i][1])
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/100-originalRGB.png', originalDataTensorTable[3][i])
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
					image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/original.png', inputTensor[1][1])
					for k=1, recon[1]:size(2) do
						image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
						image.save(experimentResultOutputPath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])
					end
				end
			end
		end
		print ("==> Finished doing forward pass forthe NYUD data set")

	elseif opt.forwardPassType == 'randomReconstruction' then
		print ('==> Reconstructing randomly-chosen samples from the test/validation set')
		experimentResultOutputPath = experimentResultOutputPath .. 'reconstruction'
		print("==> The results will be stored at '" .. experimentResultOutputPath)
		if not opt.benchmark then
			data = torch.load(allData[2][torch.random(1, #allData[2])]) -- Choose one randomly from the validation set
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
		droppedInputs = commonFuncs.dropInputVPs({depthMaps, silhouettes}, opt.dropoutNet and opt.VpToKeep or nil, true, nil, nil, opt.singleVPNet)

		for i=1, opt.nReconstructions do
			local cat = data.category[labels[i]]
			local networkInput = {}
			for j=1, 2 do
				local temp = droppedInputs[j][{{i}}]:clone()
				temp = torch.cat(temp, temp, 1)
				networkInput[j] = temp
			end

			if opt.conditional then
				-- Use the predicted classes to do the reconstruction
				local mean, log_var, predictedClassScores = unpack(model:get(2):forward(not opt.silhouetteInput and networkInput[1] or networkInput[2]))
                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, #data.category)
                recon = model:get(4+(opt.conditional and 1 or 0)):forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
			else
				recon = unpack(model:forward(not opt.silhouetteInput and networkInput[1] or networkInput[2]))
			end

			local reconPath = experimentResultOutputPath .. '/model' .. i .. '-' .. cat
			paths.mkdir(reconPath .. '/mask')
			for k=1, recon[1]:size(2) do
				image.save(reconPath ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
				image.save(reconPath ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])

				-- Save the original depth maps and silhouettes (marked on their top-right corner)
				image.save(reconPath ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', depthMaps[i][k])
				image.save(reconPath ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', silhouettes[i][k])
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
		experimentResultOutputPath = experimentResultOutputPath .. (opt.singleVPNet and opt.allViewsExp and 'AllSamplesReconstructionForAllViews' or 'AllSamplesReconstruction')
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

		                local dropIndices = torch.zeros(opt.numVPs) -- A placeholder to hold the indices of the tensor to be zeroed-out  -- Used for dropoutNet
		                local pickedVPs = torch.Tensor(2) -- A placeholder to hold the view point to be preserved -- Used for singleVPNet
		                if not opt.allViewsExp and (not expType or VpToKeep >= numVPs) then
		                    pickedVPs[1] = torch.random(1, opt.numVPs)
		                    pickedVPs[2] = pickedVPs[1]
		                else
		                    pickedVPs[1] = opt.singleVPNet and opt.allViewsExp and l or VpToKeep
		                    pickedVPs[2] = opt.singleVPNet and opt.allViewsExp and l or VpToKeep
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
						local droppedInputs = commonFuncs.dropInputVPs({depthMaps, silhouettes}, opt.dropoutNet and opt.VpToKeep or nil, true, nil, dropIndices, opt.singleVPNet, pickedVPs)
						local tempTensor = opt.silhouetteInput and droppedInputs[2] or droppedInputs[1]
						local tempNoisyInput = tempTensor[tempTensor:gt(0)]
                		tempTensor[tempTensor:gt(0)] = tempNoisyInput:add(torch.rand(tempNoisyInput:size()):div(100):cuda())

						if opt.conditional then
							-- Use the predicted classes to do the reconstruction
							local mean, log_var, predictedClassScores = unpack(model:get(2):forward(not opt.silhouetteInput and droppedInputs[1] or droppedInputs[2]))
			                local predClassVec = commonFuncs.computeClassificationAccuracy(predictedClassScores, nil, true, #data.category)
			                recon = model:get(4+(opt.conditional and 1 or 0)):forward({nn.Sampler():cuda():forward({mean, log_var}), predClassVec})
		                    classAccuracy = classAccuracy + commonFuncs.computeClassificationAccuracy(predictedClassScores, targetClassIndices)
						else
							recon = unpack(model:forward(not opt.silhouetteInput and droppedInputs[1] or droppedInputs[2]))
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
							local reconPath = experimentResultOutputPath .. (t == 1 and '/train/' or '/test/') .. (opt.singleVPNet and opt.allViewsExp and 'view' .. l .. '/' or '') .. catLabel .. '/model' .. counter .. '-' .. catLabel
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

	print ("==> Doing the nearest neighbors experiment")
	print ("==> Configurations, modelDirName: " .. opt.modelDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size (BS) Change Epoch: " .. opt.batchSizeChangeEpoch .. ", BS Change: " .. opt.batchSizeChange .. ", Target BS: " .. opt.targetBatchSize .. ", Output Fea. Maps: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", KLD Grad. Coeff:" .. opt.KLD .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', DropoutNet: ' .. (opt.dropoutNet and "True" or "False") .. ', KeepVP: ' .. opt.VpToKeep .. ', silhouetteInput: ' .. (opt.silhouetteInput and "True" or "False") .. ', singleVPNet: ' .. (opt.singleVPNet and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False") .. ', From Epoch: ' .. opt.fromEpoch)
	local samplesPath = experimentResultOutputPath .. (opt.conditional and 'conditionalSamples' or 'randomSamples') .. '/empirical'
	experimentResultOutputPath = experimentResultOutputPath .. 'nearestNeighbors' .. (opt.conditional and '/conditionalSamples' or '/randomSamples')
	print ('==> Running nearest neighbors experiment. Please be patient. This experiment might take a long time especially if your training set is large.')
	print('==> The results will be stored at ' .. "'" .. experimentResultOutputPath .. "'")

	-- Get the Zs for the training data set
	local counter = 1
	local Zs = {}
	local labels = {}
	local sampler = nn.Sampler()
	local dropIndices = torch.zeros(opt.numVPs) -- A placeholder to hold the indices of the tensor to be zeroed-out  -- Used for dropoutNet
    local pickedVPs = torch.Tensor(2) -- A placeholder to hold the view point to be preserved -- Used for singleVPNet
    if opt.singleVPNet then
    	-- Fix on view 14
        pickedVPs[1] = 14
        pickedVPs[2] = 14
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
	        local droppedInputs = commonFuncs.dropInputVPs(not opt.silhouetteInput and depthMaps or silhouettes, opt.dropoutNet and opt.VpToKeep or nil, false, nil, dropIndices, opt.singleVPNet, pickedVPs)
			local tempTensor = droppedInputs
			local tempNoisyInput = tempTensor[tempTensor:gt(0)]
			tempTensor[tempTensor:gt(0)] = tempNoisyInput:add(torch.rand(tempNoisyInput:size()):div(100):cuda())

			local encodings = commonFuncs.getEncodings(droppedInputs, model:get(2), sampler, opt.conditional)
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
		catName = opt.conditional and catName[#catName] or 'sample'
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
					 	local tempMaskImg = image.load(silhouetteFilesToLoad[j])[1]
					 	silhouetteSTensor[sampleCounter][j] = tempMaskImg[{{(rowNum-1)*opt.imgSize+1, rowNum*opt.imgSize}, {(colNum-1)*opt.imgSize+1, colNum*opt.imgSize}}]
					end
					sampleCounter = sampleCounter + 1
				end
				f:close()
			end
		end
		local samplesZs = torch.zeros(depthMapsTensor:size(1), opt.nLatents*2):cuda()
		local sampleLabels = torch.zeros(depthMapsTensor:size(1))
		for i=1, depthMapsTensor:size(1) do
			local droppedInputs = commonFuncs.dropInputVPs({depthMapsTensor[{{i}}], silhouetteSTensor[{{i}}]}, opt.dropoutNet and opt.VpToKeep or nil, true, nil, dropIndices, opt.singleVPNet, pickedVPs)
			droppedInputs = {droppedInputs[1]:cuda(), droppedInputs[2]:cuda()}
			local tempTensor = opt.silhouetteInput and droppedInputs[2] or droppedInputs[1]
			local tempNoisyInput = tempTensor[tempTensor:gt(0)]
			tempTensor[tempTensor:gt(0)] = tempNoisyInput:add(torch.rand(tempNoisyInput:size()):div(100):cuda())
			local encoding = commonFuncs.getEncodings(not opt.silhouetteInput and droppedInputs[1]:cuda() or droppedInputs[2]:cuda(), model:get(2), sampler, opt.conditional)
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
		local dataSetCatName = {}
		for i=1, samplesZs:size(1) do
			local localCounter = 0
			local randomSamplePath = experimentResultOutputPath .. '/' .. catName .. '/' .. i .. '/sample' .. (opt.conditional and '-PredictedClass-' .. data.category[sampleLabels[i]] or '') .. '/'
			paths.mkdir(randomSamplePath .. 'mask')
			for k=1, depthMapsTensor:size(2) do
				image.save(randomSamplePath .. 'file-' .. counter .. localCounter .. '-img-' .. k-1 .. '-rec.png', depthMapsTensor[i][k])
				image.save(randomSamplePath .. '/mask/file-' .. counter .. localCounter.. '-img-' .. k-1 .. '-rec.png', silhouetteSTensor[i][k])
			end

			local possibleLabel
			local minDist = 1000000
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
						dataSetCatName[i] = data.category[labels[j][k]]
					end
				end
			end
			localCounter = localCounter + 1
			minLabel = nil
		end
		data = nil
		collectgarbage()

		local lastFileID = minZFileNo[1]
		local localCounter = 1
		data = torch.load(trainDataFiles[minZFileNo[1]])
		for i=1, #minZFileNo do
			local nearestPath = experimentResultOutputPath .. '/' .. catName .. '/' .. i .. '/nearest' .. '-' .. dataSetCatName[i]
			paths.mkdir(nearestPath .. '/mask')
			if lastFileID ~= minZFileNo[i] then
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
	        commonFuncs.dropInputVPs({depths, silhouettes}, opt.dropoutNet and opt.VpToKeep or nil, true, nil, dropIndices, opt.singleVPNet, pickedVPs)
			for k=1, depthMapsTensor:size(2) do
				image.save(nearestPath .. '/file' .. counter .. localCounter .. '-img-' .. k-1 .. '-or.png', depths[1][k])
				image.save(nearestPath .. '/mask/file' .. counter ..localCounter .. '-img-' .. k-1 .. '-or.png', silhouettes[1][k])
			end
			localCounter = localCounter + 1
		end
		counter = counter + 1
		print ("==> Done storing results for " .. catName)
	end
	print '==> Nearest neighbor experiment is done'
	data = nil
	collectgarbage()
elseif opt.expType == 'tSNE' then

	experimentResultOutputPath = experimentResultOutputPath .. 'tSNE'
	local allZs, allLabels, data
	if not paths.filep(experimentResultOutputPath .. '/allZs.t7') then
		paths.mkdir(experimentResultOutputPath)
		local Zs = {}
		local labels = {}
		local sampler = nn.Sampler()
		local numDropVPs = torch.zeros(1)
		local dropIndices = torch.zeros(opt.numVPs)
		sampler = sampler:cuda()
		for i=1, #trainDataFiles do
			data = torch.load(trainDataFiles[i])

			local localZs = torch.zeros(data.dataset:size(1), opt.nLatents):cuda()
			local localLabels = torch.zeros(data.dataset:size(1))
			for j=1, data.dataset:size(1) do
				local droppedInputs = commonFuncs.dropInputVPs(data.dataset[{{j}}], opt.VpToKeep, false, numDropVPs, dropIndices, opt.singleVPNet, pickedVPs)

				local silhouettes = droppedInputs:clone()
				silhouettes[silhouettes:gt(0)] = 1
				if opt.tanh then
		            silhouettes[silhouettes:gt(-1)] = 1
		            silhouettes[silhouettes:eq(-1)] = 0
		        end

				if opt.silhouetteInput then
					droppedInputs = nil
					droppedInputs = silhouettes
				elseif silhouettes then
					droppedInputs = {droppedInputs, silhouettes}
				end

				localZs[j] = commonFuncs.getEncodings(droppedInputs, model:get(2), sampler, opt.silhouettes, opt.silhouetteInput)
				localLabels[j] = data.labels[j]
			end
			Zs[i] = localZs:float()
			labels[i] = localLabels

		end

		allZs = torch.Tensor(1, opt.nLatents)
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
	else
		allZs = torch.load(experimentResultOutputPath .. '/allZs.t7')
		allLabels = torch.load(experimentResultOutputPath .. '/allLabels.t7')
		data = torch.load(trainDataFiles[1])
	end
	tsne = require 'tsne'
	allZs = allZs:double()
	local y = tsne(allZs, 2, 2000, 5500, 0.15, 20)
	commonFuncs.show_scatter_plot('tSNE-Plot', y, allLabels, #data.category, data.category, experimentResultOutputPath)

end