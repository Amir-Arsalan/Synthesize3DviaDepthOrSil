#!~/torch/install/bin/th

require 'torch'
require 'image'
require 'paths'
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
allData = {}
if not opt.forwardPassType == 'reconAndPC' then
	trainDataFiles = commonFuncs.randPermTableContents(trainDataFiles)
else
	allData[1] = trainDataFiles
	allData[2] = validationDataFiles
	allData[3] = testDataFiles
end

local localExpDirName
localExpDirName = (opt.expType == 'random' or opt.expType == 'conditionalSample') and 'samples' or opt.expType == 'interpolate' and 'interpolation' or opt.expType == 'forwardPass' and 'forwardPass'



local storagePath = string.format('%s/experiments/epoch%d/', paths.cwd() ..'/' .. opt.expDirName, opt.fromEpoch)
modelPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/model.t7', paths.cwd() ..'/' ..opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, opt.fromEpoch)
meanLogVarPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/mean_logvar.t7', paths.cwd() ..'/' ..opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, opt.fromEpoch)
if not path.exists(meanLogVarPath) then
    meanLogVarPath = string.format('%s/save-Latents_%d-BS_%d-Ch_%d-lr_%.5f/epoch%d/man_logvar.t7', paths.cwd() ..'/' ..opt.expDirName, opt.nLatents, opt.batchSize, opt.nCh, opt.lr, opt.fromEpoch)
end

local data, modelTest, gMod
gMod = torch.load(modelPath)
cudnn.convert(gMod, nn)
gMod = gMod:cuda()
cudnn.convert(gMod, cudnn)
gMod:evaluate()
print ''

if opt.expType == 'random' then
	print ("==> Generating random samples")
	print ("==> Configurations - Loss: " .. opt.lossText .. ", expDirName: " .. opt.expDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size Change Epoch: " .. opt.batchSizeChangeEpoch .. ", Batch Size Change: " .. opt.batchSizeChange .. ", Target Batch Size: " .. opt.targetBatchSize .. ", No. Output Channels: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", Network Type: " .. opt.modelType .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', dropVPs: ' .. (opt.dropVPs and "True" or "False") .. ', notDropVP: ' .. opt.notDropVP .. ', silhouettes: ' .. (opt.silhouettes and "True" or "False") .. ', onlySilhouettes: ' .. (opt.onlySilhouettes and "True" or "False") .. ', unknownVPs: ' .. (opt.unknownVPs and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False"))
	if opt.conditional then
		data = torch.load(trainDataFiles[1])
	end
	local sampleZembeddings = torch.load(meanLogVarPath)
	sampleManifold.sample(opt.sampleType, opt.sampleCategory, opt.canvasHW, opt.nSamples, data, gMod:get(1), '', storagePath, opt.mean, opt.var, opt.nLatents, opt.gpu, 224, opt.numVPs, opt.fromEpoch, false, opt.VPsTogether, opt.mixVPs, opt.testPhase, opt.loss, opt.modelType, opt.tanh, opt.dropVPs, opt.notDropVP, opt.silhouettes, opt.onlySilhouettes, true, sampleZembeddings, opt.unknownVPs, opt.conditional, opt.expType)

elseif opt.expType == 'interpolate' then
	print ("==> Doing interpolation")
	print ("==> Configurations - Loss: " .. opt.lossText .. ", expDirName: " .. opt.expDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size Change Epoch: " .. opt.batchSizeChangeEpoch .. ", Batch Size Change: " .. opt.batchSizeChange .. ", Target Batch Size: " .. opt.targetBatchSize .. ", No. Output Channels: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", Network Type: " .. opt.modelType .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', dropVPs: ' .. (opt.dropVPs and "True" or "False") .. ', notDropVP: ' .. opt.notDropVP .. ', silhouettes: ' .. (opt.silhouettes and "True" or "False") .. ', onlySilhouettes: ' .. (opt.onlySilhouettes and "True" or "False") .. ', unknownVPs: ' .. (opt.unknownVPs and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False"))
	data = torch.load(trainDataFiles[1])
	sampleManifold.sample(opt.sampleType, opt.sampleCategory, opt.canvasHW, opt.nSamples, data, gMod:get(1), '', storagePath, opt.mean, opt.var, opt.nLatents, opt.gpu, 224, opt.numVPs, opt.fromEpoch, false, opt.VPsTogether, opt.mixVPs, opt.testPhase, opt.loss, opt.modelType, opt.tanh, opt.dropVPs, opt.notDropVP, opt.silhouettes, opt.onlySilhouettes, true, sampleZembeddings, opt.unknownVPs, opt.conditional, opt.expType)

elseif opt.expType == 'forwardPass' then
	print ("==> Doing forward pass for the '" .. opt.forwardPassType .. "' experiment")
	print ("==> Configurations - Loss: " .. opt.lossText .. ", expDirName: " .. opt.expDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size Change Epoch: " .. opt.batchSizeChangeEpoch .. ", Batch Size Change: " .. opt.batchSizeChange .. ", Target Batch Size: " .. opt.targetBatchSize .. ", No. Output Channels: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", Network Type: " .. opt.modelType .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', dropVPs: ' .. (opt.dropVPs and "True" or "False") .. ', notDropVP: ' .. opt.notDropVP .. ', silhouettes: ' .. (opt.silhouettes and "True" or "False") .. ', onlySilhouettes: ' .. (opt.onlySilhouettes and "True" or "False") .. ', unknownVPs: ' .. (opt.unknownVPs and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False"))

	if opt.forwardPassType == 'silhouettes' then
		storagePath = storagePath .. '/silhouettes'
		local silTensor = commonFuncs.loadExtraData('ExtraData/silhouettes', opt.forwardPassType, opt.numVPs)
		for i=1, silTensor:size(1) do
			silTensor = silTensor:cuda()
			local tempTensor = torch.cat(silTensor[{{i}}], silTensor[{{i}}], 1)
			local recon = unpack(gMod:get(1):forward(tempTensor))
			paths.mkdir(storagePath .. '/model' .. i .. '-silhouettes/mask')
			image.save(storagePath .. '/model' .. i .. '-silhouettes/mask/original.png', tempTensor[1][1])
			for k=1, recon[1]:size(2) do
				image.save(storagePath .. '/model' .. i .. '-silhouettes/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
				image.save(storagePath .. '/model' .. i .. '-silhouettes/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])
			end
		end
	elseif opt.forwardPassType == 'nyud' then
		storagePath = {storagePath, storagePath}
		storagePath[1] = storagePath[1] .. '/nyud/chair'
		storagePath[2] = storagePath[2] .. '/nyud/bottle'
		local dataPaths = {'ExtraData/nyud/chair', 'ExtraData/nyud/bottle'}
		local dirText = {'chair', 'bottle'}
		for j=1, #storagePath do
			local tensorTable = commonFuncs.loadExtraData(dataPaths[j], opt.forwardPassType, opt.numVPs)
			if opt.onlySilhouettes then
				local silTensor = tensorTable[2]
				for i=1, tensorTable[1]:size(1) do
					silTensor = silTensor:cuda()
					local tempTensor = torch.cat(silTensor[{{i}}], silTensor[{{i}}], 1)
					local recon = unpack(gMod:get(1):forward(tempTensor))
					paths.mkdir(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask')
					image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/originalMask.png', tempTensor[1][1])
					image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/originalDepth.png', tensorTable[1][i][1])
					image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/originalRGB.png', tensorTable[3][i][1])
					for k=1, recon[1]:size(2) do
						image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
						image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])
					end
				end
			elseif not opt.silhouettes then
				local depthTensor = tensorTable[1]
				for i=1, tensorTable[1]:size(1) do
					depthTensor = depthTensor:cuda()
					local tempTensor = torch.cat(depthTensor[{{i}}], depthTensor[{{i}}], 1)
					local recon = unpack(gMod:get(1):forward(tempTensor))
					paths.mkdir(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask')
					image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/original.png', tempTensor[1][1])
					for k=1, recon[1]:size(2) do
						image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
						image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])
					end
				end
			else
				local depthMaskTensor = {tensorTable[1], tensorTable[2]}
				for i=1, depthMaskTensor[1]:size(1) do
					depthMaskTensor[1] = depthMaskTensor[1]:cuda()
					depthMaskTensor[2] = depthMaskTensor[2]:cuda()
					local tempTensor = {torch.cat(depthMaskTensor[1][{{i}}], depthMaskTensor[1][{{i}}], 1), torch.cat(depthMaskTensor[2][{{i}}], depthMaskTensor[2][{{i}}], 1)}
					local recon = unpack(gMod:get(1):forward(tempTensor))
					paths.mkdir(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask')
					image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/original.png', tempTensor[1][1][1])
					image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/original.png', tempTensor[2][1][1])
					for k=1, recon[1]:size(2) do
						image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
						image.save(storagePath[j] .. '/model' .. i .. '-' .. dirText[j] .. '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])
					end
				end
			end
		end
	elseif opt.forwardPassType == 'completion' then
		-- Given a 3D model with missing parts, reconstruct the ground-truth
		storagePath = storagePath .. '/completion'
		local depthTensor = commonFuncs.loadExtraData('ExtraData/completion', opt.forwardPassType, opt.numVPs)
		local maskInputs = depthTensor:clone()
        maskInputs[maskInputs:gt(0)] = 1

		depthTensor = depthTensor:cuda()
		for i=1, depthTensor:size(1) do
			local tempTensor = depthTensor[{{i}}]:clone()
			tempTensor = torch.cat(tempTensor, tempTensor, 1)
			recon = unpack(gMod:get(1):forward(tempTensor))

			paths.mkdir(storagePath .. '/model' .. i .. '-complete' .. '/mask')
			for k=1, recon[1]:size(2) do
				image.save(storagePath .. '/model' .. i .. '-complete' ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
				image.save(storagePath .. '/model' .. i .. '-complete' ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])

				image.save(storagePath .. '/model' .. i .. '-complete' ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', depthTensor[i][k])
				image.save(storagePath .. '/model' .. i .. '-complete' ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', maskInputs[i][k])
			end
		end

	elseif opt.forwardPassType == 'reconstruction' then

		storagePath = storagePath .. '/reconstruction'
		data = torch.load(testDataFiles[1])

		local indicesToChoose = torch.randperm(data.dataset:size(1))
		indicesToChoose = indicesToChoose[{{1, opt.nReconstructions}}]:long()
		local inputs = data.dataset:index(1, indicesToChoose)
		inputs = inputs:cuda()
		local maskInputs = inputs:clone()
		maskInputs[maskInputs:gt(0)] = 1

		if opt.onlySilhouettes or opt.silhouettes then
			-- inputs = nil
			-- inputs = maskInputs
			inputs = {inputs, maskInputs}
		-- elseif opt.silhouettes then
		-- 	inputs = {inputs, maskInputs}
		end

		local targetClassHotVec, labels
		labels = data.labels:index(1, indicesToChoose)
		if opt.conditional then
		    targetClassHotVec = torch.zeros(opt.nReconstructions, #data.category)
		    for l=1, labels:nElement() do
		        targetClassHotVec[l][labels[l]] = 1
		    end
		    targetClassHotVec = targetClassHotVec:type(not opt.silhouettes and inputs:type() or inputs[1]:type())
		end

		local catLabels = {}
		for i=1, opt.nReconstructions do catLabels[i] = data.category[labels[i]] end


		droppedInputs = commonFuncs.dropInputVPs(inputs, opt.dropVPs and opt.notDropVP or nil, true, nil, nil, opt.unknownVPs, nil, targetClassHotVec)
		if opt.onlySilhouettes then droppedInputs = droppedInputs[2] end

		for i=1, opt.nReconstructions do
			local networkInput
			if type(droppedInputs) == 'table' then
				networkInput = {}
				for j=1, #droppedInputs do
					local temp = droppedInputs[j][{{i}}]:clone()
					temp = torch.cat(temp, temp, 1)
					networkInput[j] = temp
				end
			else
				networkInput = droppedInputs[{{i}}]:clone()
				networkInput = torch.cat(networkInput, networkInput, 1)
			end

			recon = unpack(gMod:get(1):forward(networkInput))

			paths.mkdir(storagePath .. '/model' .. i .. '-' .. catLabels[i] .. '/mask')

			for k=1, recon[1]:size(2) do
				image.save(storagePath .. '/model' .. i .. '-' .. catLabels[i] ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
				image.save(storagePath .. '/model' .. i .. '-' .. catLabels[i] ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])

				image.save(storagePath .. '/model' .. i .. '-' .. catLabels[i] ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', type(inputs) ~= 'table' and inputs[i][k] or inputs[1][i][k])
				image.save(storagePath .. '/model' .. i .. '-' .. catLabels[i] ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', type(inputs) ~= 'table' and maskInputs[i][k] or inputs[2][i][k])
			end

		end

	elseif opt.forwardPassType == 'reconAndPC' then -- reconAndPC --> Reconstruction and getting the point cloud files

		local data = torch.load(allData[2][1])
		local tempData = torch.load(allData[3][1])
		data.dataset = torch.cat(data.dataset, tempData.dataset, 1)
		data.labels = torch.cat(data.labels, tempData.labels, 1)
		data.category = commonFuncs.tableConcat(data.category, tempData.category)
		tempData.dataset = nil
		tempData = nil

		storagePath = storagePath .. 'reconstruction/test'
		local counter = 1

		local indicesToChoose = commonFuncs.findEligibleCatsIndices(data)
		for i=1, indicesToChoose:size(1) do
			local input = data.dataset[{{indicesToChoose[i]}}]:cuda()
			input = torch.cat(input, input, 1)
			local maskInput = input:clone()
			maskInput[maskInput:gt(0)] = 1
			if opt.onlySilhouettes or opt.silhouettes then
				input = {input, maskInput}
			end
			local label = data.labels[indicesToChoose[i]]
			local catLabel = data.category[label]
			droppedInputs = commonFuncs.dropInputVPs(input, opt.dropVPs and opt.notDropVP or nil, true, nil, nil, opt.unknownVPs, nil)
			if opt.onlySilhouettes then droppedInputs = droppedInputs[2] end

			local networkInput
			if type(droppedInputs) == 'table' then
				networkInput = {}
				for j=1, #droppedInputs do
					local temp = droppedInputs[j][{{1}}]:clone()
					temp = torch.cat(temp, temp, 1)
					networkInput[j] = temp
				end
			else
				networkInput = droppedInputs[{{1}}]:clone()
				networkInput = torch.cat(networkInput, networkInput, 1)
			end

			recon = unpack(gMod:get(1):forward(networkInput))

			paths.mkdir(storagePath .. '/model' .. i .. '-' .. catLabel .. '/mask')

			for k=1, recon[1]:size(2) do
				image.save(storagePath .. '/model' .. i .. '-' .. catLabel ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[1][1][k])
				image.save(storagePath .. '/model' .. i .. '-' .. catLabel ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-rec.png', recon[2][1][k])

				image.save(storagePath .. '/model' .. i .. '-' .. catLabel ..  '/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', type(input) ~= 'table' and input[1][k] or input[1][1][k])
				image.save(storagePath .. '/model' .. i .. '-' .. catLabel ..  '/mask/file' .. i .. '-img' .. i .. '-' .. k-1 .. '-or.png', type(input) ~= 'table' and maskInput[1][k] or input[2][1][k])
			end
			if i == 4 then collectgarbage() end
		end

	elseif opt.forwardPassType == 'calcErr' then
		local data = torch.load(trainDataFiles[1])
		local cr1 = nn.AbsCriterion()
        local cr2 = nn.AbsCriterion()
        local criterion = nn.ParallelCriterion():add(cr1,1):add(cr2, 1)
        criterion = criterion:cuda()
		
		local counter = 0
		local depthErr = 0
		local maskErr = 0
		for j=2, 5 do
			for i=1, data.dataset:size(1) do
				local input = data.dataset[{{i}}]:cuda()
				input = torch.cat(input, input, 1)
				local maskInput = input:clone()
				maskInput[maskInput:gt(0)] = 1
				-- if opt.onlySilhouettes or opt.silhouettes then
				-- 	input = {input, maskInput}
				-- end
				droppedInputs = commonFuncs.dropInputVPs(not opt.onlySilhouettes and input or maskInput, opt.dropVPs and opt.notDropVP or nil, false, nil, nil, opt.unknownVPs, nil)
				input = {input[{{1}}], maskInput[{{1}}]}
				-- if opt.onlySilhouettes then droppedInputs = droppedInputs[2] end

				-- local networkInput = droppedInputs[{{1}}]:clone()
				-- networkInput = torch.cat(networkInput, networkInput, 1)

				recon = unpack(gMod:get(1):forward(droppedInputs))
				recon = {recon[1][{{1}}], recon[2][{{1}}]}

				criterion:forward(recon, input)
				depthErr = depthErr + criterion.criterions[1].output
				maskErr = maskErr + criterion.criterions[2].output

				if i == 4 then collectgarbage() end
				counter = counter + 1
			end
			data = torch.load(trainDataFiles[j])
		end

		print (string.format('The train depth error is %.4f and mask error is %.4f for %d samples', depthErr/counter, maskErr/counter, counter))

		local data = torch.load(validationDataFiles[1])
		local tempData = torch.load(testDataFiles[1])
		data.dataset = torch.cat(data.dataset, tempData.dataset, 1)
		data.labels = torch.cat(data.labels, tempData.labels, 1)
		data.category = commonFuncs.tableConcat(data.category, tempData.category)
		tempData.dataset = nil
		tempData = nil

		local counter = 0
		local depthErr = 0
		local maskErr = 0
		for i=1, data.dataset:size(1) do
			local input = data.dataset[{{i}}]:cuda()
			input = torch.cat(input, input, 1)
			local maskInput = input:clone()
			maskInput[maskInput:gt(0)] = 1
			-- if opt.onlySilhouettes or opt.silhouettes then
			-- 	input = {input, maskInput}
			-- end
			droppedInputs = commonFuncs.dropInputVPs(not opt.onlySilhouettes and input or maskInput, opt.dropVPs and opt.notDropVP or nil, false, nil, nil, opt.unknownVPs, nil)
			input = {input[{{1}}], maskInput[{{1}}]}
			-- if opt.onlySilhouettes then droppedInputs = droppedInputs[2] end

			-- local networkInput = droppedInputs[{{1}}]:clone()
			-- networkInput = torch.cat(networkInput, networkInput, 1)

			recon = unpack(gMod:get(1):forward(droppedInputs))
			recon = {recon[1][{{1}}], recon[2][{{1}}]}

			criterion:forward(recon, input)
			depthErr = depthErr + criterion.criterions[1].output
			maskErr = maskErr + criterion.criterions[2].output
			counter = counter + 1

			if i == 4 then collectgarbage() end
		end
			
		print (string.format('The validation depth error is %.4f and mask error is %.4f for %d samples', depthErr/counter, maskErr/counter, counter))


	end
elseif opt.expType == 'NNs' then

	print ("==> Doing the nearest neighbors experiment")
	print ("==> Configurations - Loss: " .. opt.lossText .. ", expDirName: " .. opt.expDirName .. ", No. Latents: " .. opt.nLatents .. ", Batch Size: " .. opt.batchSize .. ", Batch Size Change Epoch: " .. opt.batchSizeChangeEpoch .. ", Batch Size Change: " .. opt.batchSizeChange .. ", Target Batch Size: " .. opt.targetBatchSize .. ", No. Output Channels: " .. opt.nCh .. ", LR Decay: " .. opt.lrDecay .. ", Learning Rate: " .. opt.lr .. ", InitialLR: " .. opt.initialLR .. ", Network Type: " .. opt.modelType .. ", Tanh: " .. (opt.tanh and "True" or "False") .. ', dropVPs: ' .. (opt.dropVPs and "True" or "False") .. ', notDropVP: ' .. opt.notDropVP .. ', silhouettes: ' .. (opt.silhouettes and "True" or "False") .. ', onlySilhouettes: ' .. (opt.onlySilhouettes and "True" or "False") .. ', unknownVPs: ' .. (opt.unknownVPs and "True" or "False") .. ', conditional: ' .. (opt.conditional and "True" or "False"))
	storagePath = storagePath .. 'nearestNeighbors'
	-- Doing nearest neighbors in Z space
	local samplesPath = 'ExtraData/samples'
	-- local samplesPath = '/home/amir/Desktop/samples/Unconditional'
	if opt.unknownVPs then
		if opt.onlySilhouettes then
			samplesPath = samplesPath .. '/SingleVPNet-Silhouettes/empirical'
		else
			samplesPath = samplesPath .. '/SingleVPNet-Depth/empirical'
		end
	elseif not opt.dropVPs then
		if opt.onlySilhouettes then
			samplesPath = samplesPath .. '/AllVPNet-Silhouettes/empirical'
		else
			samplesPath = samplesPath .. '/AllVPNet-Depth/empirical'
		end
	else
		if opt.onlySilhouettes then
			samplesPath = samplesPath .. '/DropoutNet-Silhouettes/empirical'
		else
			samplesPath = samplesPath .. '/DropoutNet-Depth/empirical'
		end
	end


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
			local droppedInputs = commonFuncs.dropInputVPs(data.dataset[{{j}}], opt.notDropVP, false, numDropVPs, dropIndices, opt.unknownVPs)

			local maskInputs = droppedInputs:clone()
			maskInputs[maskInputs:gt(0)] = 1

			if onlySilhouettes then
				droppedInputs = nil
				droppedInputs = maskInputs
			elseif silhouettes then
				droppedInputs = {droppedInputs, maskInputs}
			end

			localZs[j] = commonFuncs.getEncodings(droppedInputs, gMod:get(1):get(2), sampler, opt.silhouettes, opt.onlySilhouettes)
			localLabels[j] = data.labels[j]
		end
		Zs[i] = localZs:float()
		labels[i] = localLabels

	end

	-- local allZs = torch.Tensor(1, opt.nLatents)
	-- local allLabels = torch.Tensor(1)
	-- for i=1, #Zs do
	-- 	Zs[i] = Zs[i]:float()
	-- 	labels[i] = labels[i]:float()
	-- 	allZs = torch.cat(allZs, Zs[i], 1)
	-- 	allLabels = torch.cat(allLabels, labels[i], 1)
	-- end
	-- allZs = allZs[{{2, allZs:size(1)}}]
	-- allLabels = allLabels[{{2, allLabels:size(1)}}]


	local samplesDirs = commonFuncs.getFileNames(samplesPath)
	local numOfSamplestoVisualize = commonFuncs.getNumOfSamplesToViz(samplesDirs)
	local depthMapsTensor = torch.Tensor(numOfSamplestoVisualize, opt.numVPs, 224, 224)
	local masksTensor = torch.Tensor(numOfSamplestoVisualize, opt.numVPs, 224, 224)
	local sampleCounter = 1
	for i=1, #samplesDirs do
		local samplesToVisualize = commonFuncs.getFileNames(samplesDirs[i], 'viz.txt')
		if #samplesToVisualize == 1 then
			-- Fill the tensor with depth map images
			local depthMapFilesToRead = commonFuncs.getFileNames(samplesDirs[i], '.png')
			local maskFilesToRead
			maskFilesToRead = commonFuncs.getFileNames(samplesDirs[i] .. '/mask', '.png')
			f = assert(io.open(samplesToVisualize[1], 'r'))
			for line in f:lines() do
				local rowNum, colNum = commonFuncs.commaSeparatedStrToTable(line, true)
				for j=1, opt.numVPs do
					local tempDepthImg = image.load(depthMapFilesToRead[j])[1]
					depthMapsTensor[sampleCounter][j] = tempDepthImg[{{(rowNum-1)*224+1, rowNum*224}, {(colNum-1)*224+1, colNum*224}}]
				 	local tempMaskImg = image.load(maskFilesToRead[j])[1]
				 	masksTensor[sampleCounter][j] = tempMaskImg[{{(rowNum-1)*224+1, rowNum*224}, {(colNum-1)*224+1, colNum*224}}]
				end
				sampleCounter = sampleCounter + 1
			end
			f:close()
		end
	end

	-- local samplesDirs = commonFuncs.getFileNames(samplesPath)
	local samplesZs = torch.Tensor(depthMapsTensor:size(1), opt.nLatents):cuda()
	for i=1, depthMapsTensor:size(1) do
		local inputs = not opt.onlySilhouettes and depthMapsTensor[{{i}}] or {depthMapsTensor[{{i}}], masksTensor[{{i}}]}
		local droppedInputs = commonFuncs.dropInputVPs(inputs, opt.dropVPs and opt.notDropVP or nil, true, numDropVPs, dropIndices, opt.unknownVPs)
		droppedInputs = not opt.onlySilhouettes and torch.cat(droppedInputs, droppedInputs, 1) or {torch.cat(droppedInputs[1], droppedInputs[1], 1), torch.cat(droppedInputs[2], droppedInputs[2], 1)}
		droppedInputs = not opt.onlySilhouettes and droppedInputs:cuda() or droppedInputs[2]:cuda()
		-- if opt.onlySilhouettes then droppedInputs = droppedInputs[2] end
		samplesZs[i] = commonFuncs.getEncodings(droppedInputs, gMod:get(1):get(2), sampler, opt.silhouettes, opt.onlySilhouettes)
	end
	

	samplesZs = samplesZs:float()
	local minZFileNo = {}
	local minZIndex = {}
	local minCatName = {}
	for i=1, samplesZs:size(1) do
		local randomSamplePath = storagePath .. '/randomSample' .. i .. '/original/'
		paths.mkdir(randomSamplePath .. 'mask')
		for k=1, depthMapsTensor:size(2) do
			image.save(randomSamplePath .. 'VP-' .. k-1 .. '.png', depthMapsTensor[i][k])
			image.save(randomSamplePath .. '/mask/VP-' .. k-1 .. '.png', masksTensor[i][k])
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
					minCatName[i] = data.category[labels[j][k]]
				end
			end
		end
		minLabel = nil
	end
	data = nil
	collectgarbage()

	local lastFileID = minZFileNo[1]
	data = torch.load(trainDataFiles[minZFileNo[1]])
	for i=1, #minZFileNo do
		local nearestPath = storagePath .. '/randomSample' .. i .. '/nearest' .. '-' .. minCatName[i]
		paths.mkdir(nearestPath .. '/mask')
		if lastFileID ~= minZFileNo[i] then
			lastFileID = minZFileNo[i]
			data = nil
			collectgarbage()
			data = torch.load(trainDataFiles[minZFileNo[i]])
		end
		local maskInputs = data.dataset[minZIndex[i]]:clone()
		maskInputs[maskInputs:gt(0)] = 1
		for k=1, depthMapsTensor:size(2) do
			image.save(nearestPath .. '/VP-' .. k-1 .. '.png', data.dataset[minZIndex[i]][k])
			image.save(nearestPath .. '/mask/VP-' .. k-1 .. '.png', maskInputs[k])
		end

	end
	data = nil
	collectgarbage()
elseif opt.expType == 'tSNE' then

	storagePath = storagePath .. 'tSNE'
	local allZs, allLabels, data
	if not paths.filep(storagePath .. '/allZs.t7') then
		paths.mkdir(storagePath)
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
				local droppedInputs = commonFuncs.dropInputVPs(data.dataset[{{j}}], opt.notDropVP, false, numDropVPs, dropIndices, opt.unknownVPs)

				local maskInputs = droppedInputs:clone()
				maskInputs[maskInputs:gt(0)] = 1

				if onlySilhouettes then
					droppedInputs = nil
					droppedInputs = maskInputs
				elseif silhouettes then
					droppedInputs = {droppedInputs, maskInputs}
				end

				localZs[j] = commonFuncs.getEncodings(droppedInputs, gMod:get(1):get(2), sampler, opt.silhouettes, opt.onlySilhouettes)
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
		torch.save(storagePath .. '/allZs.t7', allZs)
		torch.save(storagePath .. '/allLabels.t7', allLabels)
	else
		allZs = torch.load(storagePath .. '/allZs.t7')
		allLabels = torch.load(storagePath .. '/allLabels.t7')
		data = torch.load(trainDataFiles[1])
	end
	tsne = require 'tsne'
	allZs = allZs:double()
	local y = tsne(allZs, 2, 2000, 5500, 0.15, 20)
	commonFuncs.show_scatter_plot('tSNE-Plot', y, allLabels, #data.category, data.category, storagePath)

end