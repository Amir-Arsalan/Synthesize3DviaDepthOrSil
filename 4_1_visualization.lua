require 'paths'
require 'sys'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-inputDir', '', "The directory to read the results from. The input directory contains other directories for each category (e.g. airplane, car etc).")
cmd:option('-outputDir', '', "The directory to output the .ply files")
cmd:option('-resultType', '', "Available options: reconstruction, sampling, interpolation, NN") -- NN is used for the results obtained after running the nearest neighbor experiment on conditional or unconditional samples
cmd:option('-res', 224, "Grid size resolution")
cmd:option('-maskThreshold', 0.5, "The threshold to be used for filtering out the noise using the produced silhouettes")
cmd:text()
opt = cmd:parse(arg or {})

local function getFileNames(thePath)
	-- Returns the file names in a directory

	local handle = assert(io.popen('ls -1v ' .. thePath)) 
	local allFileNames = string.split(assert(handle:read('*a')), '\n')
	for i=1, #allFileNames do allFileNames[i] = thePath .. '/' .. allFileNames[i] end
	
	return allFileNames
end

local function splitTxt(inputStr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputStr, "([^".. sep .."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end


if opt.resultType ~= 'reconstruction' and opt.resultType ~= 'sampling' and opt.resultType ~= 'interpolation' and opt.resultType ~= 'NN' then
	print "reconType argument is invalid. Please give either of the options 'reconstruction', 'sampling' or 'interpolation'"
	os.exit()
end

if opt.inputDir == '' then
	print ('Please specify the results directory')
	os.exit()
end

local outputDirName = '/3DReconstructions'
if opt.outputDir == '' then
	opt.outputDir = opt.inputDir .. outputDirName
else
	if not paths.dirp(opt.outputDir) then
		paths.mkdir(opt.outputDir)
	end
	outputDirName = splitTxt(opt.outputDir, '/')
	outputDirName = outputDirName[#outputDirName]
end

if not paths.filep('depthReconstruction') then
	print ("Make sure you have complied the code in /depth_render_reconstruction/code/depthReconstruction_Ubuntu/depthReconstruction and place the built executable file 'depthReconstruction' in '" .. paths.cwd() .. "' directory")
	os.exit()
end

if not paths.filep('camPosList.txt') then
	print ("Make sure 'camPosList.txt' has been copied into '" .. paths.cwd() .. "' directory")
	os.exit()
end



local dirs = getFileNames(opt.inputDir)
local totalSamples = 0
local ticTotal = torch.tic()
for i=1, #dirs do
	local catName = splitTxt(dirs[i], '/')
	local catSamples = 0
	catName = catName[#catName]
	if catName ~= outputDirName then
		print ("==> Fusing the generated depth maps and silhouettes to obtain final point cloud reconstructions for the category: " .. catName)
		local subDirs = getFileNames(dirs[i])
		local ticCat = torch.tic()
		for j=1, opt.resultType == 'NN' and #subDirs or 1 do
			catSamples = catSamples + 1
			totalSamples = totalSamples + 1
			local subSubDirs = opt.resultType ~= 'NN' and subDirs or getFileNames(subDirs[j])
			for k=1, #subSubDirs do
				subSubDirs[k] = splitTxt(subSubDirs[k], '/')
				subSubDirs[k] = dirs[i] .. '/' .. (opt.resultType == 'NN' and subSubDirs[k][#subSubDirs[k] - 1] .. '/' or '') .. subSubDirs[k][#subSubDirs[k]]
				local splitDirNames = splitTxt(subSubDirs[k], '/')
				if string.match(splitDirNames[#splitDirNames], 'nearest') then
					nearestRecon = true
					subSubSubDirs = getFileNames(getFileNames(subDirs[j])[1])
					for l=1, #subSubSubDirs do
						if string.match(subSubSubDirs[l], 'nearestRecon') then
							nearestReconDir = subSubSubDirs[l]
						end
					end
				else
					nearestRecon = false
				end
				for l=1, nearestRecon and 2 or 1 do
					local reconDirName = (opt.resultType == 'NN' and splitDirNames[#splitDirNames - 1] or '') ..  '/' .. splitDirNames[#splitDirNames] .. (l == 2 and '/nearestRecon' or '')
					paths.mkdir(opt.outputDir .. '/' .. catName .. '/' .. reconDirName)
					os.execute("./depthReconstruction -input '" .. (l == 1 and subSubDirs[k] or nearestReconDir) .. "' -output '" .. opt.outputDir .. '/' .. catName .. '/' .. reconDirName .. "' -resolution " .. opt.res .. " -" .. (opt.resultType ~= 'NN' and opt.resultType or 'reconstruction') .. " -mask " .. opt.maskThreshold .. " >/dev/null 2>&1")
				end
			end
		end
		print(string.format("==> Done with getting 3D reconstruction for %s. No of samples: %d. Time took: %.1f minutes", catName, catSamples, torch.toc(ticCat)/60))
		sys.sleep(5)
	end
end
print(string.format("==> Done with creating all 3D reconstructions. No of total samples: %d. Total time: %.1f minutes", totalSamples, torch.toc(ticTotal)/60))