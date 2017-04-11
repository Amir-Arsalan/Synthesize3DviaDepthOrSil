require 'paths'
require 'sys'


--[[
To visualize the results of the neural network model you first need to compile

--]]

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-inputDir', '', "The directory to read the data from. The input directory contains other directories for each category. The category directories contain reconstructions")
cmd:option('-outputDir', '', "The directory to output the .ply files")
cmd:option('-mode', '', "Available options: reconstruction, sampling, interpolation")
cmd:option('-res', 224, "Grid size resolution")
cmd:option('-maskThreshold', 0.01, "The threshold to use the silhouettes to filter out the noise")
cmd:option('-conditional', 0, "Set to 1 for visualizing samples obtained from a conditional model")
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

if opt.mode ~= 'reconstruction' and opt.mode ~= 'sampling' and opt.mode ~= 'interpolation' then
	print "reconType argument is invalid. Please give either of the options 'reconstruction', 'sampling' or 'interpolation'"
	os.exit()
end
if opt.inputDir == '' then
	print "Please specify the input directory"
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


local dirs = getFileNames(opt.inputDir)
local ticTotal = torch.tic()
for i=1, #dirs do
	local catName = splitTxt(dirs[i], '/')
	catName = catName[#catName]
	if catName ~= outputDirName then
		print ("==> Generating 3D reconstructions for the category: " .. catName)
		local subDirs = getFileNames(dirs[i])
		local ticCat = torch.tic()
		for j=1, #subDirs do
			subDirs[j] = splitTxt(subDirs[j], '/')
			subDirs[j] = dirs[i] .. '/' .. subDirs[j][#subDirs[j]]
			local modelName = splitTxt(subDirs[j], '/')
			modelName = modelName[#modelName]
			paths.mkdir(opt.outputDir .. '/' .. catName .. '/' .. modelName)
			os.execute("./depthReconstruction -input '" .. subDirs[j] .. "' -output '" .. opt.outputDir .. '/' .. catName .. '/' .. modelName .. "' -resolution " .. opt.res .. " -" .. opt.mode .. " -mask " .. opt.maskThreshold .. " >/dev/null 2>&1")
		end
		print(string.format("==> Done with getting 3D reconstruction for %s. Time took: %.1f minutes", catName, torch.toc(ticCat)/60)
		sys.sleep(5)
	end
	print(string.format("==> Done with getting all 3D reconstruction. Total time: %.1f minutes", torch.toc(ticTotal)/60)
end