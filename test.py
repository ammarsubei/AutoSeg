import os

data_dir = '../../color'

allfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

for f in allfiles:
    f = f[:27]
    f = f.split('_')[0] + '/' + f
    print(f)

training_file_list = []
for f in allfiles:
    self.training_file_list.append( ('leftImg8bit/train/' + f + '_leftImg8bit.png', 'gtCoarse/train/' + f + '_gtCoarse_labelIds.png') )